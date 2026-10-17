"""Run the full RetailPulse workflow (steps 1-10) and save outputs for the dashboard.

Usage:
    python run_pipeline.py                      # looks in data/raw, then tries Kaggle download
    python run_pipeline.py --path data/raw/online_retail_II.xlsx --years both
    python run_pipeline.py --sample             # synthetic data, for testing only
"""
import argparse
import logging
import time

import pandas as pd

from src import config
from src.churn import run_churn
from src.cleaning import clean
from src.data_loader import load_data
from src.drift import run_drift
from src.eda import run_eda
from src.features import add_time_features, customer_features, daily_sales
from src.forecasting import run_forecast
from src.inventory import inventory_plan
from src.rfm import rfm_segmentation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("retailpulse")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", help="path to the raw csv/xlsx/zip")
    ap.add_argument("--years", default="2010-2011", choices=["2009-2010", "2010-2011", "both"])
    ap.add_argument("--sample", action="store_true", help="use synthetic data if no real file is found")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--optuna-trials", type=int, default=15)
    a = ap.parse_args()
    t0 = time.time()

    log.info("STEP 1-3  collect / import / load")
    raw = load_data(a.path, a.years, allow_sample=a.sample)

    log.info("STEP 4    clean")
    sales, returns, rep = clean(raw)

    log.info("STEP 5    feature engineering")
    sales = add_time_features(sales)
    feats = customer_features(sales, returns)
    daily = daily_sales(sales)
    sales.to_parquet(config.CLEAN_PARQUET, index=False)
    feats.to_parquet(config.FEATURES_PARQUET, index=False)
    daily.to_parquet(config.DAILY_PARQUET, index=False)

    log.info("STEP 6    EDA")
    run_eda(sales)

    log.info("STEP 7    RFM segmentation")
    rfm, info = rfm_segmentation(feats)
    rfm.to_parquet(config.RFM_PARQUET, index=False)

    log.info("STEP 8    demand forecasting")
    fc, fc_rep = run_forecast(daily, use_mlflow=not a.no_mlflow)
    fc.to_parquet(config.FORECAST_PARQUET, index=False)

    log.info("STEP 9    churn prediction")
    churn, ch_rep = run_churn(sales, returns, feats, n_trials=a.optuna_trials, use_mlflow=not a.no_mlflow)
    churn = churn.merge(rfm[["Customer ID", "Segment", "Cluster_Name"]], on="Customer ID", how="left")
    churn.to_parquet(config.CHURN_PARQUET, index=False)

    log.info("STEP 10   inventory optimisation")
    inv = inventory_plan(sales)
    inv.to_parquet(config.INVENTORY_PARQUET, index=False)
    run_drift(sales)

    # CSV exports for the dashboard-tool track (Power BI / Tableau / Looker Studio)
    pbi = config.OUT / "powerbi"
    pbi.mkdir(exist_ok=True)
    sales.drop(columns=["Customer ID"]).assign(**{"Customer ID": sales["Customer ID"].astype("float")}).to_csv(pbi / "transactions_clean.csv", index=False)
    for name, frame in {"rfm_segments": rfm, "churn_scores": churn, "forecast": fc,
                        "inventory_plan": inv, "daily_sales": daily}.items():
        frame.to_csv(pbi / f"{name}.csv", index=False)
    log.info("CSV exports for Power BI written to %s", pbi)

    log.info("Custom row check: in raw=%s, survived cleaning=%s, in a customer profile=%s",
             bool(raw["is_custom_row"].any()), rep["custom_row_survived_cleaning"],
             bool((feats["Customer ID"] == config.CUSTOM_ROW["Customer ID"]).any()))
    log.info("DONE in %.0fs. Launch the dashboard with:  streamlit run app/streamlit_app.py", time.time() - t0)


if __name__ == "__main__":
    main()
