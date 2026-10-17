p# RetailPulse

**AI-Powered Customer Analytics & Demand Forecasting** — Zidio Internship Project

RetailPulse helps retailers understand customer behaviour, forecast product demand, and plan inventory using machine learning and MLOps practices. It is built on the **Online Retail II (UCI)** dataset and delivered two ways:

| Track | What it is | Where |
|---|---|---|
| **1. Dashboard-based data analysis** | EDA and business insights, explored in a BI tool (Power BI / Tableau / Looker Studio) using the exported CSVs | `outputs/powerbi/*.csv`, `outputs/eda/*.png` |
| **2. Streamlit app** | A four-page interactive dashboard on top of the ML pipeline | `app/streamlit_app.py` |

---

## Workflow

The pipeline follows the 12-step project procedure. Steps 1–10 run with one command (`run_pipeline.py`).

| # | Step | Code |
|---|---|---|
| 1–3 | Collect, import, load (+ the custom row, see below) | `src/data_loader.py` |
| 4 | Data cleaning | `src/cleaning.py` |
| 5 | Feature engineering | `src/features.py` |
| 6 | EDA | `src/eda.py` |
| 7 | Customer segmentation (RFM + K-Means) | `src/rfm.py` |
| 8 | Demand forecasting (Prophet, LSTM, Hybrid) | `src/forecasting.py` |
| 9 | Churn prediction (+ Optuna tuning) | `src/churn.py` |
| 10 | Inventory optimisation | `src/inventory.py` |
| 11 | Dashboards | `app/`, `outputs/powerbi/` |
| 12 | Deployment | `Dockerfile`, Streamlit Cloud |

Also included: drift detection (`src/drift.py`), MLflow experiment tracking, and an Airflow DAG (`dags/retailpulse_dag.py`).

## The custom row

The brief asks for the data not to be used exactly as downloaded. `src/config.py` defines one extra transaction (`CUSTOM_ROW`: invoice 999001, customer 99999, 12 × "RETAILPULSE CUSTOM GIFT SET" at 4.95). It is appended at load time and flagged `is_custom_row=True`, so it can be traced through cleaning, features, segmentation and the dashboard. Edit the values to your own record. `tests/` checks that it is added and survives cleaning.

## Quick start

```bash
pip install -r requirements.txt

# 1. Get the data: download from Kaggle and put the file in data/raw/ (csv, xlsx or zip)
#    https://www.kaggle.com/datasets/cgrymn/online-retail-ii-uci-dataset
#    (or let the script try kagglehub, which needs internet and a Kaggle login)

# 2. Run steps 1-10
python run_pipeline.py                         # 2010-2011 sheet (the period in the report)
python run_pipeline.py --years both            # both years
python run_pipeline.py --path data/raw/online_retail_II.xlsx

# 3. Launch the dashboard
streamlit run app/streamlit_app.py

# 4. Experiment tracking
mlflow ui --backend-store-uri sqlite:///outputs/mlflow.db
```

`python run_pipeline.py --sample` runs everything on synthetic data to test the setup. **Never use `--sample` results in your report.**

## Streamlit dashboard

| Page | Contents |
|---|---|
| Sales | Revenue, orders, customers, AOV; monthly trend; top products and countries; day/hour patterns; date and country filters |
| Customer | RFM segments, churn-risk distribution, churn drivers, high-risk watchlist (downloadable), customer lookup |
| Forecast | Actual vs forecast with prediction range, model comparison table, 30-day forecast download |
| Inventory | Safety stock, reorder point, EOQ for the top 50 products; stock status; adjustable assumed stock |

## Models

- **Segmentation:** quintile RFM scores mapped to named segments, plus K-Means on log-scaled RFM with *k* chosen by silhouette score.
- **Forecasting:** seasonal-naive baseline, gradient boosting on lag/calendar features, Prophet, LSTM, and a hybrid average. The best model on a 30-day hold-out (lowest WAPE) produces the forecast. Models that aren't installed are skipped.
- **Churn:** a customer churns if they make no purchase in the 90 days after a cut-off date. Features use only data before the cut-off, so there is no leakage. Logistic regression, random forest, gradient boosting and an Optuna-tuned gradient boosting are compared by ROC-AUC. Recency is tightly linked to this churn definition, so it dominates feature importance (as in the report).
- **Inventory:** safety stock = z × σ × √lead time; reorder point = demand over lead time + safety stock; EOQ from order and holding cost. Parameters are in `src/config.py`.

## Data notes and limitations

- Cleaning removes duplicates, cancellations (kept separately for return-rate features), non-positive quantity or price, and non-product codes (postage, fees). Rows without a Customer ID stay in sales totals but are excluded from customer analysis.
- The dataset has **no stock-on-hand column**. Inventory status labels use an *assumed* stock level (slider in the dashboard); reorder points and EOQ come from real sales.
- Forecast accuracy and churn metrics depend on the real data — run the pipeline on the Kaggle file and copy the numbers from `outputs/forecast_metrics.json` and `outputs/churn_metrics.json` into your report.

## Deployment

**Docker**

```bash
python run_pipeline.py            # create outputs/ first
docker build -t retailpulse .
docker run -p 8501:8501 retailpulse
```

**Streamlit Community Cloud:** push the repo to GitHub (including `outputs/*.parquet` and `outputs/*.json`, which the app reads), then create an app with `app/streamlit_app.py` as the entry point.

**Airflow:** copy `dags/retailpulse_dag.py` into your Airflow `dags/` folder and set `PROJECT` to the repo path.

## Track 1: using a BI tool

`run_pipeline.py` writes tidy CSVs to `outputs/powerbi/`: `transactions_clean`, `daily_sales`, `rfm_segments`, `churn_scores`, `forecast`, `inventory_plan`. Load them into Power BI, Tableau or Looker Studio and relate tables on `Customer ID` and `StockCode`. Suggested pages mirror the Streamlit app: Sales overview, Customer segments, Forecast, Inventory. `outputs/eda/` has the static EDA charts.

## Project structure

```
retailpulse/
├── run_pipeline.py          # steps 1-10
├── src/                     # loader, cleaning, features, eda, rfm, forecasting, churn, inventory, drift
├── app/                     # Streamlit app (entry: streamlit_app.py, pages in views/)
├── dags/                    # Airflow DAG
├── tests/                   # pytest
├── data/raw/                # put the Kaggle file here
├── outputs/                 # generated: parquet, json, charts, powerbi CSVs, mlflow.db
├── Dockerfile
└── requirements.txt
```

## Tests

```bash
pytest -q
```

## Roadmap

Real-time ingestion, cloud-native deployment, recommendation system, and automated retraining and alerting (from the project report).
