"""Central configuration for RetailPulse."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
DATA_RAW.mkdir(parents=True, exist_ok=True)

# Kaggle dataset used in the project brief
KAGGLE_HANDLE = "cgrymn/online-retail-ii-uci-dataset"

# Output artefacts (consumed by the Streamlit dashboard)
CLEAN_PARQUET = OUT / "clean_transactions.parquet"
FEATURES_PARQUET = OUT / "customer_features.parquet"
RFM_PARQUET = OUT / "rfm_segments.parquet"
DAILY_PARQUET = OUT / "daily_sales.parquet"
FORECAST_PARQUET = OUT / "forecast.parquet"
FORECAST_METRICS = OUT / "forecast_metrics.json"
CHURN_PARQUET = OUT / "churn_scores.parquet"
CHURN_METRICS = OUT / "churn_metrics.json"
INVENTORY_PARQUET = OUT / "inventory_plan.parquet"
DRIFT_JSON = OUT / "drift.json"
EDA_DIR = OUT / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)

# Custom row the assignment asks us to add to the dataset.
# It is flagged with is_custom_row=True so it stays traceable end to end.
CUSTOM_ROW = {
    "Invoice": "999001",
    "StockCode": "RP-0001",
    "Description": "RETAILPULSE CUSTOM GIFT SET",
    "Quantity": 12,
    "InvoiceDate": "2011-12-09 12:30:00",
    "Price": 4.95,
    "Customer ID": 99999,
    "Country": "United Kingdom",
}

RANDOM_STATE = 42
FORECAST_HORIZON_DAYS = 30
CHURN_WINDOW_DAYS = 90   # no purchase in the last 90 days of history => churned
SERVICE_LEVEL_Z = 1.65   # ~95% service level
LEAD_TIME_DAYS = 7
ORDER_COST = 50.0
HOLDING_COST_RATE = 0.2

# MLflow experiment tracking (SQLite backend; view with: mlflow ui --backend-store-uri sqlite:///outputs/mlflow.db)
MLFLOW_URI = f"sqlite:///{OUT / 'mlflow.db'}"
