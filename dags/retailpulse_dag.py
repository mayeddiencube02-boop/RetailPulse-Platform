"""Airflow DAG: refresh data, retrain models and rebuild dashboard inputs every night.

NOTE: written to match the Airflow orchestration described in the project report.
It has not been executed in the build environment (Airflow is not installed there),
so test it in your own Airflow instance. Requires Airflow 2.4+.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT = "/opt/retailpulse"   # change to wherever the repo lives on the Airflow worker

with DAG(
    dag_id="retailpulse_pipeline",
    description="Clean data -> features -> RFM -> forecast -> churn -> inventory -> drift check",
    start_date=datetime(2026, 5, 1),
    schedule="0 2 * * *",
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=10)},
    tags=["retailpulse", "mlops"],
) as dag:
    run = BashOperator(
        task_id="run_pipeline",
        bash_command=f"cd {PROJECT} && python run_pipeline.py --years both",
    )
    test = BashOperator(
        task_id="run_tests",
        bash_command=f"cd {PROJECT} && python -m pytest -q tests",
    )
    run >> test
