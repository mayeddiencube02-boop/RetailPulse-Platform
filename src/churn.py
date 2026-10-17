"""Step 9: churn prediction.

Leakage-safe design:
  * Pick a cut-off date = (last date - CHURN_WINDOW_DAYS).
  * Features are built ONLY from transactions before the cut-off.
  * Label: customer did NOT purchase in the window after the cut-off (churned = 1).
  * Customers are scored for the live dashboard using features as of the last date.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .features import customer_features

log = logging.getLogger("retailpulse")
FEATURES = ["Recency", "Frequency", "Monetary", "Items", "Distinct_Products", "Tenure_Days",
            "Avg_Order_Value", "Avg_Items_Per_Order", "Avg_Days_Between_Orders", "Return_Rate",
            "Orders_Last_90d", "Spend_Last_90d"]


def build_training_set(sales: pd.DataFrame, returns: pd.DataFrame):
    last = sales["InvoiceDate"].max()
    cutoff = (last - pd.Timedelta(days=config.CHURN_WINDOW_DAYS)).normalize()
    feats = customer_features(sales, returns, as_of=cutoff)
    future = sales[(sales["InvoiceDate"] >= cutoff) & sales["has_customer"]]
    active = set(future["Customer ID"].astype(float).astype(int).unique())
    feats["churned"] = (~feats["Customer ID"].isin(active)).astype(int)
    return feats, cutoff


def run_churn(sales: pd.DataFrame, returns: pd.DataFrame, live_feats: pd.DataFrame,
              n_trials: int = 15, use_mlflow: bool = True) -> tuple[pd.DataFrame, dict]:
    train_df, cutoff = build_training_set(sales, returns)
    X, y = train_df[FEATURES], train_df["churned"]
    if y.nunique() < 2 or y.value_counts().min() < 10:
        raise ValueError("Not enough churned/retained customers to train a classifier.")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=config.RANDOM_STATE)

    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                                random_state=config.RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=config.RANDOM_STATE),
    }

    # Optional hyper-parameter tuning of gradient boosting with Optuna
    tuned = None
    try:
        import optuna
        from sklearn.model_selection import cross_val_score

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            m = GradientBoostingClassifier(
                n_estimators=trial.suggest_int("n_estimators", 50, 300),
                max_depth=trial.suggest_int("max_depth", 2, 5),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                random_state=config.RANDOM_STATE)
            return cross_val_score(m, Xtr, ytr, cv=3, scoring="roc_auc").mean()

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE))
        study.optimize(objective, n_trials=n_trials)
        models["Gradient Boosting (Optuna)"] = GradientBoostingClassifier(**study.best_params, random_state=config.RANDOM_STATE)
        tuned = study.best_params
    except Exception as exc:
        log.warning("Optuna tuning skipped: %s", exc)

    results = {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        proba = m.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        results[name] = {"ROC_AUC": float(roc_auc_score(yte, proba)), "Accuracy": float(accuracy_score(yte, pred)),
                         "Precision": float(precision_score(yte, pred, zero_division=0)),
                         "Recall": float(recall_score(yte, pred, zero_division=0)),
                         "F1": float(f1_score(yte, pred, zero_division=0))}
    best = max(results, key=lambda k: results[k]["ROC_AUC"])
    log.info("Churn results: %s | best=%s", {k: round(v["ROC_AUC"], 3) for k, v in results.items()}, best)

    final = models[best].fit(X, y)   # refit on all labelled customers
    live = live_feats.copy()
    live["Churn_Probability"] = final.predict_proba(live[FEATURES])[:, 1]
    live["Risk_Level"] = pd.cut(live["Churn_Probability"], [-0.01, 0.33, 0.66, 1.01], labels=["Low", "Medium", "High"]).astype(str)

    # feature importance
    est = final[-1] if hasattr(final, "steps") else final
    if hasattr(est, "feature_importances_"):
        imp = dict(zip(FEATURES, est.feature_importances_.astype(float)))
    else:
        imp = dict(zip(FEATURES, np.abs(est.coef_[0]).astype(float)))

    report = {"cutoff_date": str(cutoff.date()), "window_days": config.CHURN_WINDOW_DAYS,
              "train_customers": int(len(train_df)), "churn_rate": float(y.mean()),
              "best_model": best, "metrics": results, "feature_importance": imp, "optuna_best_params": tuned,
              "note": "Recency is strongly tied to the churn definition; interpret importance with that in mind."}
    json.dump(report, open(config.CHURN_METRICS, "w"), indent=2)

    if use_mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri(config.MLFLOW_URI)
            mlflow.set_experiment("retailpulse-churn")
            for name, m in results.items():
                with mlflow.start_run(run_name=name):
                    mlflow.log_param("model", name)
                    mlflow.log_metrics(m)
        except Exception as exc:
            log.warning("MLflow logging skipped: %s", exc)
    return live, report
