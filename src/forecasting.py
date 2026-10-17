"""Step 8: demand forecasting (time series).

Models compared on a hold-out of the last N days:
  - Seasonal-naive baseline
  - Gradient boosting on calendar + lag features (always available)
  - Prophet (if installed)
  - LSTM (if TensorFlow is installed)
  - Hybrid = average of the available non-baseline models
The best model by WAPE is used to produce the future forecast.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from . import config

log = logging.getLogger("retailpulse")


def _metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    mask = y > 0
    return {
        "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
        "MAE": float(np.mean(np.abs(y - p))),
        "WAPE": float(np.abs(y - p).sum() / max(y.sum(), 1e-9)),
        "MAPE": float(np.mean(np.abs((y[mask] - p[mask]) / y[mask]))) if mask.any() else float("nan"),
    }


def _calendar(dates: pd.Series) -> pd.DataFrame:
    d = pd.to_datetime(dates)
    return pd.DataFrame({
        "dow": d.dt.dayofweek, "month": d.dt.month, "dom": d.dt.day,
        "woy": d.dt.isocalendar().week.astype(int).values, "doy": d.dt.dayofyear,
    })


# ---------- individual models ----------
def seasonal_naive(train: pd.DataFrame, horizon_dates: pd.DatetimeIndex) -> np.ndarray:
    s = train.set_index("Date")["y"]
    out = []
    for d in horizon_dates:
        ref = d - pd.Timedelta(days=7)
        while ref not in s.index and ref > s.index.min():
            ref -= pd.Timedelta(days=7)
        out.append(s.get(ref, s.tail(28).mean()))
    return np.array(out)


def gbm_forecast(train: pd.DataFrame, horizon_dates: pd.DatetimeIndex) -> np.ndarray:
    """Recursive forecast using lag-7/14/28 + rolling means + calendar."""
    hist = train.set_index("Date")["y"].copy()

    def feats(date, series):
        row = {f"lag{l}": series.get(date - pd.Timedelta(days=l), np.nan) for l in (7, 14, 28)}
        window = series[series.index < date].tail(28)
        row["roll7"] = series[series.index < date].tail(7).mean()
        row["roll28"] = window.mean()
        cal = _calendar(pd.Series([date])).iloc[0].to_dict()
        return {**row, **cal}

    X = pd.DataFrame([feats(d, hist) for d in hist.index[28:]], index=hist.index[28:]).bfill().fillna(0)
    y = hist.loc[X.index]
    model = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                      subsample=0.8, random_state=config.RANDOM_STATE).fit(X, y)
    preds = []
    for d in horizon_dates:
        f = pd.DataFrame([feats(d, hist)]).bfill(axis=1).fillna(0)
        p = max(float(model.predict(f[X.columns])[0]), 0.0)
        hist.loc[d] = p
        preds.append(p)
    return np.array(preds)


def prophet_forecast(train: pd.DataFrame, horizon_dates: pd.DatetimeIndex):
    try:
        from prophet import Prophet
    except Exception:
        return None
    try:
        m = Prophet(weekly_seasonality=True, yearly_seasonality=len(train) > 365,
                    daily_seasonality=False, changepoint_prior_scale=0.1)
        m.fit(train.rename(columns={"Date": "ds"})[["ds", "y"]])
        fc = m.predict(pd.DataFrame({"ds": horizon_dates}))
        return np.clip(fc["yhat"].values, 0, None)
    except Exception as exc:
        log.warning("Prophet failed: %s", exc)
        return None


def lstm_forecast(train: pd.DataFrame, horizon_dates: pd.DatetimeIndex, lookback: int = 28, epochs: int = 30):
    try:
        import os
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import tensorflow as tf
    except Exception:
        return None
    try:
        tf.random.set_seed(config.RANDOM_STATE)
        y = train["y"].values.astype("float32")
        mu, sd = y.mean(), y.std() + 1e-6
        z = (y - mu) / sd
        Xs = np.stack([z[i:i + lookback] for i in range(len(z) - lookback)])[..., None]
        ys = z[lookback:]
        model = tf.keras.Sequential([
            tf.keras.layers.Input((lookback, 1)),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(Xs, ys, epochs=epochs, batch_size=16, verbose=0, validation_split=0.1,
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
        window = list(z[-lookback:])
        out = []
        for _ in horizon_dates:
            p = float(model.predict(np.array(window[-lookback:])[None, :, None], verbose=0)[0, 0])
            window.append(p)
            out.append(max(p * sd + mu, 0.0))
        return np.array(out)
    except Exception as exc:
        log.warning("LSTM failed: %s", exc)
        return None


# ---------- orchestration ----------
def run_forecast(daily: pd.DataFrame, target: str = "Revenue", holdout_days: int = 30,
                 horizon: int | None = None, use_mlflow: bool = True) -> tuple[pd.DataFrame, dict]:
    horizon = horizon or config.FORECAST_HORIZON_DAYS
    df = daily[["Date", target]].rename(columns={target: "y"}).copy()
    train, test = df.iloc[:-holdout_days], df.iloc[-holdout_days:]
    test_dates = pd.DatetimeIndex(test["Date"])

    candidates = {
        "Seasonal Naive": seasonal_naive(train, test_dates),
        "Gradient Boosting": gbm_forecast(train, test_dates),
    }
    for name, fn in (("Prophet", prophet_forecast), ("LSTM", lstm_forecast)):
        p = fn(train, test_dates)
        if p is not None:
            candidates[name] = p
        else:
            log.warning("%s not available - skipped", name)
    ml = [v for k, v in candidates.items() if k != "Seasonal Naive"]
    if len(ml) >= 2:
        candidates["Hybrid (avg)"] = np.mean(ml, axis=0)

    results = {k: _metrics(test["y"], v) for k, v in candidates.items()}
    best = min(results, key=lambda k: results[k]["WAPE"])
    log.info("Forecast hold-out results: %s | best=%s", {k: round(v["WAPE"], 3) for k, v in results.items()}, best)

    # refit on all data, forecast the future with the best model
    future = pd.date_range(df["Date"].max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    runner = {
        "Seasonal Naive": seasonal_naive, "Gradient Boosting": gbm_forecast,
        "Prophet": prophet_forecast, "LSTM": lstm_forecast,
    }
    if best.startswith("Hybrid"):
        parts = [runner[k](df, future) for k in candidates if k in ("Gradient Boosting", "Prophet", "LSTM")]
        fut = np.mean([p for p in parts if p is not None], axis=0)
    else:
        fut = runner[best](df, future)

    resid_sd = float(np.std(test["y"].values - candidates[best]))
    out = pd.concat([
        pd.DataFrame({"Date": df["Date"], "Actual": df["y"], "Forecast": np.nan, "Lower": np.nan, "Upper": np.nan, "Type": "history"}),
        pd.DataFrame({"Date": test_dates, "Actual": test["y"].values, "Forecast": candidates[best],
                      "Lower": np.clip(candidates[best] - 1.65 * resid_sd, 0, None),
                      "Upper": candidates[best] + 1.65 * resid_sd, "Type": "holdout"}),
        pd.DataFrame({"Date": future, "Actual": np.nan, "Forecast": fut,
                      "Lower": np.clip(fut - 1.65 * resid_sd, 0, None),
                      "Upper": fut + 1.65 * resid_sd, "Type": "future"}),
    ], ignore_index=True)

    report = {"target": target, "holdout_days": holdout_days, "best_model": best,
              "metrics": results, "horizon_days": horizon, "residual_sd": resid_sd}
    json.dump(report, open(config.FORECAST_METRICS, "w"), indent=2)

    if use_mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri(config.MLFLOW_URI)
            mlflow.set_experiment("retailpulse-forecasting")
            for name, m in results.items():
                with mlflow.start_run(run_name=name):
                    mlflow.log_params({"model": name, "target": target, "holdout_days": holdout_days})
                    mlflow.log_metrics({k: v for k, v in m.items() if not np.isnan(v)})
        except Exception as exc:
            log.warning("MLflow logging skipped: %s", exc)
    return out, report
