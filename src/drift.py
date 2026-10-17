"""Simple data-drift check (Population Stability Index) between early and recent data."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    edges = np.unique(np.quantile(expected.dropna(), np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    e = np.histogram(expected, edges)[0] / max(len(expected), 1)
    a = np.histogram(actual, edges)[0] / max(len(actual), 1)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def run_drift(sales: pd.DataFrame) -> dict:
    s = sales[~sales["is_outlier"]]
    mid = s["InvoiceDate"].min() + (s["InvoiceDate"].max() - s["InvoiceDate"].min()) / 2
    ref, cur = s[s["InvoiceDate"] < mid], s[s["InvoiceDate"] >= mid]
    out = {}
    for col in ("Quantity", "Price", "Revenue"):
        v = psi(ref[col], cur[col])
        out[col] = {"psi": round(v, 4), "status": "stable" if v < 0.1 else "moderate drift" if v < 0.25 else "significant drift"}
    out["split_date"] = str(mid.date())
    json.dump(out, open(config.DRIFT_JSON, "w"), indent=2)
    return out
