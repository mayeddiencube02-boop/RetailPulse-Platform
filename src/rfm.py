"""Step 7: customer segmentation - RFM scoring + K-Means clustering."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from . import config

log = logging.getLogger("retailpulse")


def _score(series: pd.Series, ascending: bool) -> pd.Series:
    """1-5 quintile score via rank (robust to ties)."""
    r = series.rank(method="first", ascending=ascending)
    return pd.qcut(r, 5, labels=[1, 2, 3, 4, 5]).astype(int)


def _segment(row) -> str:
    r, f = row["R_Score"], row["F_Score"]
    if r >= 4 and f >= 4: return "Champions"
    if r >= 3 and f >= 4: return "Loyal Customers"
    if r >= 4 and f <= 2: return "New Customers"
    if r >= 3 and f >= 2: return "Potential Loyalists"
    if r <= 2 and f >= 4: return "Cannot Lose Them"
    if r <= 2 and f >= 3: return "At Risk"
    if r <= 2: return "Hibernating"
    return "Need Attention"


def rfm_segmentation(feat: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = feat.copy()
    df["R_Score"] = _score(df["Recency"], ascending=False)       # recent = high
    df["F_Score"] = _score(df["Frequency"], ascending=True)
    df["M_Score"] = _score(df["Monetary"], ascending=True)
    df["RFM_Score"] = df["R_Score"].astype(str) + df["F_Score"].astype(str) + df["M_Score"].astype(str)
    df["Segment"] = df.apply(_segment, axis=1)

    # K-Means on log-scaled RFM; choose k by silhouette
    X = np.log1p(df[["Recency", "Frequency", "Monetary"]])
    X = StandardScaler().fit_transform(X)
    best_k, best_s = 3, -1.0
    scores = {}
    for k in range(2, 7):
        km = KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_STATE).fit(X)
        sc = silhouette_score(X, km.labels_, sample_size=min(5000, len(X)), random_state=config.RANDOM_STATE)
        scores[k] = round(float(sc), 4)
        if sc > best_s:
            best_k, best_s = k, sc
    km = KMeans(n_clusters=best_k, n_init=10, random_state=config.RANDOM_STATE).fit(X)
    df["Cluster"] = km.labels_

    # name clusters by average monetary value (highest = "High Value")
    order = df.groupby("Cluster")["Monetary"].mean().sort_values(ascending=False).index
    names = {c: f"Cluster {i + 1}" for i, c in enumerate(order)}
    df["Cluster_Name"] = df["Cluster"].map(names)

    info = {"best_k": int(best_k), "silhouette_by_k": scores, "n_customers": int(len(df)),
            "segment_counts": df["Segment"].value_counts().to_dict()}
    log.info("RFM done: k=%s silhouette=%.3f", best_k, best_s)
    return df, info
