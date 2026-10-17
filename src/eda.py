"""Step 6: exploratory data analysis. Saves PNG charts + a JSON summary for the report."""
from __future__ import annotations

import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import config

log = logging.getLogger("retailpulse")
BLUE, ORANGE = "#2F6FDE", "#E8833A"


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(config.EDA_DIR / name, dpi=130)
    plt.close(fig)


def run_eda(sales: pd.DataFrame) -> dict:
    s = sales[~sales["is_outlier"]]
    summary = {
        "revenue_total": round(float(s["Revenue"].sum()), 2),
        "orders": int(s["Invoice"].nunique()),
        "customers": int(s.loc[s["has_customer"], "Customer ID"].nunique()),
        "products": int(s["StockCode"].nunique()),
        "avg_order_value": round(float(s.groupby("Invoice")["Revenue"].sum().mean()), 2),
        "countries": int(s["Country"].nunique()),
        "uk_revenue_share": round(float(s.loc[s["Country"] == "United Kingdom", "Revenue"].sum() / s["Revenue"].sum()), 3),
    }

    monthly = s.groupby("YearMonth")["Revenue"].sum()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(monthly.index, monthly.values, color=BLUE)
    ax.set(title="Monthly revenue", ylabel="Revenue"); ax.tick_params(axis="x", rotation=60)
    _save(fig, "monthly_revenue.png")
    summary["peak_month"] = str(monthly.idxmax())

    top = s.groupby("Description")["Revenue"].sum().nlargest(10).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top.index.astype(str), top.values, color=BLUE); ax.set_title("Top 10 products by revenue")
    _save(fig, "top_products.png")

    ctry = s.groupby("Country")["Revenue"].sum().nlargest(10).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(ctry.index.astype(str), ctry.values, color=ORANGE); ax.set_title("Top 10 countries by revenue")
    _save(fig, "top_countries.png")

    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow = s.groupby("DayOfWeek")["Revenue"].sum().reindex(order).fillna(0)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(dow.index, dow.values, color=BLUE); ax.set_title("Revenue by day of week"); ax.tick_params(axis="x", rotation=30)
    _save(fig, "revenue_by_weekday.png")

    hour = s.groupby("Hour")["Revenue"].sum()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hour.index, hour.values, marker="o", color=BLUE); ax.set_title("Revenue by hour of day")
    _save(fig, "revenue_by_hour.png")

    summary["busiest_weekday"] = str(dow.idxmax())
    summary["busiest_hour"] = int(hour.idxmax())
    json.dump(summary, open(config.OUT / "eda_summary.json", "w"), indent=2)
    log.info("EDA done: %s", summary)
    return summary
