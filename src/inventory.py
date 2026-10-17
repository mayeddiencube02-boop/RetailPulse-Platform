"""Step 10: inventory optimisation (safety stock, reorder point, EOQ) for top products.

NOTE: the Online Retail data has no stock-on-hand column. We therefore derive demand
statistics from sales and report the *recommended* policy values. Current stock is
simulated only if you pass `assumed_stock_days` (clearly labelled in the dashboard).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger("retailpulse")


def inventory_plan(sales: pd.DataFrame, top_n: int = 50, assumed_stock_days: float = 14.0) -> pd.DataFrame:
    s = sales[~sales["is_outlier"]]
    top = s.groupby("StockCode")["Revenue"].sum().nlargest(top_n).index
    s = s[s["StockCode"].isin(top)]
    days = pd.date_range(s["Date"].min(), s["Date"].max(), freq="D")

    rows = []
    for code, g in s.groupby("StockCode"):
        daily = g.groupby("Date")["Quantity"].sum().reindex(days, fill_value=0)
        mean_d, std_d = float(daily.mean()), float(daily.std(ddof=0))
        recent_mean = float(daily.tail(60).mean())
        L = config.LEAD_TIME_DAYS
        safety = config.SERVICE_LEVEL_Z * std_d * np.sqrt(L)
        rop = recent_mean * L + safety
        annual_demand = mean_d * 365
        unit_cost = float(g["Price"].median()) * 0.6          # assumption: cost = 60% of price
        h = max(unit_cost * config.HOLDING_COST_RATE, 1e-6)
        eoq = np.sqrt(2 * annual_demand * config.ORDER_COST / h)
        stock = recent_mean * assumed_stock_days               # ASSUMPTION - no real stock data
        rows.append({
            "StockCode": code,
            "Description": g["Description"].mode().iat[0],
            "Avg_Daily_Demand": mean_d,
            "Recent_Daily_Demand_60d": recent_mean,
            "Demand_Std": std_d,
            "Safety_Stock": safety,
            "Reorder_Point": rop,
            "EOQ": eoq,
            "Unit_Price": float(g["Price"].median()),
            "Revenue": float(g["Revenue"].sum()),
            "Assumed_Stock": stock,
        })
    df = pd.DataFrame(rows)
    df["Days_of_Cover"] = df["Assumed_Stock"] / df["Recent_Daily_Demand_60d"].replace(0, np.nan)
    df["Status"] = np.where(df["Assumed_Stock"] <= df["Reorder_Point"], "Reorder now",
                            np.where(df["Days_of_Cover"] > 60, "Overstock", "OK"))
    df = df.sort_values("Revenue", ascending=False).reset_index(drop=True)
    log.info("Inventory plan for %s products", len(df))
    return df
