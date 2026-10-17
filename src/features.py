"""Step 5: feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    d = df["InvoiceDate"]
    df["Year"] = d.dt.year
    df["Month"] = d.dt.month
    df["YearMonth"] = d.dt.to_period("M").astype(str)
    df["DayOfWeek"] = d.dt.day_name()
    df["Hour"] = d.dt.hour
    return df


def customer_features(sales: pd.DataFrame, returns: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """One row per customer using only data up to `as_of` (default: all data).

    Recency / Frequency / Monetary plus behavioural features.
    """
    s = sales[sales["has_customer"] & ~sales["is_outlier"]].copy()
    r = returns[returns["Customer ID"].notna()].copy()
    if as_of is None:
        as_of = s["InvoiceDate"].max() + pd.Timedelta(days=1)
    s = s[s["InvoiceDate"] < as_of]
    r = r[r["InvoiceDate"] < as_of]
    s["cid"] = s["Customer ID"].astype(float)
    r["cid"] = r["Customer ID"].astype(float)

    g = s.groupby("cid")
    feat = pd.DataFrame({
        "Recency": (as_of - g["InvoiceDate"].max()).dt.days,
        "Frequency": g["Invoice"].nunique(),
        "Monetary": g["Revenue"].sum(),
        "Items": g["Quantity"].sum(),
        "Distinct_Products": g["StockCode"].nunique(),
        "First_Purchase": g["InvoiceDate"].min(),
        "Last_Purchase": g["InvoiceDate"].max(),
        "Country": g["Country"].agg(lambda x: x.mode().iat[0]),
    })
    feat["Tenure_Days"] = (as_of - feat["First_Purchase"]).dt.days
    feat["Avg_Order_Value"] = feat["Monetary"] / feat["Frequency"]
    feat["Avg_Items_Per_Order"] = feat["Items"] / feat["Frequency"]

    # average gap between orders
    inv = s.drop_duplicates("Invoice").sort_values("InvoiceDate")
    gaps = inv.groupby("cid")["InvoiceDate"].apply(lambda x: x.diff().dt.days.mean())
    feat["Avg_Days_Between_Orders"] = gaps.reindex(feat.index).fillna(feat["Tenure_Days"])

    # return behaviour
    ret_val = (-r.groupby("cid")["Revenue"].sum()).reindex(feat.index).fillna(0)
    feat["Return_Rate"] = (ret_val / feat["Monetary"]).clip(0, 1)

    # recent activity (last 90 days) as a trend signal
    recent = s[s["InvoiceDate"] >= as_of - pd.Timedelta(days=90)].groupby("cid")
    feat["Orders_Last_90d"] = recent["Invoice"].nunique().reindex(feat.index).fillna(0)
    feat["Spend_Last_90d"] = recent["Revenue"].sum().reindex(feat.index).fillna(0)

    feat.index = feat.index.astype(int)
    feat.index.name = "Customer ID"
    return feat.reset_index()


def daily_sales(sales: pd.DataFrame) -> pd.DataFrame:
    """Daily revenue / units / orders - the time series used for forecasting."""
    s = sales[~sales["is_outlier"]]
    d = s.groupby("Date").agg(Revenue=("Revenue", "sum"), Units=("Quantity", "sum"),
                              Orders=("Invoice", "nunique")).reset_index()
    full = pd.DataFrame({"Date": pd.date_range(d["Date"].min(), d["Date"].max(), freq="D")})
    d = full.merge(d, on="Date", how="left")
    d[["Revenue", "Units", "Orders"]] = d[["Revenue", "Units", "Orders"]].fillna(0)
    d["DayOfWeek"] = d["Date"].dt.dayofweek
    d["is_trading_day"] = d["Revenue"] > 0
    return d


def product_daily(sales: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    """Daily units for the top-N products by revenue (used by the inventory model)."""
    s = sales[~sales["is_outlier"]]
    top = s.groupby("StockCode")["Revenue"].sum().nlargest(top_n).index
    p = s[s["StockCode"].isin(top)].groupby(["StockCode", "Date"])["Quantity"].sum().reset_index()
    return p
