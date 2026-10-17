"""Step 4: data cleaning."""
from __future__ import annotations

import json
import logging

import pandas as pd

from . import config

log = logging.getLogger("retailpulse")

# Stock codes that are fees/adjustments, not products
NON_PRODUCT = r"^(?:POST|D|M|DOT|CRUK|C2|S|B|BANK CHARGES|AMAZONFEE|ADJUST2?|TEST\d*|PADS|GIFT_.*)$"


def clean(df: pd.DataFrame, write_report: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (sales, returns, report).

    sales   - valid purchase lines (positive qty and price, real products)
    returns - cancellation lines (invoice starts with 'C'), kept for return-rate features
    report  - counts of what was removed and why (saved to outputs/cleaning_report.json)
    """
    rep: dict = {"rows_raw": int(len(df))}
    df = df.copy()
    df["Invoice"] = df["Invoice"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
    df["Description"] = df["Description"].astype("string").str.strip().str.upper()
    df["Country"] = df["Country"].astype("string").str.strip()

    rep["missing_customer_id"] = int(df["Customer ID"].isna().sum())
    rep["missing_description"] = int(df["Description"].isna().sum())

    # 1. exact duplicates (never drop the custom row)
    dup = df.duplicated(subset=[c for c in df.columns if c != "is_custom_row"], keep="first") & ~df["is_custom_row"]
    rep["duplicates_removed"] = int(dup.sum())
    df = df[~dup]

    # 2. cancellations -> separate table
    is_cancel = df["Invoice"].str.startswith("C")
    returns = df[is_cancel].copy()
    df = df[~is_cancel]
    rep["cancellation_lines"] = int(len(returns))

    # 3. invalid quantity / price
    bad_qp = (df["Quantity"] <= 0) | (df["Price"] <= 0)
    rep["invalid_qty_or_price_removed"] = int(bad_qp.sum())
    df = df[~bad_qp]

    # 4. non-product rows (postage, fees, adjustments)
    non_prod = df["StockCode"].str.match(NON_PRODUCT, case=False)
    rep["non_product_removed"] = int(non_prod.sum())
    df = df[~non_prod]

    # 5. missing description
    no_desc = df["Description"].isna()
    rep["missing_description_removed"] = int(no_desc.sum())
    df = df[~no_desc]

    # 6. outliers: flag (not drop) absurd quantities so the sales totals stay honest
    cap = df["Quantity"].quantile(0.9999)
    df["is_outlier"] = df["Quantity"] > max(cap, 5000)
    rep["quantity_outliers_flagged"] = int(df["is_outlier"].sum())

    # 7. types & derived columns
    df["Revenue"] = (df["Quantity"] * df["Price"]).round(2)
    df["Date"] = df["InvoiceDate"].dt.normalize()
    df["has_customer"] = df["Customer ID"].notna()
    df["Customer ID"] = df["Customer ID"].astype("Float64")
    df = df.reset_index(drop=True)

    returns["Revenue"] = (returns["Quantity"] * returns["Price"]).round(2)
    returns["Customer ID"] = returns["Customer ID"].astype("Float64")

    rep["rows_clean"] = int(len(df))
    rep["rows_with_customer"] = int(df["has_customer"].sum())
    rep["date_min"] = str(df["InvoiceDate"].min())
    rep["date_max"] = str(df["InvoiceDate"].max())
    rep["custom_row_survived_cleaning"] = bool(df["is_custom_row"].any())
    if write_report:
        with open(config.OUT / "cleaning_report.json", "w") as fh:
            json.dump(rep, fh, indent=2)
    log.info("Cleaning: %s -> %s rows", rep["rows_raw"], rep["rows_clean"])
    return df, returns.reset_index(drop=True), rep