"""Run with:  pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.cleaning import clean
from src.data_loader import add_custom_row
from src.features import add_time_features, customer_features
from src.make_sample_data import make_sample
from src.rfm import rfm_segmentation


@pytest.fixture(scope="module")
def prepared():
    raw = add_custom_row(make_sample(n_customers=200, n_products=60))
    sales, returns, rep = clean(raw, write_report=False)   # don't overwrite the real pipeline's report
    return raw, sales, returns, rep


def test_custom_row_added_and_flagged(prepared):
    raw, *_ = prepared
    assert raw["is_custom_row"].sum() == 1
    assert (raw.loc[raw["is_custom_row"], "Customer ID"] == config.CUSTOM_ROW["Customer ID"]).all()


def test_custom_row_survives_cleaning(prepared):
    _, sales, _, rep = prepared
    assert rep["custom_row_survived_cleaning"]
    assert sales["is_custom_row"].sum() == 1


def test_cleaning_removes_bad_rows(prepared):
    _, sales, returns, _ = prepared
    assert (sales["Quantity"] > 0).all() and (sales["Price"] > 0).all()
    assert not sales["Invoice"].str.startswith("C").any()
    assert returns["Invoice"].str.startswith("C").all()
    assert not sales["StockCode"].isin(["POST"]).any()


def test_revenue_column(prepared):
    _, sales, *_ = prepared
    assert np.allclose(sales["Revenue"], (sales["Quantity"] * sales["Price"]).round(2))


def test_rfm_scores_in_range(prepared):
    _, sales, returns, _ = prepared
    feats = customer_features(add_time_features(sales), returns)
    rfm, info = rfm_segmentation(feats)
    for c in ("R_Score", "F_Score", "M_Score"):
        assert rfm[c].between(1, 5).all()
    assert rfm["Customer ID"].is_unique
    assert info["best_k"] >= 2
