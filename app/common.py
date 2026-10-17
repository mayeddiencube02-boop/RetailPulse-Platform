"""Shared helpers for the RetailPulse dashboard: data loading, palette, chart styling."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402

# Validated default palette (slot 1 blue, slot 2 orange, slot 3 aqua)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#52514e", "#898781", "rgba(137,135,129,0.25)"
CUSTOM_ID = int(config.CUSTOM_ROW["Customer ID"])


@st.cache_data(show_spinner=False)
def load_parquet(path: str, columns: tuple | None = None, categorize: bool = False) -> pd.DataFrame | None:
    """Read a parquet file. `columns` limits what is loaded; `categorize` stores text columns
    as categories, which cuts memory a lot on big tables (keeps the app inside free-tier limits)."""
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_parquet(p, columns=list(columns) if columns else None)
    if categorize:
        for c in df.select_dtypes(include=["object", "string"]).columns:
            df[c] = df[c].astype("category")
    return df


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict | None:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def require(df, name: str):
    if df is None:
        st.error(f"`{name}` not found. Run the pipeline first:  `python run_pipeline.py`")
        st.stop()
    return df


def style(fig: go.Figure, title: str | None = None, height: int = 360) -> go.Figure:
    """Recessive grid, thin marks, transparent surface so it works in light and dark themes."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)) if title else None,
        height=height, margin=dict(l=8, r=8, t=44 if title else 12, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=12),
        hovermode="x unified", legend=dict(orientation="h", y=-0.2),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, ticks="outside", tickcolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def kpi(col, label: str, value: str, help_: str | None = None):
    col.metric(label, value, help=help_)


def money(x: float) -> str:
    return f"£{x:,.0f}"


def custom_row_banner(df: pd.DataFrame, id_col: str = "Customer ID"):
    if df is not None and id_col in df and (df[id_col] == CUSTOM_ID).any():
        st.caption(f"Customer {CUSTOM_ID} is the extra record added to the dataset for this project.")
