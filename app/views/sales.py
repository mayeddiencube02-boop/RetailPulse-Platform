import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.common import BLUE, ORANGE, config, kpi, load_parquet, money, require, style

st.title("Sales Dashboard")
SALES_COLS = ("Invoice", "Description", "Country", "Revenue", "Date", "has_customer", "Customer ID",
              "is_outlier", "YearMonth", "DayOfWeek", "Hour")
sales = require(load_parquet(str(config.CLEAN_PARQUET), SALES_COLS, True), "clean_transactions.parquet")
sales = sales[~sales["is_outlier"]]

# ---- filters (one row above the charts) ----
c1, c2 = st.columns([2, 3])
dmin, dmax = sales["Date"].min().date(), sales["Date"].max().date()
rng = c1.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
countries = sorted(sales["Country"].dropna().unique())
picked = c2.multiselect("Country", countries, placeholder="All countries")

if len(rng) == 2:
    sales = sales[(sales["Date"] >= pd.Timestamp(rng[0])) & (sales["Date"] <= pd.Timestamp(rng[1]))]
if picked:
    sales = sales[sales["Country"].isin(picked)]
if sales.empty:
    st.warning("No sales match these filters.")
    st.stop()

# ---- KPIs ----
orders = sales["Invoice"].nunique()
k = st.columns(4)
kpi(k[0], "Revenue", money(sales["Revenue"].sum()))
kpi(k[1], "Orders", f"{orders:,}")
kpi(k[2], "Customers", f"{sales.loc[sales['has_customer'], 'Customer ID'].nunique():,}",
    "Orders without a customer ID (guest checkouts) are not counted here.")
kpi(k[3], "Avg order value", money(sales["Revenue"].sum() / orders))

# ---- revenue over time ----
monthly = sales.groupby("YearMonth", as_index=False, observed=True)["Revenue"].sum()
fig = go.Figure(go.Scatter(x=monthly["YearMonth"], y=monthly["Revenue"], mode="lines+markers",
                           line=dict(color=BLUE, width=2), marker=dict(size=8), name="Revenue"))
st.plotly_chart(style(fig, "Monthly revenue"), width="stretch")

# ---- products & countries ----
a, b = st.columns(2)
top_p = sales.groupby("Description", observed=True)["Revenue"].sum().nlargest(10).sort_values().reset_index()
fig = go.Figure(go.Bar(x=top_p["Revenue"], y=top_p["Description"], orientation="h", marker_color=BLUE))
a.plotly_chart(style(fig.update_layout(hovermode="closest"), "Top 10 products by revenue"), width="stretch")

top_c = sales.groupby("Country", observed=True)["Revenue"].sum().nlargest(10).sort_values().reset_index()
fig = go.Figure(go.Bar(x=top_c["Revenue"], y=top_c["Country"], orientation="h", marker_color=ORANGE))
b.plotly_chart(style(fig.update_layout(hovermode="closest"), "Top 10 countries by revenue"), width="stretch")

# ---- when do people buy ----
order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
a, b = st.columns(2)
dow = sales.groupby("DayOfWeek", observed=True)["Revenue"].sum().reindex(order).fillna(0).reset_index()
fig = go.Figure(go.Bar(x=dow["DayOfWeek"], y=dow["Revenue"], marker_color=BLUE))
a.plotly_chart(style(fig, "Revenue by day of week"), width="stretch")
hr = sales.groupby("Hour", as_index=False)["Revenue"].sum()
fig = go.Figure(go.Scatter(x=hr["Hour"], y=hr["Revenue"], mode="lines+markers", line=dict(color=BLUE, width=2)))
b.plotly_chart(style(fig, "Revenue by hour of day"), width="stretch")

with st.expander("Table view"):
    st.dataframe(monthly.rename(columns={"YearMonth": "Month"}), width="stretch", hide_index=True)
