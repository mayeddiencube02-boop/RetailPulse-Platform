import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app.common import BLUE, ORANGE, config, kpi, load_parquet, require, style

st.title("Inventory Dashboard")
inv = require(load_parquet(str(config.INVENTORY_PARQUET)), "inventory_plan.parquet").copy()

st.warning("The dataset has no stock-on-hand column, so current stock is **assumed** from the slider below. "
           "Reorder points, safety stock and EOQ are calculated from real sales; the status labels depend on the assumption.",
           icon="⚠️")
days = st.slider("Assumed stock on hand (days of recent demand)", 1, 60, 14)

inv["Assumed_Stock"] = inv["Recent_Daily_Demand_60d"] * days
inv["Days_of_Cover"] = inv["Assumed_Stock"] / inv["Recent_Daily_Demand_60d"].replace(0, np.nan)
inv["Status"] = np.where(inv["Assumed_Stock"] <= inv["Reorder_Point"], "🔴 Reorder now",
                         np.where(inv["Days_of_Cover"] > 60, "🟡 Overstock", "🟢 OK"))

k = st.columns(4)
kpi(k[0], "Products analysed", f"{len(inv)}")
kpi(k[1], "Reorder now", f"{(inv['Status'] == '🔴 Reorder now').sum()}")
kpi(k[2], "Overstock", f"{(inv['Status'] == '🟡 Overstock').sum()}")
kpi(k[3], "Lead time / service level", f"{config.LEAD_TIME_DAYS}d / {int(0.95 * 100)}%")

a, b = st.columns(2)
top = inv.nlargest(12, "Reorder_Point").sort_values("Reorder_Point")
fig = go.Figure()
fig.add_trace(go.Bar(x=top["Safety_Stock"], y=top["Description"], orientation="h", name="Safety stock", marker_color=BLUE))
fig.add_trace(go.Bar(x=top["Reorder_Point"] - top["Safety_Stock"], y=top["Description"], orientation="h",
                     name="Lead-time demand", marker_color=ORANGE))
fig.update_layout(barmode="stack", hovermode="closest")
a.plotly_chart(style(fig, "Reorder point = lead-time demand + safety stock"), width="stretch")

counts = inv["Status"].value_counts().reset_index()
counts.columns = ["Status", "Products"]
fig = go.Figure(go.Bar(x=counts["Status"], y=counts["Products"], marker_color=BLUE))
b.plotly_chart(style(fig, "Products by stock status"), width="stretch")

st.subheader("Inventory plan")
status_filter = st.multiselect("Status", sorted(inv["Status"].unique()), default=sorted(inv["Status"].unique()))
view = inv[inv["Status"].isin(status_filter)][
    ["StockCode", "Description", "Status", "Avg_Daily_Demand", "Safety_Stock", "Reorder_Point", "EOQ", "Days_of_Cover"]]
st.dataframe(view.round(1), width="stretch", hide_index=True)
st.download_button("Download plan (CSV)", view.to_csv(index=False), "inventory_plan.csv", "text/csv")
st.caption("EOQ assumes unit cost = 60% of price, £50 per order and 20% annual holding cost. Change these in src/config.py.")
