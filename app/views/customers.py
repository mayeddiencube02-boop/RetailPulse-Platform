import plotly.graph_objects as go
import streamlit as st

from app.common import AQUA, BLUE, ORANGE, config, custom_row_banner, kpi, load_json, load_parquet, money, require, style

st.title("Customer Dashboard")
rfm = require(load_parquet(str(config.RFM_PARQUET)), "rfm_segments.parquet")
churn = require(load_parquet(str(config.CHURN_PARQUET)), "churn_scores.parquet")
metrics = load_json(str(config.CHURN_METRICS)) or {}

k = st.columns(4)
kpi(k[0], "Customers", f"{len(rfm):,}")
kpi(k[1], "Avg spend / customer", money(rfm["Monetary"].mean()))
kpi(k[2], "High churn risk", f"{(churn['Risk_Level'] == 'High').sum():,}")
kpi(k[3], "Churn model ROC-AUC",
    f"{metrics['metrics'][metrics['best_model']]['ROC_AUC']:.2f}" if metrics else "n/a", metrics.get("best_model"))

tab1, tab2, tab3 = st.tabs(["Segments (RFM)", "Churn risk", "Customer lookup"])

with tab1:
    seg = (rfm.groupby("Segment")
              .agg(Customers=("Customer ID", "count"), Avg_Recency=("Recency", "mean"),
                   Avg_Frequency=("Frequency", "mean"), Avg_Spend=("Monetary", "mean"),
                   Revenue=("Monetary", "sum"))
              .sort_values("Revenue", ascending=False).reset_index())
    a, b = st.columns(2)
    fig = go.Figure(go.Bar(x=seg["Customers"], y=seg["Segment"], orientation="h", marker_color=BLUE))
    fig.update_yaxes(autorange="reversed")
    a.plotly_chart(style(fig.update_layout(hovermode="closest"), "Customers per segment"), width="stretch")
    fig = go.Figure(go.Bar(x=seg["Revenue"], y=seg["Segment"], orientation="h", marker_color=ORANGE))
    fig.update_yaxes(autorange="reversed")
    b.plotly_chart(style(fig.update_layout(hovermode="closest"), "Revenue per segment"), width="stretch")
    st.dataframe(seg.round(1), width="stretch", hide_index=True)
    st.caption("Segments come from 1-5 quintile scores on Recency, Frequency and Monetary value.")

with tab2:
    a, b = st.columns(2)
    order = ["Low", "Medium", "High"]
    cnt = churn["Risk_Level"].value_counts().reindex(order).fillna(0).reset_index()
    cnt.columns = ["Risk", "Customers"]
    fig = go.Figure(go.Bar(x=cnt["Risk"], y=cnt["Customers"], marker_color=BLUE))
    a.plotly_chart(style(fig, "Customers by churn risk"), width="stretch")
    if metrics:
        imp = (sorted(metrics["feature_importance"].items(), key=lambda x: x[1])[-8:])
        fig = go.Figure(go.Bar(x=[v for _, v in imp], y=[k_ for k_, _ in imp], orientation="h", marker_color=AQUA))
        b.plotly_chart(style(fig.update_layout(hovermode="closest"), "What drives churn (top features)"),
                       width="stretch")
        st.caption(f"Churn = no purchase in the {metrics['window_days']} days after {metrics['cutoff_date']}. "
                   "Recency is closely tied to that definition, so it naturally ranks high.")

    st.subheader("High-risk watchlist")
    wl = (churn[churn["Risk_Level"] == "High"]
          .sort_values("Monetary", ascending=False)
          [["Customer ID", "Segment", "Churn_Probability", "Recency", "Frequency", "Monetary", "Country"]])
    st.dataframe(wl.head(200), width="stretch", hide_index=True,
                 column_config={"Churn_Probability": st.column_config.ProgressColumn("Churn probability", min_value=0, max_value=1, format="%.2f"),
                                "Monetary": st.column_config.NumberColumn(format="£%.0f")})
    st.download_button("Download full watchlist (CSV)", wl.to_csv(index=False), "churn_watchlist.csv", "text/csv")

with tab3:
    cid = st.number_input("Customer ID", min_value=0, value=int(churn.sort_values("Monetary").iloc[-1]["Customer ID"]), step=1)
    row = churn[churn["Customer ID"] == cid]
    if row.empty:
        st.info("No customer with that ID in the data.")
    else:
        r = row.iloc[0]
        c = st.columns(4)
        kpi(c[0], "Segment", str(r["Segment"]))
        kpi(c[1], "Churn probability", f"{r['Churn_Probability']:.0%}")
        kpi(c[2], "Orders", f"{int(r['Frequency'])}")
        kpi(c[3], "Total spend", money(r["Monetary"]))
        st.write(f"Last purchase {int(r['Recency'])} days ago · risk level **{r['Risk_Level']}** · {r['Country']}")
    custom_row_banner(churn)
