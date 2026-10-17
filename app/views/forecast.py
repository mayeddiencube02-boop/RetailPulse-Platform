import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.common import BLUE, ORANGE, config, kpi, load_json, load_parquet, money, require, style

st.title("Forecast Dashboard")
fc = require(load_parquet(str(config.FORECAST_PARQUET)), "forecast.parquet")
rep = require(load_json(str(config.FORECAST_METRICS)), "forecast_metrics.json")
best = rep["best_model"]
m = rep["metrics"][best]

future = fc[fc["Type"] == "future"]
k = st.columns(4)
kpi(k[0], "Best model", best, "Chosen by lowest WAPE on the hold-out period")
kpi(k[1], f"Next {rep['horizon_days']} days revenue", money(future["Forecast"].sum()))
kpi(k[2], "Hold-out WAPE", f"{m['WAPE']:.1%}", "Weighted absolute % error; lower is better")
kpi(k[3], "Hold-out RMSE", money(m["RMSE"]))

history_days = st.slider("Days of history to show", 30, int(len(fc)), min(180, len(fc)), step=10)
view = fc.iloc[-(history_days + rep["horizon_days"]):] if history_days < len(fc) else fc
hist = view[view["Type"] != "future"]
fut = view[view["Type"] == "future"]
hold = view[view["Type"] == "holdout"]

fig = go.Figure()
fig.add_trace(go.Scatter(x=hist["Date"], y=hist["Actual"], name="Actual", mode="lines", line=dict(color=BLUE, width=2)))
fig.add_trace(go.Scatter(x=pd.concat([fut["Date"], fut["Date"][::-1]]),
                         y=pd.concat([fut["Upper"], fut["Lower"][::-1]]),
                         fill="toself", fillcolor="rgba(235,104,52,0.15)", line=dict(width=0),
                         name="90% range", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=pd.concat([hold["Date"], fut["Date"]]), y=pd.concat([hold["Forecast"], fut["Forecast"]]),
                         name="Forecast", mode="lines", line=dict(color=ORANGE, width=2, dash="dot")))
if len(hold):
    fig.add_vline(x=hold["Date"].min(), line_color="rgba(137,135,129,0.5)", line_dash="dash")
    fig.add_annotation(x=hold["Date"].min(), y=1, yref="paper", text="hold-out starts", showarrow=False, xanchor="left")
st.plotly_chart(style(fig, "Daily revenue: actual vs forecast", height=420), width="stretch")

st.subheader("Model comparison (hold-out period)")
tbl = pd.DataFrame(rep["metrics"]).T.reset_index().rename(columns={"index": "Model"}).sort_values("WAPE")
tbl["Best"] = tbl["Model"].eq(best).map({True: "✓ selected", False: ""})
st.dataframe(tbl, width="stretch", hide_index=True,
             column_config={"WAPE": st.column_config.NumberColumn(format="%.3f"),
                            "MAPE": st.column_config.NumberColumn(format="%.3f"),
                            "RMSE": st.column_config.NumberColumn(format="%.0f"),
                            "MAE": st.column_config.NumberColumn(format="%.0f")})
st.caption("Models that are not installed in the environment (e.g. Prophet, LSTM) are skipped and do not appear here. "
           "MAPE is unreliable on days with very low sales, so WAPE is used to pick the winner.")

with st.expander("Forecast table"):
    st.dataframe(future[["Date", "Forecast", "Lower", "Upper"]].round({"Forecast": 0, "Lower": 0, "Upper": 0}),
                 width="stretch", hide_index=True)
    st.download_button("Download forecast (CSV)", future.to_csv(index=False), "forecast.csv", "text/csv")
