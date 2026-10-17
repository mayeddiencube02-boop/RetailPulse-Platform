"""RetailPulse - Streamlit dashboard entry point.

Run:  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

st.set_page_config(page_title="RetailPulse", page_icon="📈", layout="wide")

pages = [
    st.Page("views/sales.py", title="Sales Dashboard", icon="💷", default=True),
    st.Page("views/customers.py", title="Customer Dashboard", icon="👥"),
    st.Page("views/forecast.py", title="Forecast Dashboard", icon="🔮"),
    st.Page("views/inventory.py", title="Inventory Dashboard", icon="📦"),
]
st.sidebar.title("RetailPulse")
st.sidebar.caption("Customer analytics & demand forecasting")
st.navigation(pages).run()
