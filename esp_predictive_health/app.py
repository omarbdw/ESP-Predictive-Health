"""Streamlit entry point for ESP Predictive Health."""

import streamlit as st

from esp_predictive_health.ui.case_library import render as render_case_library
from esp_predictive_health.ui.case_page import render as render_case_page

st.set_page_config(
    page_title="ESP Predictive Health",
    page_icon="⚙",
    layout="wide",
)

st.title("ESP Predictive Health")
st.caption("Phase 1: confirmed case management")

page = st.sidebar.radio(
    "Navigation",
    options=["Dashboard", "Register Confirmed Case", "Case Library"],
)

if page == "Register Confirmed Case":
    render_case_page()
elif page == "Case Library":
    render_case_library()
else:
    st.header("Dashboard")
    st.info("Case management is ready. Register confirmed cases to begin building the case library.")
