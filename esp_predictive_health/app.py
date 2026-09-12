"""Streamlit entry point for ESP Predictive Health."""

import streamlit as st

from esp_predictive_health.ui.baseline_page import render as render_baseline_page
from esp_predictive_health.ui.case_library import render as render_case_library
from esp_predictive_health.ui.case_page import render as render_case_page
from esp_predictive_health.ui.components import apply_app_theme, render_sidebar_status
from esp_predictive_health.ui.dashboard import render as render_dashboard
from esp_predictive_health.ui.feature_page import render as render_feature_page
from esp_predictive_health.ui.event_page import render as render_event_page
from esp_predictive_health.ui.import_page import render as render_import_page
from esp_predictive_health.ui.model_page import render as render_model_page
from esp_predictive_health.ui.training_page import render as render_training_page

st.set_page_config(
    page_title="ESP Predictive Health",
    page_icon="⚙",
    layout="wide",
)

apply_app_theme()
st.sidebar.markdown(
    '<div class="esp-brand"><div class="esp-brand-title">ESP Predictive Health</div>'
    '<div class="esp-brand-note">Well telemetry intelligence</div></div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown('<div class="esp-section-label">Workflow</div>', unsafe_allow_html=True)
page = st.sidebar.radio(
    "Navigation",
    options=[
        "Dashboard",
        "Import Data",
        "Feature Engineering",
        "Event Detection",
        "Healthy-Well Baseline",
        "Training Windows",
        "Train Model",
        "Register Confirmed Case",
        "Case Library",
    ],
    label_visibility="collapsed",
)
render_sidebar_status()

if page == "Dashboard":
    render_dashboard()
elif page == "Import Data":
    render_import_page()
elif page == "Feature Engineering":
    render_feature_page()
elif page == "Event Detection":
    render_event_page()
elif page == "Healthy-Well Baseline":
    render_baseline_page()
elif page == "Training Windows":
    render_training_page()
elif page == "Train Model":
    render_model_page()
elif page == "Register Confirmed Case":
    render_case_page()
elif page == "Case Library":
    render_case_library()
