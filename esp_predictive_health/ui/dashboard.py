"""Operational dashboard landing page."""

from __future__ import annotations

import streamlit as st

from esp_predictive_health.ui.components import render_page_header


def _stage_status(ready: bool) -> str:
    return "Ready" if ready else "Pending"


def render() -> None:
    """Render the pipeline overview dashboard."""
    render_page_header(
        "ESP Predictive Health",
        "Operations overview",
        "Prepare trustworthy well telemetry before diagnosis and model training.",
    )
    standardized = st.session_state.get("standardized_import")
    engineered = st.session_state.get("engineered_features")
    events = st.session_state.get("detected_events")
    baseline = st.session_state.get("baseline_profile")
    quality = None
    if standardized is not None:
        from esp_predictive_health.core.quality import analyze_quality

        quality = analyze_quality(standardized)

    metrics = st.columns(4)
    metrics[0].metric("Telemetry rows", f"{len(standardized):,}" if standardized is not None else "--")
    metrics[1].metric("Quality score", f"{quality.score}/100" if quality else "--")
    metrics[2].metric("Detected events", f"{len(events):,}" if events is not None else "--")
    metrics[3].metric("Baseline bins", f"{len(baseline):,}" if baseline is not None else "--")

    st.subheader("Analysis pipeline")
    pipeline = [
        {"Stage": "Import and standardize", "Status": _stage_status(standardized is not None), "Next action": "Upload and map telemetry"},
        {"Stage": "Data quality", "Status": _stage_status(quality is not None), "Next action": "Review flagged readings"},
        {"Stage": "Feature engineering", "Status": _stage_status(engineered is not None), "Next action": "Calculate time-aware features"},
        {"Stage": "Event detection", "Status": _stage_status(events is not None), "Next action": "Review trips and transitions"},
        {"Stage": "Healthy-well baseline", "Status": _stage_status(baseline is not None), "Next action": "Compare at similar frequency"},
    ]
    st.dataframe(pipeline, use_container_width=True, hide_index=True)

    if standardized is None:
        st.info("Begin with Import Data. Raw uploads are preserved, units are normalized, and quality checks run before downstream analysis.")
    else:
        st.success("Telemetry is loaded. Use the sidebar pipeline stages to continue analysis.")
