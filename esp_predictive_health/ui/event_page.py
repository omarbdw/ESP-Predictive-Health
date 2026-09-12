"""Streamlit page for reviewing detected ESP operational events."""

from __future__ import annotations

import streamlit as st

from esp_predictive_health.core.events import EventDetectionConfig, detect_events
from esp_predictive_health.ui.components import render_page_header, require_quality_gate


def render() -> None:
    """Render the Event Detection page."""
    render_page_header(
        "Analysis stage",
        "Event Detection",
        "Detect operational events and symptoms without assigning root cause.",
    )

    standardized = st.session_state.get("standardized_import")
    if standardized is None:
        st.info("Apply a mapping on the Import Data page before detecting events.")
        return
    if not require_quality_gate():
        return

    controls = st.columns(3)
    with controls[0]:
        underload = st.number_input("Underload threshold (A, 0 disables)", min_value=0.0, value=0.0, step=1.0)
    with controls[1]:
        overload = st.number_input("Overload threshold (A, 0 disables)", min_value=0.0, value=0.0, step=1.0)
    with controls[2]:
        frequency_change = st.number_input("Frequency change threshold (Hz)", min_value=0.1, value=2.0, step=0.5)

    if st.button("Detect events", type="primary"):
        result = detect_events(
            standardized,
            config=EventDetectionConfig(
                underload_current_a=underload or None,
                overload_current_a=overload or None,
                frequency_change_hz=frequency_change,
            ),
        )
        st.session_state["detected_events"] = result.events
        st.session_state["event_features"] = result.features

    events = st.session_state.get("detected_events")
    if events is None:
        st.info("Configure thresholds and detect events to review the result.")
        return

    st.write(f"Detected {len(events):,} event(s).")
    if events.empty:
        st.success("No events matched the configured thresholds.")
    else:
        st.dataframe(events, use_container_width=True, hide_index=True)
        st.download_button(
            "Download event table",
            data=events.to_csv(index=False).encode("utf-8"),
            file_name="esp_events.csv",
            mime="text/csv",
        )

    event_features = st.session_state.get("event_features")
    if event_features is not None:
        st.subheader("Event features")
        st.dataframe(event_features.head(50), use_container_width=True, hide_index=True)
        st.download_button(
            "Download event features",
            data=event_features.to_csv(index=False).encode("utf-8"),
            file_name="esp_event_features.csv",
            mime="text/csv",
        )
