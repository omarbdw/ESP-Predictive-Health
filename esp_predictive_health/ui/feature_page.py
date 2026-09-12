"""Streamlit page for reviewing engineered ESP features."""

from __future__ import annotations

import streamlit as st

from esp_predictive_health.core.feature_engineering import (
    DEFAULT_ROLLING_WINDOWS_MINUTES,
    add_engineered_features,
)
from esp_predictive_health.ui.components import render_page_header


def render() -> None:
    """Render the Feature Engineering page."""
    render_page_header(
        "Analysis stage",
        "Feature Engineering",
        "Create operating-condition-aware features from standardized telemetry.",
    )

    standardized = st.session_state.get("standardized_import")
    if standardized is None:
        st.info("Apply a mapping on the Import Data page before engineering features.")
        return

    selected_windows = st.multiselect(
        "Rolling windows",
        options=list(DEFAULT_ROLLING_WINDOWS_MINUTES),
        default=[30, 60, 360],
        format_func=lambda minutes: f"{minutes} minutes" if minutes < 1440 else f"{minutes // 1440} day(s)",
        key="feature_rolling_windows",
    )
    if not selected_windows:
        st.warning("Select at least one rolling window.")
        return

    if st.button("Calculate features", type="primary"):
        try:
            engineered = add_engineered_features(
                standardized,
                rolling_windows_minutes=selected_windows,
            )
        except ValueError as error:
            st.error(str(error))
            return
        st.session_state["engineered_features"] = engineered
        st.session_state["engineered_windows"] = tuple(selected_windows)

    engineered = st.session_state.get("engineered_features")
    if engineered is None:
        st.info("Choose windows and calculate features to preview the engineered dataset.")
        return

    st.success(
        f"Created {len(engineered.columns):,} columns for {len(engineered):,} telemetry rows."
    )
    st.dataframe(engineered.head(50), use_container_width=True, hide_index=True)
    st.download_button(
        "Download engineered CSV",
        data=engineered.to_csv(index=False).encode("utf-8"),
        file_name="engineered_esp_features.csv",
        mime="text/csv",
    )
