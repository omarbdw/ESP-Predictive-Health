"""Streamlit page for generating case-level training windows."""

from __future__ import annotations

import streamlit as st

from esp_predictive_health.cases.case_manager import list_cases
from esp_predictive_health.config.failure_classes import CONFIRMATION_LEVELS
from esp_predictive_health.core.training_windows import (
    DEFAULT_PRE_FAILURE_DAYS,
    TrainingWindowConfig,
    build_training_windows,
)
from esp_predictive_health.database.db import DEFAULT_DB_PATH
from esp_predictive_health.ui.components import render_page_header, require_quality_gate


def render() -> None:
    """Render the Training Windows page."""
    render_page_header(
        "Model preparation",
        "Training Windows",
        "Create case-level pre-failure feature vectors without treating telemetry rows as independent cases.",
    )
    telemetry = st.session_state.get("engineered_features")
    if telemetry is None:
        telemetry = st.session_state.get("standardized_import")
    if telemetry is None:
        st.info("Apply a mapping on Import Data before generating training windows.")
        return
    if not require_quality_gate():
        return

    cases = list_cases(db_path=DEFAULT_DB_PATH)
    eligible_cases = [
        case for case in cases if case.confirmation_level in CONFIRMATION_LEVELS[:2] and case.failure_date
    ]
    if not eligible_cases:
        st.info("No eligible A/B cases with failure dates are registered yet.")
        return

    case_options = {f"{case.case_id} | {case.well_name} | {case.root_cause}": case for case in eligible_cases}
    selected_labels = st.multiselect(
        "Cases to include",
        options=list(case_options),
        default=list(case_options),
        key="training_case_selection",
    )
    windows = st.multiselect(
        "Pre-failure windows",
        options=list(DEFAULT_PRE_FAILURE_DAYS),
        default=list(DEFAULT_PRE_FAILURE_DAYS),
        format_func=lambda days: f"{days} day" if days == 1 else f"{days} days",
        key="training_window_selection",
    )
    controls = st.columns(2)
    with controls[0]:
        minimum_rows = st.number_input("Minimum telemetry rows per window", min_value=1, value=2, step=1)
    with controls[1]:
        include_b = st.checkbox("Include highly probable B cases", value=True)

    if st.button("Generate training windows", type="primary"):
        selected_cases = [case_options[label] for label in selected_labels]
        try:
            training_windows = build_training_windows(
                telemetry,
                selected_cases,
                config=TrainingWindowConfig(
                    pre_failure_days=tuple(windows),
                    minimum_rows=minimum_rows,
                    include_highly_probable=include_b,
                ),
            )
        except ValueError as error:
            st.error(str(error))
            return
        st.session_state["training_windows"] = training_windows
        st.session_state.pop("model_metrics", None)
        st.session_state.pop("model_artifact", None)
        st.session_state.pop("model_probabilities", None)
        st.session_state.pop("model_confidence", None)
        st.session_state.pop("model_calibration_status", None)

    training_windows = st.session_state.get("training_windows")
    if training_windows is None:
        st.info("Select cases and windows, then generate the case-level training table.")
        return

    st.success(f"Generated {len(training_windows):,} case-level training row(s).")
    if training_windows.empty:
        st.warning("No windows met the minimum telemetry-row requirement.")
        return
    st.dataframe(training_windows, use_container_width=True, hide_index=True)
    st.download_button(
        "Download training windows",
        data=training_windows.to_csv(index=False).encode("utf-8"),
        file_name="esp_training_windows.csv",
        mime="text/csv",
    )
