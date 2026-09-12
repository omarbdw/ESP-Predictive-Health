"""Streamlit page for grouped validation and first model training."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from esp_predictive_health.core.modeling import (
    fit_classifier,
    grouped_oof_probabilities,
    predict_probabilities,
    validate_grouped,
)
from esp_predictive_health.core.temperature_calibration import (
    ConfidenceConfig,
    apply_temperature,
    assess_confidence,
    calibrate_temperature,
)
from esp_predictive_health.ui.components import render_page_header, require_quality_gate


def render() -> None:
    """Render the Train Model page."""
    render_page_header(
        "Model preparation",
        "Train Model",
        "Validate by well before fitting an interpretable first-pass diagnosis model.",
    )
    training_windows = st.session_state.get("training_windows")
    if training_windows is None or training_windows.empty:
        st.info("Generate eligible training windows before training a model.")
        return
    if not require_quality_gate():
        return

    st.write(
        f"Training table: {len(training_windows):,} case-level rows, "
        f"{training_windows['well_name'].nunique():,} wells, "
        f"{training_windows['root_cause'].nunique():,} root-cause classes."
    )
    folds = st.number_input("Grouped validation folds", min_value=2, max_value=5, value=3, step=1)
    minimum_probability = st.slider("Minimum diagnosis probability", 0.34, 0.90, 0.55, 0.01)
    minimum_margin = st.slider("Minimum probability margin", 0.00, 0.50, 0.10, 0.01)

    if st.button("Validate and train model", type="primary"):
        try:
            metrics = validate_grouped(training_windows, n_splits=folds)
            artifact = fit_classifier(training_windows)
            probabilities = predict_probabilities(artifact, training_windows)
            try:
                holdout_probabilities, holdout_labels = grouped_oof_probabilities(
                    training_windows, n_splits=folds
                )
                temperature = calibrate_temperature(holdout_probabilities, holdout_labels)
                calibration_status = f"Calibrated from grouped holdout data (T={temperature:.2f})"
                probabilities = apply_temperature(probabilities, temperature)
            except ValueError:
                temperature = None
                calibration_status = "Uncalibrated: insufficient grouped holdout data"
            confidence = assess_confidence(
                probabilities,
                config=ConfidenceConfig(
                    minimum_probability=minimum_probability,
                    minimum_margin=minimum_margin,
                ),
                required_missing=training_windows["missing_rate"].gt(0.5),
            )
        except ValueError as error:
            st.error(str(error))
            return
        st.session_state["model_metrics"] = metrics
        st.session_state["model_artifact"] = artifact
        st.session_state["model_probabilities"] = probabilities
        st.session_state["model_confidence"] = confidence
        st.session_state["model_calibration_status"] = calibration_status

    metrics = st.session_state.get("model_metrics")
    artifact = st.session_state.get("model_artifact")
    probabilities = st.session_state.get("model_probabilities")
    confidence = st.session_state.get("model_confidence")
    calibration_status = st.session_state.get("model_calibration_status")
    if metrics is None or artifact is None or probabilities is None or confidence is None:
        st.info("Run grouped validation and training to create the first model artifact.")
        return

    st.success(f"Model trained for classes: {', '.join(artifact.classes)}")
    st.info(calibration_status)
    st.subheader("Grouped validation")
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    metric_columns = st.columns(2)
    metric_columns[0].metric("Mean balanced accuracy", f"{metrics['balanced_accuracy'].mean():.2f}")
    metric_columns[1].metric("Mean weighted F1", f"{metrics['weighted_f1'].mean():.2f}")

    st.subheader("Training-row probability distribution")
    probability_preview = probabilities.copy()
    probability_preview.insert(0, "case_id", training_windows["case_id"].to_numpy())
    probability_preview.insert(1, "window_days", training_windows["window_days"].to_numpy())
    probability_preview = pd.concat([probability_preview, confidence.reset_index(drop=True)], axis=1)
    st.dataframe(probability_preview, use_container_width=True, hide_index=True)
    st.download_button(
        "Download validation metrics",
        data=metrics.to_csv(index=False).encode("utf-8"),
        file_name="model_validation_metrics.csv",
        mime="text/csv",
    )
