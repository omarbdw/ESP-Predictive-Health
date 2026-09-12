"""Streamlit page for building and reviewing frequency-aware ESP baselines."""

from __future__ import annotations

import streamlit as st

from esp_predictive_health.core.baseline import BaselineConfig, apply_baseline, build_baseline
from esp_predictive_health.ui.components import render_page_header, require_quality_gate


def render() -> None:
    """Render the Healthy-Well Baseline page."""
    render_page_header(
        "Analysis stage",
        "Healthy-Well Baseline",
        "Compare ESP behavior against the same well at similar operating frequency.",
    )

    standardized = st.session_state.get("standardized_import")
    if standardized is None:
        st.info("Apply a mapping on the Import Data page before building a baseline.")
        return
    if not require_quality_gate():
        return

    controls = st.columns(4)
    with controls[0]:
        bin_width = st.number_input("Frequency bin width (Hz)", min_value=0.1, value=1.0, step=0.5)
    with controls[1]:
        minimum_samples = st.number_input("Minimum samples per bin", min_value=1, value=3, step=1)
    with controls[2]:
        tolerance = st.number_input("Matching tolerance (Hz)", min_value=0.1, value=2.0, step=0.5)
    with controls[3]:
        exclude_trips = st.checkbox("Exclude trip rows", value=True)

    if "frequency_hz" in standardized.columns:
        frequency_values = standardized["frequency_hz"]
        st.caption(
            f"Frequency coverage: {frequency_values.notna().sum():,} valid rows; "
            f"maximum {frequency_values.max() if frequency_values.notna().any() else 'missing'} Hz."
        )

    if st.button("Build baseline", type="primary"):
        healthy_mask = None
        if exclude_trips and "trip_reason" in standardized.columns:
            reasons = standardized["trip_reason"].astype("string").str.strip().str.lower()
            healthy_mask = reasons.isin({"", "nan", "none", "normal", "no trip"})
        config = BaselineConfig(
            frequency_bin_hz=bin_width,
            minimum_samples_per_bin=minimum_samples,
            matching_tolerance_hz=tolerance,
        )
        try:
            baseline = build_baseline(standardized, healthy_mask=healthy_mask, config=config)
            scored = apply_baseline(standardized, baseline, config=config)
        except ValueError as error:
            st.error(str(error))
            return
        st.session_state["baseline_profile"] = baseline
        st.session_state["baseline_scored"] = scored

    baseline = st.session_state.get("baseline_profile")
    scored = st.session_state.get("baseline_scored")
    if baseline is None or scored is None:
        st.info("Configure the baseline and build it to review frequency-conditioned expectations.")
        return

    st.success(f"Built baseline across {len(baseline):,} frequency bin(s).")
    st.subheader("Baseline profile")
    st.dataframe(baseline, use_container_width=True, hide_index=True)
    st.download_button(
        "Download baseline profile",
        data=baseline.to_csv(index=False).encode("utf-8"),
        file_name="esp_baseline_profile.csv",
        mime="text/csv",
    )
    st.subheader("Telemetry deviations")
    st.dataframe(scored.head(50), use_container_width=True, hide_index=True)
    st.download_button(
        "Download baseline deviations",
        data=scored.to_csv(index=False).encode("utf-8"),
        file_name="esp_baseline_deviations.csv",
        mime="text/csv",
    )
