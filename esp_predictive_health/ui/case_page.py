"""Streamlit page for registering confirmed ESP cases."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from uuid import uuid4

import streamlit as st

from esp_predictive_health.cases.case_manager import create_case
from esp_predictive_health.config.failure_classes import (
    CONFIRMATION_LEVELS,
    FAILURE_CLASSES,
    WELL_TYPES,
)
from esp_predictive_health.database.db import PROJECT_ROOT
from esp_predictive_health.ui.components import render_page_header

def _save_raw_upload(uploaded_file) -> str | None:
    """Persist an upload under a unique filename and return its project-relative path."""
    if uploaded_file is None:
        return None
    raw_directory = PROJECT_ROOT / "data" / "raw"
    raw_directory.mkdir(parents=True, exist_ok=True)
    safe_suffix = Path(uploaded_file.name).suffix.lower() or ".bin"
    destination = raw_directory / f"{date.today():%Y%m%d}_{uuid4().hex[:8]}{safe_suffix}"
    destination.write_bytes(uploaded_file.getbuffer())
    return destination.relative_to(PROJECT_ROOT).as_posix()


def _optional_date(label: str, key: str):
    return st.date_input(label, value=None, key=key)


def render() -> None:
    """Render the Register Confirmed Case page."""
    render_page_header(
        "Case management",
        "Register Confirmed Case",
        "Store the investigation record separately from the raw time-series file.",
    )

    with st.form("register_case_form"):
        st.subheader("Well and ESP")
        well_name = st.text_input("Well name", placeholder="e.g. ESP-001", key="case_well_name")
        well_id = st.text_input("Well / asset ID", placeholder="Must match telemetry well_id", key="case_well_id")
        field_name = st.text_input("Field name", key="case_field_name")
        well_type = st.selectbox("Well type", WELL_TYPES, key="case_well_type")
        esp_model = st.text_input("ESP model", key="case_esp_model")
        pump_model = st.text_input("Pump model", key="case_pump_model")
        motor_hp = st.number_input("Motor horsepower", min_value=0.0, step=1.0, key="case_motor_hp")

        st.subheader("Case timeline")
        case_start = _optional_date("Case start", "case_start")
        case_end = _optional_date("Case end", "case_end")
        symptom_start = _optional_date("Symptom start", "case_symptom_start")
        failure_date = _optional_date("Failure date", "case_failure_date")

        st.subheader("Diagnosis and confirmation")
        root_cause_code = st.selectbox(
            "Primary root cause",
            options=list(FAILURE_CLASSES),
            format_func=lambda code: f"{code} - {FAILURE_CLASSES[code]}",
            key="case_root_cause",
        )
        secondary_cause = st.selectbox(
            "Secondary cause (optional)",
            options=["None", *FAILURE_CLASSES],
            format_func=lambda code: "None" if code == "None" else f"{code} - {FAILURE_CLASSES[code]}",
            key="case_secondary_cause",
        )
        confirmation_level = st.selectbox("Confirmation level", CONFIRMATION_LEVELS, key="case_confirmation_level")
        confirmation_method = st.text_input(
            "Confirmation method",
            placeholder="e.g. teardown report, pressure test",
            key="case_confirmation_method",
        )
        observed_symptoms = st.text_area("Observed symptoms", key="case_observed_symptoms")
        notes = st.text_area("Notes", key="case_notes")
        uploaded_file = st.file_uploader(
            "Historical dataset (CSV or Excel, optional)",
            type=["csv", "xlsx", "xls"],
            key="case_raw_upload",
        )

        submitted = st.form_submit_button("Save confirmed case", type="primary")

    if not submitted:
        return

    if not well_name.strip():
        st.error("Well name is required.")
        return
    if not well_id.strip():
        st.error("Well / asset ID is required for safe telemetry association.")
        return
    if not confirmation_method.strip():
        st.error("Confirmation method is required.")
        return
    if case_start and case_end and case_end < case_start:
        st.error("Case end cannot be before case start.")
        return
    if symptom_start and failure_date and failure_date < symptom_start:
        st.error("Failure date cannot be before symptom start.")
        return

    try:
        raw_data_path = _save_raw_upload(uploaded_file)
        case = create_case(
            well_name=well_name,
            well_id=well_id,
            field_name=field_name,
            well_type=well_type,
            case_start=case_start,
            case_end=case_end,
            symptom_start=symptom_start,
            failure_date=failure_date,
            root_cause=root_cause_code,
            secondary_cause=None if secondary_cause == "None" else secondary_cause,
            observed_symptoms=observed_symptoms,
            confirmation_level=confirmation_level,
            confirmation_method=confirmation_method,
            esp_model=esp_model,
            pump_model=pump_model,
            motor_hp=motor_hp or None,
            notes=notes,
            raw_data_path=raw_data_path,
        )
    except (OSError, ValueError) as error:
        st.error(f"Case was not saved: {error}")
        return

    st.success(f"Saved case {case.case_id}.")
    if raw_data_path:
        st.info(f"Raw dataset preserved at `{raw_data_path}`.")
