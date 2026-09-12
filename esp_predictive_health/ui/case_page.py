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


def _optional_date(label: str):
    return st.date_input(label, value=None)


def render() -> None:
    """Render the Register Confirmed Case page."""
    st.header("Register Confirmed Case")
    st.caption("Store the investigation record separately from the raw time-series file.")

    with st.form("register_case_form", clear_on_submit=True):
        st.subheader("Well and ESP")
        well_name = st.text_input("Well name", placeholder="e.g. ESP-001")
        field_name = st.text_input("Field name")
        well_type = st.selectbox("Well type", WELL_TYPES)
        esp_model = st.text_input("ESP model")
        pump_model = st.text_input("Pump model")
        motor_hp = st.number_input("Motor horsepower", min_value=0.0, step=1.0)

        st.subheader("Case timeline")
        case_start = _optional_date("Case start")
        case_end = _optional_date("Case end")
        symptom_start = _optional_date("Symptom start")
        failure_date = _optional_date("Failure date")

        st.subheader("Diagnosis and confirmation")
        root_cause_code = st.selectbox(
            "Primary root cause",
            options=list(FAILURE_CLASSES),
            format_func=lambda code: f"{code} - {FAILURE_CLASSES[code]}",
        )
        secondary_cause = st.selectbox(
            "Secondary cause (optional)",
            options=["None", *FAILURE_CLASSES],
            format_func=lambda code: "None" if code == "None" else f"{code} - {FAILURE_CLASSES[code]}",
        )
        confirmation_level = st.selectbox("Confirmation level", CONFIRMATION_LEVELS)
        confirmation_method = st.text_input(
            "Confirmation method", placeholder="e.g. teardown report, pressure test"
        )
        observed_symptoms = st.text_area("Observed symptoms")
        notes = st.text_area("Notes")
        uploaded_file = st.file_uploader(
            "Historical dataset (CSV or Excel, optional)", type=["csv", "xlsx", "xls"]
        )

        submitted = st.form_submit_button("Save confirmed case", type="primary")

    if not submitted:
        return

    if not well_name.strip():
        st.error("Well name is required.")
        return
    if not confirmation_method.strip():
        st.error("Confirmation method is required.")
        return

    try:
        raw_data_path = _save_raw_upload(uploaded_file)
        case = create_case(
            well_name=well_name,
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
