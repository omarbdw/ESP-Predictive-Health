"""Streamlit page for browsing confirmed ESP cases."""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from esp_predictive_health.cases.case_manager import list_cases
from esp_predictive_health.config.failure_classes import CONFIRMATION_LEVELS, FAILURE_CLASSES


def render() -> None:
    """Render the searchable Case Library page."""
    st.header("Case Library")
    st.caption("Search and filter investigation records without modifying raw uploads.")

    all_cases = list_cases()
    if not all_cases:
        st.info("No cases registered yet. Use Register Confirmed Case to add the first record.")
        return

    fields = sorted({case.field_name for case in all_cases if case.field_name})
    wells = sorted({case.well_name for case in all_cases})
    cause_options = ["All causes", *FAILURE_CLASSES]
    field_options = ["All fields", *fields]
    well_options = ["All wells", *wells]
    level_options = ["All levels", *CONFIRMATION_LEVELS]

    search_query = st.text_input("Search cases", placeholder="Case ID, well, field, notes...")
    filter_columns = st.columns(5)
    with filter_columns[0]:
        selected_cause = st.selectbox(
            "Failure type",
            cause_options,
            format_func=lambda code: "All causes" if code == "All causes" else f"{code} - {FAILURE_CLASSES[code]}",
        )
    with filter_columns[1]:
        selected_field = st.selectbox("Field", field_options)
    with filter_columns[2]:
        selected_well = st.selectbox("Well", well_options)
    with filter_columns[3]:
        selected_level = st.selectbox("Confirmation level", level_options)
    with filter_columns[4]:
        selected_date = st.date_input(
            "Failure date from",
            value=date(2000, 1, 1),
            min_value=date(2000, 1, 1),
            max_value=date.today(),
        )

    filtered = [
        case
        for case in all_cases
        if (
            not search_query.strip()
            or search_query.strip().lower() in " ".join(
                (
                    case.case_id,
                    case.well_name,
                    case.field_name,
                    case.root_cause,
                    case.notes,
                )
            ).lower()
        )
        if (selected_cause == "All causes" or case.root_cause == selected_cause)
        and (selected_field == "All fields" or case.field_name == selected_field)
        and (selected_well == "All wells" or case.well_name == selected_well)
        and (selected_level == "All levels" or case.confirmation_level == selected_level)
        and (case.failure_date is None or case.failure_date >= selected_date)
    ]

    rows = [
        {
            "Case ID": case.case_id,
            "Well": case.well_name,
            "Field": case.field_name,
            "Well type": case.well_type,
            "Failure type": f"{case.root_cause} - {FAILURE_CLASSES[case.root_cause]}",
            "Confirmation": case.confirmation_level,
            "Failure date": case.failure_date.isoformat() if case.failure_date else "",
            "Raw data": case.raw_data_path or "Not provided",
        }
        for case in filtered
    ]
    st.write(f"{len(filtered)} case(s)")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
