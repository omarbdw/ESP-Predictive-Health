"""Streamlit page for importing and standardizing ESP time-series data."""

from __future__ import annotations

from datetime import date
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import streamlit as st

from esp_predictive_health.core.column_mapping import (
    CANONICAL_FIELDS,
    FIELD_LABELS,
    ColumnMatch,
    detect_column_mapping,
)
from esp_predictive_health.core.data_import import read_uploaded_data, standardize_dataframe
from esp_predictive_health.core.mapping_templates import list_templates, load_template, save_template
from esp_predictive_health.core.quality import analyze_quality, clean_dataframe
from esp_predictive_health.core.units import UNIT_OPTIONS, detect_unit
from esp_predictive_health.database.db import PROJECT_ROOT
from esp_predictive_health.ui.components import render_page_header


def _save_raw_upload(uploaded_file) -> str:
    raw_directory = PROJECT_ROOT / "data" / "raw"
    raw_directory.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix.lower()
    destination = raw_directory / f"{date.today():%Y%m%d}_{uuid4().hex[:8]}{suffix}"
    destination.write_bytes(uploaded_file.getvalue())
    return destination.relative_to(PROJECT_ROOT).as_posix()


def _mapping_editor(
    columns: list[str], detected: list[ColumnMatch]
) -> tuple[list[ColumnMatch], dict[str, str]]:
    detected_by_source = {match.source_column: match for match in detected}
    options = ["Ignore", *CANONICAL_FIELDS]
    matches: list[ColumnMatch] = []
    unit_overrides: dict[str, str] = {}
    header = st.columns([2, 2, 1, 2, 2])
    header[0].markdown("**Source column**")
    header[1].markdown("**Detected field**")
    header[2].markdown("**Confidence**")
    header[3].markdown("**Unit**")
    header[4].markdown("**Manual mapping**")

    for column in columns:
        detected_match = detected_by_source[column]
        confidence_label = f"{detected_match.confidence:.0%}"
        cells = st.columns([2, 2, 1, 2, 2])
        cells[0].write(column)
        cells[1].write(
            FIELD_LABELS.get(detected_match.canonical_field, "Unmapped")
            if detected_match.canonical_field
            else "Unmapped"
        )
        cells[2].write(confidence_label)
        if detected_match.alternatives:
            cells[2].caption(
                " / ".join(
                    f"{FIELD_LABELS[field]} {probability:.0%}"
                    for field, probability in detected_match.alternatives
                )
            )
        default = detected_match.canonical_field or "Ignore"
        selected = cells[4].selectbox(
            "Canonical field",
            options,
            index=options.index(default),
            format_func=lambda field: "Ignore" if field == "Ignore" else FIELD_LABELS[field],
            key=f"mapping_{column}",
            label_visibility="collapsed",
        )
        if selected != "Ignore":
            unit_options = UNIT_OPTIONS[selected]
            detected_unit = detect_unit(column, selected)
            unit = cells[3].selectbox(
                "Source unit",
                unit_options,
                index=unit_options.index(detected_unit),
                key=f"unit_{column}",
                label_visibility="collapsed",
            )
            unit_overrides[column] = unit
        matches.append(
            ColumnMatch(
                source_column=column,
                canonical_field=None if selected == "Ignore" else selected,
                confidence=detected_match.confidence if selected == default else 1.0,
                method="automatic" if selected == default else "manual correction",
            )
        )
    return matches, unit_overrides


def render() -> None:
    """Render the Import Data page."""
    render_page_header(
        "Data intake",
        "Import Data",
        "Normalize vendor columns and units before quality, event, and feature analysis.",
    )
    uploaded_file = st.file_uploader(
        "Upload a CSV or Excel dataset",
        type=["csv", "xlsx", "xls"],
        key="import_upload",
    )
    if uploaded_file is None:
        st.info("Upload a file to inspect its columns and create a canonical dataset.")
        return

    try:
        upload_bytes = uploaded_file.getvalue()
        upload_fingerprint = sha256(upload_bytes).hexdigest()
        if st.session_state.get("import_fingerprint") != upload_fingerprint:
            st.session_state.pop("standardized_import", None)
            st.session_state.pop("standardized_raw_path", None)
            st.session_state.pop("engineered_features", None)
            st.session_state.pop("engineered_windows", None)
            st.session_state.pop("detected_events", None)
            st.session_state.pop("event_features", None)
            st.session_state.pop("baseline_profile", None)
            st.session_state.pop("baseline_scored", None)
            st.session_state["import_fingerprint"] = upload_fingerprint
        dataframe = read_uploaded_data(upload_bytes, uploaded_file.name)
    except (ValueError, OSError, UnicodeDecodeError) as error:
        st.error(f"Could not read the dataset: {error}")
        return

    st.write(f"Loaded {len(dataframe):,} rows and {len(dataframe.columns)} columns.")
    detected = detect_column_mapping([str(column) for column in dataframe.columns])
    templates = ["Automatic detection", *list_templates()]
    selected_template = st.selectbox("Mapping template", templates, key="mapping_template_select")
    if selected_template != "Automatic detection":
        try:
            template_mapping = load_template(selected_template)
        except (OSError, ValueError) as error:
            st.error(f"Could not load mapping template: {error}")
        else:
            template_matches = []
            for column in dataframe.columns:
                canonical_field = template_mapping.get(str(column))
                template_matches.append(
                    ColumnMatch(
                        source_column=str(column),
                        canonical_field=canonical_field,
                        confidence=1.0 if canonical_field else 0.0,
                        method="saved template" if canonical_field else "template column not found",
                    )
                )
            detected = template_matches
    matches, unit_overrides = _mapping_editor([str(column) for column in dataframe.columns], detected)

    duplicate_fields = [
        field
        for field in {match.canonical_field for match in matches if match.canonical_field}
        if sum(match.canonical_field == field for match in matches) > 1
    ]
    if duplicate_fields:
        st.warning("Each canonical field can be assigned once. Resolve duplicate mappings before applying.")
        return

    template_name = st.text_input("Save mapping template as (optional)", key="mapping_template_name")
    action_columns = st.columns(2)
    with action_columns[0]:
        apply_mapping = st.button("Apply mapping", type="primary")
    with action_columns[1]:
        save_mapping_template = st.button("Save mapping template")

    mapping_dict = {
        match.source_column: match.canonical_field
        for match in matches
        if match.canonical_field
    }
    if save_mapping_template:
        try:
            path = save_template(template_name, mapping_dict)
        except ValueError as error:
            st.error(str(error))
        else:
            st.success(f"Saved mapping template `{path.stem}`.")

    if apply_mapping:
        raw_path = _save_raw_upload(uploaded_file)
        standardized = standardize_dataframe(dataframe, matches, unit_overrides=unit_overrides)
        st.session_state["standardized_import"] = standardized
        st.session_state["standardized_raw_path"] = raw_path

    standardized = st.session_state.get("standardized_import")
    if standardized is None:
        st.subheader("Original preview")
        st.dataframe(dataframe.head(20), use_container_width=True, hide_index=True)
        return

    st.success(f"Standardized data ready. Raw file preserved at `{st.session_state['standardized_raw_path']}`.")
    st.subheader("Standardized preview")
    st.dataframe(standardized.head(20), use_container_width=True, hide_index=True)
    report = analyze_quality(standardized)
    st.subheader("Data Quality Report")
    metric_columns = st.columns(5)
    metric_columns[0].metric("Quality score", f"{report.score}/100")
    metric_columns[1].metric("Sampling interval", f"{report.sampling_interval_minutes:.1f} min" if report.sampling_interval_minutes else "Unknown")
    metric_columns[2].metric("Missing samples", f"{report.missing_samples:,}")
    metric_columns[3].metric("Duplicate timestamps", f"{report.duplicate_timestamps:,}")
    metric_columns[4].metric(
        "Invalid / impossible",
        f"{sum(report.invalid_numeric.values()) + sum(report.impossible_values.values()):,}",
    )
    if report.issues:
        for issue in report.issues:
            st.warning(issue)
    else:
        st.success("No quality issues detected by the current checks.")

    interpolate_short_gaps = st.checkbox(
        "Interpolate short numeric gaps",
        help="Only interior gaps up to the selected sample count are filled. Larger gaps remain missing.",
        key="quality_interpolate_short_gaps",
    )
    max_gap_samples = st.number_input(
        "Maximum gap size to interpolate (samples)",
        min_value=1,
        max_value=24,
        value=2,
        step=1,
        key="quality_max_gap_samples",
    )
    cleaned = clean_dataframe(
        standardized,
        interpolate_short_gaps=interpolate_short_gaps,
        max_gap_samples=max_gap_samples,
    )
    st.write(f"Cleaned dataset: {len(cleaned):,} rows; standardized source remains unchanged.")
    st.download_button(
        "Download standardized CSV",
        data=standardized.to_csv(index=False).encode("utf-8"),
        file_name="standardized_esp_data.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download cleaned CSV",
        data=cleaned.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_esp_data.csv",
        mime="text/csv",
    )
