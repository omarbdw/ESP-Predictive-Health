from pathlib import Path

import pandas as pd

from esp_predictive_health.core.column_mapping import detect_column_mapping, match_column
from esp_predictive_health.core.data_import import read_uploaded_data, standardize_dataframe
from esp_predictive_health.core import mapping_templates


def test_vendor_aliases_standardize_to_canonical_fields():
    source = pd.DataFrame(
        {
            "Time": ["2026-01-01T00:00:00"],
            "Amps": [42.0],
            "Motor Temp": [80.0],
            "PIP": [500.0],
            "Operator Note": ["steady"],
        }
    )
    matches = detect_column_mapping([str(column) for column in source.columns])
    standardized = standardize_dataframe(source, matches)

    assert standardized.loc[0, "motor_current_a"] == 42.0
    assert standardized.loc[0, "motor_temperature_c"] == 80.0
    assert standardized.loc[0, "intake_pressure_psi"] == 500.0
    assert standardized.loc[0, "Operator Note"] == "steady"
    assert "timestamp" in standardized.columns


def test_semicolon_delimited_csv_is_detected():
    source = read_uploaded_data(
        b"Time;Amps;PIP\n2026-01-01T00:00:00;42;500\n",
        "semicolon.csv",
    )

    matches = detect_column_mapping([str(column) for column in source.columns])
    standardized = standardize_dataframe(source, matches)

    assert list(source.columns) == ["Time", "Amps", "PIP"]
    assert standardized.loc[0, "motor_current_a"] == 42
    assert standardized.loc[0, "intake_pressure_psi"] == 500


def test_duplicate_canonical_matches_are_not_silently_assigned():
    matches = detect_column_mapping(["Amps", "Motor Current"])

    assert sum(match.canonical_field == "motor_current_a" for match in matches) == 1
    assert any(match.method == "duplicate canonical field" for match in matches)


def test_ambiguous_header_requires_manual_mapping():
    match = match_column("Temperature")

    assert match.canonical_field is None
    assert len(match.alternatives) >= 2
    assert sum(probability for _, probability in match.alternatives) <= 1.001


def test_mapping_template_round_trip(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(mapping_templates, "TEMPLATE_DIRECTORY", tmp_path)

    mapping_templates.save_template("Vendor A", {"Amps": "motor_current_a"})

    assert mapping_templates.list_templates() == ["vendor_a"]
    assert mapping_templates.load_template("vendor_a") == {"Amps": "motor_current_a"}
