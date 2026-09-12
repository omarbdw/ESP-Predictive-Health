import pandas as pd

from esp_predictive_health.core.column_mapping import detect_column_mapping
from esp_predictive_health.core.quality import analyze_quality, clean_dataframe
from esp_predictive_health.core.units import convert_series, detect_unit


def test_common_esp_historian_abbreviations_map():
    matches = detect_column_mapping(["AMA [A]", "Pi [PSI]", "Pd [PSI]", "Tm [F]", "VX [G]"])
    mapping = {match.source_column: match.canonical_field for match in matches}

    assert mapping == {
        "AMA [A]": "motor_current_a",
        "Pi [PSI]": "intake_pressure_psi",
        "Pd [PSI]": "discharge_pressure_psi",
        "Tm [F]": "motor_temperature_c",
        "VX [G]": "vibration",
    }


def test_vibration_g_is_preserved_as_acceleration():
    assert detect_unit("VX [G]", "vibration") == "g"
    values = convert_series(pd.Series([0.5]), "g", "vibration")
    assert values.iloc[0] == 0.5


def test_invalid_numeric_values_are_reported_and_cleaned():
    source = pd.DataFrame(
        {
            "timestamp": ["2026-01-01 00:00:00", "2026-01-01 00:05:00"],
            "motor_current_a": ["bad", -2],
        }
    )

    report = analyze_quality(source)
    cleaned = clean_dataframe(source)

    assert report.invalid_numeric["motor_current_a"] == 1
    assert report.impossible_values["motor_current_a"] == 1
    assert pd.isna(cleaned.loc[0, "motor_current_a"])
    assert pd.isna(cleaned.loc[1, "motor_current_a"])
