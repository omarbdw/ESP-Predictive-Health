import pandas as pd

from esp_predictive_health.core.column_mapping import detect_column_mapping
from esp_predictive_health.core.data_import import standardize_dataframe
from esp_predictive_health.core.units import apply_unit_conversions, detect_unit


def test_detects_units_from_headers():
    assert detect_unit("Intake Pressure [kPa]", "intake_pressure_psi") == "kPa"
    assert detect_unit("Motor Temperature (F)", "motor_temperature_c") == "F"
    assert detect_unit("Pump Speed rpm", "frequency_hz") == "rpm"


def test_converts_pressure_temperature_and_frequency():
    source = pd.DataFrame(
        {
            "Time": ["2026-01-01T00:00:00"],
            "Intake Pressure [kPa]": [100.0],
            "Motor Temperature (F)": [212.0],
            "Pump Speed rpm": [3600.0],
        }
    )
    matches = detect_column_mapping([str(column) for column in source.columns])
    standardized = standardize_dataframe(source, matches)

    assert standardized.loc[0, "intake_pressure_psi"] == 14.50377377
    assert standardized.loc[0, "motor_temperature_c"] == 100.0
    assert standardized.loc[0, "frequency_hz"] == 60.0
    assert str(standardized.loc[0, "timestamp"]) == "2026-01-01 00:00:00"


def test_manual_unit_override_converts_rate():
    data = pd.DataFrame({"Flow": [100.0]})

    converted = apply_unit_conversions(
        data.rename(columns={"Flow": "flow_rate_bpd"}),
        {"Flow": "flow_rate_bpd"},
        {"Flow": "m3/d"},
    )

    assert converted.loc[0, "flow_rate_bpd"] == 628.981077
