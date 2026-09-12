import pandas as pd

from esp_predictive_health.core.events import EventDetectionConfig, detect_events


def test_detects_operational_events_and_preserves_trip_reason():
    source = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="5min"),
            "frequency_hz": [0, 50, 50, 0, 50],
            "motor_current_a": [0, 100, 130, 0, 100],
            "intake_pressure_psi": [500, 500, 480, 520, 520],
            "esp_status": ["OFF", "ON", "ON", "OFF", "ON"],
            "trip_reason": ["", "", "OVERLOAD", "", ""],
        }
    )

    result = detect_events(
        source,
        config=EventDetectionConfig(
            overload_current_a=120,
            rapid_drawdown_psi_per_hour=-100,
            pressure_buildup_psi_per_hour=100,
        ),
    )

    event_types = result.events["event_type"].tolist()
    assert "START" in event_types
    assert "SHUTDOWN" in event_types
    assert "RESTART" in event_types
    assert "TRIP" in event_types
    assert "OVERLOAD" in event_types
    assert "RAPID_DRAWDOWN" in event_types
    assert "PRESSURE_BUILDUP" in event_types
    assert result.features.loc[4, "restarts_last_1h"] >= 1


def test_detects_underload_and_frequency_change():
    source = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="5min"),
            "frequency_hz": [40, 50, 50],
            "motor_current_a": [20, 40, 5],
        }
    )

    result = detect_events(
        source,
        config=EventDetectionConfig(underload_current_a=10, frequency_change_hz=5),
    )

    assert "FREQUENCY_CHANGE" in result.events["event_type"].tolist()
    assert "UNDERLOAD" in result.events["event_type"].tolist()
