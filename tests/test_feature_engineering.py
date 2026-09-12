import numpy as np
import pandas as pd

from esp_predictive_health.core.feature_engineering import add_engineered_features


def test_engineered_features_use_safe_esp_calculations():
    source = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="5min"),
            "frequency_hz": [0.0, 50.0, 50.0, 50.0],
            "motor_current_a": [0.0, 100.0, 110.0, 120.0],
            "flow_rate_bpd": [0.0, 1000.0, 1100.0, 1200.0],
            "intake_pressure_psi": [500.0, 500.0, 500.0, 500.0],
            "discharge_pressure_psi": [700.0, 700.0, 710.0, 720.0],
        }
    )

    features = add_engineered_features(source, rolling_windows_minutes=(10,))

    assert features.loc[0, "pump_dp_psi"] == 200.0
    assert np.isnan(features.loc[0, "current_per_hz"])
    assert features.loc[1, "current_per_hz"] == 2.0
    assert features.loc[2, "pump_dp_per_hz2"] == 0.084
    assert "motor_current_a_mean_10m" in features.columns
    assert "motor_current_a_slope_10m" in features.columns
    assert features.loc[3, "motor_current_a_slope_10m"] > 0


def test_engineering_features_sort_and_drop_invalid_timestamps():
    source = pd.DataFrame(
        {
            "timestamp": ["2026-01-01 00:10", "bad", "2026-01-01 00:00"],
            "frequency_hz": [50, 50, 50],
        }
    )

    features = add_engineered_features(source, rolling_windows_minutes=(30,))

    assert len(features) == 2
    assert features["timestamp"].is_monotonic_increasing
