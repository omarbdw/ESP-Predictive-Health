import pandas as pd

from esp_predictive_health.core.baseline import BaselineConfig, apply_baseline, build_baseline


def test_baseline_is_conditioned_on_frequency_and_running_rows():
    source = pd.DataFrame(
        {
            "frequency_hz": [49.8, 50.0, 50.2, 60.0, 60.1, 0.0],
            "motor_current_a": [100, 102, 98, 130, 132, 0],
            "intake_pressure_psi": [500, 500, 500, 480, 480, 520],
            "discharge_pressure_psi": [700, 700, 700, 690, 690, 520],
            "flow_rate_bpd": [1000, 1000, 1000, 1200, 1200, 0],
            "esp_status": ["ON", "ON", "ON", "ON", "ON", "OFF"],
        }
    )

    baseline = build_baseline(
        source,
        config=BaselineConfig(frequency_bin_hz=1.0, minimum_samples_per_bin=2),
    )
    scored = apply_baseline(source, baseline)

    assert len(baseline) == 2
    assert baseline.loc[baseline["frequency_bin_hz"] == 50, "sample_count"].iloc[0] == 3
    assert scored.loc[0, "pump_dp_psi"] == 200
    assert scored.loc[0, "motor_current_a_expected"] == 100
    assert scored.loc[3, "motor_current_a_expected"] == 131
    assert pd.isna(scored.loc[5, "motor_current_a_expected"])


def test_baseline_requires_eligible_samples():
    source = pd.DataFrame({"frequency_hz": [0, 0], "motor_current_a": [0, 0]})

    try:
        build_baseline(source)
    except ValueError as error:
        assert "eligible" in str(error)
    else:
        raise AssertionError("Expected baseline construction to fail without running samples")
