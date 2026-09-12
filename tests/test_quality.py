import pandas as pd

from esp_predictive_health.core.quality import analyze_quality, clean_dataframe


def test_quality_report_detects_timestamp_and_sensor_issues():
    source = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01 00:00:00",
                "2026-01-01 00:05:00",
                "2026-01-01 00:05:00",
                "not-a-time",
                "2026-01-01 00:30:00",
            ],
            "motor_current_a": [10, 10, 10, 10, -2],
            "water_cut_pct": [50, 50, 50, 50, 120],
        }
    )

    report = analyze_quality(source)

    assert report.timestamp_invalid == 1
    assert report.duplicate_timestamps == 1
    assert report.impossible_values["motor_current_a"] == 1
    assert report.impossible_values["water_cut_pct"] == 1
    assert report.frozen_sensors["motor_current_a"] >= 1
    assert report.score < 100
    assert report.issues


def test_clean_dataframe_keeps_source_unchanged_and_interpolates_short_gap():
    source = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01 00:10:00",
                "2026-01-01 00:00:00",
                "2026-01-01 00:00:00",
                "2026-01-01 00:05:00",
            ],
            "motor_current_a": [30.0, 10.0, 999.0, None],
        }
    )

    cleaned = clean_dataframe(source, interpolate_short_gaps=True, max_gap_samples=1)

    assert len(source) == 4
    assert len(cleaned) == 3
    assert cleaned["timestamp"].is_monotonic_increasing
    assert cleaned["motor_current_a"].isna().sum() == 0
