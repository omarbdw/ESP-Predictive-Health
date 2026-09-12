import pandas as pd

from esp_predictive_health.core.quality import analyze_quality, clean_dataframe, quality_status


def test_repeated_date_only_observations_are_not_blocked_or_deleted():
    source = pd.DataFrame(
        {
            "timestamp": ["2024-11-16", "2024-11-16", "2024-11-17"],
            "motor_current_a": [29.0, 30.0, 31.0],
        }
    )

    report = analyze_quality(source)
    cleaned = clean_dataframe(source)

    assert report.date_only_timestamps
    assert report.duplicate_timestamps == 1
    assert quality_status(report) == "PASS_WITH_WARNINGS"
    assert len(cleaned) == 3


def test_exact_timestamp_duplicates_remain_blocked():
    source = pd.DataFrame(
        {
            "timestamp": ["2024-11-16 00:00:00", "2024-11-16 00:00:00"],
            "motor_current_a": [29.0, 30.0],
        }
    )

    report = analyze_quality(source)

    assert not report.date_only_timestamps
    assert quality_status(report) == "BLOCKED"
