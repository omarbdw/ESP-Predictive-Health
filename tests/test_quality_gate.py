from esp_predictive_health.core.quality import QualityReport, quality_status


def report(**overrides):
    values = {
        "score": 100,
        "row_count": 10,
        "timestamp_invalid": 0,
        "duplicate_timestamps": 0,
        "non_monotonic_timestamps": 0,
        "sampling_interval_minutes": 5.0,
        "missing_samples": 0,
        "missing_sample_rate": 0.0,
        "irregular_intervals": 0,
        "long_missing_intervals": 0,
    }
    values.update(overrides)
    return QualityReport(**values)


def test_quality_gate_blocks_critical_findings():
    assert quality_status(report(impossible_values={"motor_current_a": 1})) == "BLOCKED"
    assert quality_status(report(invalid_numeric={"flow_rate_bpd": 1})) == "BLOCKED"
    assert quality_status(report(duplicate_timestamps=1)) == "BLOCKED"


def test_quality_gate_allows_warnings_but_marks_them():
    assert quality_status(report(issues=("Long gap detected.",))) == "PASS_WITH_WARNINGS"
    assert quality_status(report()) == "PASS"
