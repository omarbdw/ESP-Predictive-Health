from datetime import datetime

import pandas as pd

from esp_predictive_health.cases.schemas import CaseRecord
from esp_predictive_health.core.training_windows import TrainingWindowConfig, build_training_windows


def make_case(case_id: str, level: str, root_cause: str = "GAS") -> CaseRecord:
    now = datetime(2026, 1, 10)
    return CaseRecord(
        case_id=case_id,
        well_name="W-001",
        field_name="Field A",
        well_type="Oil Producer",
        case_start=None,
        case_end=None,
        symptom_start=None,
        failure_date=now.date(),
        root_cause=root_cause,
        secondary_cause=None,
        observed_symptoms="",
        confirmation_level=level,
        confirmation_method="Intervention report",
        esp_model="",
        pump_model="",
        motor_hp=None,
        notes="",
        raw_data_path=None,
        created_at=now,
        updated_at=now,
        well_id="W-001",
    )


def test_training_windows_are_case_level_and_weighted():
    telemetry = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-08", periods=5, freq="12h"),
            "well_id": ["W-001"] * 5,
            "motor_current_a": [100, 105, 110, 115, 120],
            "flow_rate_bpd": [1000, 990, 980, 970, 960],
        }
    )
    cases = [make_case("A", "A = Confirmed"), make_case("B", "B = Highly Probable"), make_case("C", "C = Suspected")]

    windows = build_training_windows(
        telemetry,
        cases,
        config=TrainingWindowConfig(pre_failure_days=(1, 3), minimum_rows=2),
    )

    assert set(windows["case_id"]) == {"A", "B"}
    assert set(windows["sample_weight"]) == {1.0, 0.7}
    assert len(windows) == 4
    assert "motor_current_a_mean" in windows.columns
    assert "motor_current_a_slope" in windows.columns
    assert set(windows["well_name"]) == {"W-001"}


def test_training_windows_skip_cases_without_enough_telemetry():
    telemetry = pd.DataFrame(
        {"timestamp": ["2026-01-10"], "well_id": ["W-001"], "motor_current_a": [100.0]}
    )

    windows = build_training_windows(
        telemetry,
        [make_case("A", "A = Confirmed")],
        config=TrainingWindowConfig(pre_failure_days=(1,), minimum_rows=2),
    )

    assert windows.empty
