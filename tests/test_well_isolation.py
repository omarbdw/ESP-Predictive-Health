from datetime import datetime

import pandas as pd

from esp_predictive_health.cases.schemas import CaseRecord
from esp_predictive_health.core.training_windows import TrainingWindowConfig, build_training_windows


def test_training_window_never_mixes_wells():
    telemetry = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-09", periods=4, freq="12h"),
            "well_id": ["W-001", "W-002", "W-001", "W-002"],
            "motor_current_a": [10.0, 900.0, 12.0, 900.0],
        }
    )
    case = CaseRecord(
        case_id="CASE-1",
        well_name="Well One",
        field_name="Field",
        well_type="Oil Producer",
        case_start=None,
        case_end=None,
        symptom_start=None,
        failure_date=datetime(2026, 1, 10).date(),
        root_cause="GAS",
        secondary_cause=None,
        observed_symptoms="",
        confirmation_level="A = Confirmed",
        confirmation_method="Report",
        esp_model="",
        pump_model="",
        motor_hp=None,
        notes="",
        raw_data_path=None,
        created_at=datetime(2026, 1, 1),
        updated_at=datetime(2026, 1, 1),
        well_id="W-001",
    )

    windows = build_training_windows(
        telemetry,
        [case],
        config=TrainingWindowConfig(pre_failure_days=(1,), minimum_rows=2),
    )

    assert len(windows) == 1
    assert windows.loc[0, "motor_current_a_mean"] == 11.0
