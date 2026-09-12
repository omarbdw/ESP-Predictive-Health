from datetime import date

import pytest

from esp_predictive_health.cases.case_manager import create_case, list_cases


def case_values(**overrides):
    values = {
        "well_name": "W-001",
        "field_name": "Test Field",
        "well_type": "Oil Producer",
        "case_start": date(2026, 1, 1),
        "case_end": date(2026, 1, 10),
        "symptom_start": date(2026, 1, 8),
        "failure_date": date(2026, 1, 10),
        "root_cause": "GAS",
        "secondary_cause": None,
        "observed_symptoms": "Underload trip",
        "confirmation_level": "A = Confirmed",
        "confirmation_method": "Intervention report",
        "esp_model": "ESP-X",
        "pump_model": "P-1",
        "motor_hp": 100.0,
        "notes": "Test record",
        "raw_data_path": None,
    }
    values.update(overrides)
    return values


def test_create_and_filter_case(tmp_path):
    first = create_case(**case_values(db_path=tmp_path / "cases.db"))
    create_case(
        **case_values(
            well_name="W-002",
            root_cause="TUBING_LEAK",
            db_path=tmp_path / "cases.db",
        )
    )

    cases = list_cases(root_cause="GAS", db_path=tmp_path / "cases.db")

    assert len(cases) == 1
    assert cases[0].case_id == first.case_id
    assert cases[0].well_name == "W-001"


def test_rejects_invalid_case_dates(tmp_path):
    with pytest.raises(ValueError, match="Failure date cannot be before symptom start"):
        create_case(
            **case_values(
                symptom_start=date(2026, 1, 10),
                failure_date=date(2026, 1, 9),
                db_path=tmp_path / "cases.db",
            )
        )
