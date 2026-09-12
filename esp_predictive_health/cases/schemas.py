"""Typed records used by the Phase 1 case-management workflow."""

from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    well_name: str
    field_name: str
    well_type: str
    case_start: date | None
    case_end: date | None
    symptom_start: date | None
    failure_date: date | None
    root_cause: str
    secondary_cause: str | None
    observed_symptoms: str
    confirmation_level: str
    confirmation_method: str
    esp_model: str
    pump_model: str
    motor_hp: float | None
    notes: str
    raw_data_path: str | None
    created_at: datetime
    updated_at: datetime
    well_id: str = ""
