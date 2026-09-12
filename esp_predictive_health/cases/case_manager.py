"""CRUD operations for confirmed ESP cases."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from esp_predictive_health.cases.schemas import CaseRecord
from esp_predictive_health.config.failure_classes import (
    CONFIRMATION_LEVELS,
    FAILURE_CLASSES,
    WELL_TYPES,
)
from esp_predictive_health.database.db import DEFAULT_DB_PATH, get_connection


def _iso(value: date | datetime | None) -> str | None:
    return value.isoformat() if value else None


def _date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def _datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _row_to_case(row: sqlite3.Row) -> CaseRecord:
    return CaseRecord(
        case_id=row["case_id"],
        well_name=row["well_name"],
        field_name=row["field_name"],
        well_type=row["well_type"],
        case_start=_date(row["case_start"]),
        case_end=_date(row["case_end"]),
        symptom_start=_date(row["symptom_start"]),
        failure_date=_date(row["failure_date"]),
        root_cause=row["root_cause"],
        secondary_cause=row["secondary_cause"],
        observed_symptoms=row["observed_symptoms"],
        confirmation_level=row["confirmation_level"],
        confirmation_method=row["confirmation_method"],
        esp_model=row["esp_model"],
        pump_model=row["pump_model"],
        motor_hp=row["motor_hp"],
        notes=row["notes"],
        raw_data_path=row["raw_data_path"],
        created_at=_datetime(row["created_at"]),
        updated_at=_datetime(row["updated_at"]),
        well_id=row["well_id"],
    )


def _validate_case(case: CaseRecord) -> None:
    if not case.well_name.strip():
        raise ValueError("Well name is required.")
    if case.well_type not in WELL_TYPES:
        raise ValueError(f"Unsupported well type: {case.well_type}")
    if case.root_cause not in FAILURE_CLASSES:
        raise ValueError(f"Unsupported root cause: {case.root_cause}")
    if case.confirmation_level not in CONFIRMATION_LEVELS:
        raise ValueError(f"Unsupported confirmation level: {case.confirmation_level}")
    if case.motor_hp is not None and case.motor_hp < 0:
        raise ValueError("Motor horsepower cannot be negative.")
    if case.case_start and case.case_end and case.case_end < case.case_start:
        raise ValueError("Case end cannot be before case start.")
    if case.symptom_start and case.failure_date and case.failure_date < case.symptom_start:
        raise ValueError("Failure date cannot be before symptom start.")


def _payload(case: CaseRecord) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "well_name": case.well_name.strip(),
        "field_name": case.field_name.strip(),
        "well_type": case.well_type,
        "case_start": _iso(case.case_start),
        "case_end": _iso(case.case_end),
        "symptom_start": _iso(case.symptom_start),
        "failure_date": _iso(case.failure_date),
        "root_cause": case.root_cause,
        "secondary_cause": case.secondary_cause or None,
        "observed_symptoms": case.observed_symptoms.strip(),
        "confirmation_level": case.confirmation_level,
        "confirmation_method": case.confirmation_method.strip(),
        "esp_model": case.esp_model.strip(),
        "pump_model": case.pump_model.strip(),
        "motor_hp": case.motor_hp,
        "notes": case.notes.strip(),
        "raw_data_path": case.raw_data_path,
        "created_at": _iso(case.created_at),
        "updated_at": _iso(case.updated_at),
        "well_id": case.well_id.strip(),
    }


def create_case(
    *,
    well_name: str,
    well_id: str = "",
    field_name: str,
    well_type: str,
    case_start: date | None,
    case_end: date | None,
    symptom_start: date | None,
    failure_date: date | None,
    root_cause: str,
    secondary_cause: str | None,
    observed_symptoms: str,
    confirmation_level: str,
    confirmation_method: str,
    esp_model: str,
    pump_model: str,
    motor_hp: float | None,
    notes: str,
    raw_data_path: str | None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> CaseRecord:
    """Validate and persist one confirmed case without touching raw data."""
    now = datetime.now(timezone.utc).replace(microsecond=0)
    case = CaseRecord(
        case_id=f"ESP-{now:%Y%m%d}-{uuid4().hex[:8].upper()}",
        well_name=well_name,
        field_name=field_name,
        well_type=well_type,
        case_start=case_start,
        case_end=case_end,
        symptom_start=symptom_start,
        failure_date=failure_date,
        root_cause=root_cause,
        secondary_cause=secondary_cause,
        observed_symptoms=observed_symptoms,
        confirmation_level=confirmation_level,
        confirmation_method=confirmation_method,
        esp_model=esp_model,
        pump_model=pump_model,
        motor_hp=motor_hp,
        notes=notes,
        raw_data_path=raw_data_path,
        created_at=now,
        updated_at=now,
        well_id=well_id,
    )
    _validate_case(case)
    payload = _payload(case)
    columns = ", ".join(payload)
    placeholders = ", ".join(f":{column}" for column in payload)
    with get_connection(db_path) as connection:
        connection.execute(
            f"INSERT INTO cases ({columns}) VALUES ({placeholders})", payload
        )
        connection.commit()
    return case


def list_cases(
    *,
    root_cause: str | None = None,
    field_name: str | None = None,
    well_name: str | None = None,
    confirmation_level: str | None = None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> list[CaseRecord]:
    """Return cases matching optional exact filters, newest first."""
    clauses: list[str] = []
    parameters: dict[str, str] = {}
    for column, value in (
        ("root_cause", root_cause),
        ("field_name", field_name),
        ("well_name", well_name),
        ("confirmation_level", confirmation_level),
    ):
        if value:
            clauses.append(f"{column} = :{column}")
            parameters[column] = value
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with get_connection(db_path) as connection:
        rows = connection.execute(
            f"SELECT * FROM cases {where} ORDER BY updated_at DESC", parameters
        ).fetchall()
    return [_row_to_case(row) for row in rows]
