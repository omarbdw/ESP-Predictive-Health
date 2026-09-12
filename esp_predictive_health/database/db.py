"""SQLite connection and schema management for confirmed cases."""

from __future__ import annotations

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "cases" / "cases.db"


CREATE_CASES_TABLE = """
CREATE TABLE IF NOT EXISTS cases (
    case_id TEXT PRIMARY KEY,
    well_name TEXT NOT NULL,
    field_name TEXT NOT NULL DEFAULT '',
    well_type TEXT NOT NULL,
    case_start TEXT,
    case_end TEXT,
    symptom_start TEXT,
    failure_date TEXT,
    root_cause TEXT NOT NULL,
    secondary_cause TEXT,
    observed_symptoms TEXT NOT NULL DEFAULT '',
    confirmation_level TEXT NOT NULL,
    confirmation_method TEXT NOT NULL DEFAULT '',
    esp_model TEXT NOT NULL DEFAULT '',
    pump_model TEXT NOT NULL DEFAULT '',
    motor_hp REAL,
    notes TEXT NOT NULL DEFAULT '',
    raw_data_path TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""


def get_connection(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open an initialized SQLite connection with dictionary-like rows."""
    resolved_path = Path(db_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(resolved_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(CREATE_CASES_TABLE)
    connection.commit()
    return connection
