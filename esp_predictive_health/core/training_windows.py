"""Case-level pre-failure training-window generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from esp_predictive_health.cases.schemas import CaseRecord

DEFAULT_PRE_FAILURE_DAYS: tuple[int, ...] = (1, 3, 7, 14, 30)
TRAINING_CONFIRMATION_WEIGHTS: dict[str, float] = {
    "A = Confirmed": 1.0,
    "B = Highly Probable": 0.7,
}


@dataclass(frozen=True)
class TrainingWindowConfig:
    """Rules for selecting case-level pre-failure windows."""

    pre_failure_days: tuple[int, ...] = DEFAULT_PRE_FAILURE_DAYS
    minimum_rows: int = 2
    include_highly_probable: bool = True


def _numeric_columns(dataframe: pd.DataFrame, timestamp_column: str) -> list[str]:
    columns: list[str] = []
    for column in dataframe.columns:
        if column == timestamp_column:
            continue
        if pd.api.types.is_numeric_dtype(dataframe[column]):
            columns.append(column)
    return columns


def _aggregate_window(dataframe: pd.DataFrame, numeric_columns: list[str]) -> dict[str, float]:
    values: dict[str, float] = {}
    for column in numeric_columns:
        series = pd.to_numeric(dataframe[column], errors="coerce")
        non_null = series.dropna()
        if non_null.empty:
            values[f"{column}_mean"] = np.nan
            values[f"{column}_std"] = np.nan
            values[f"{column}_min"] = np.nan
            values[f"{column}_max"] = np.nan
            values[f"{column}_slope"] = np.nan
            continue
        values[f"{column}_mean"] = float(non_null.mean())
        values[f"{column}_std"] = float(non_null.std(ddof=0))
        values[f"{column}_min"] = float(non_null.min())
        values[f"{column}_max"] = float(non_null.max())
        if len(non_null) >= 2:
            positions = np.arange(len(non_null), dtype=float)
            values[f"{column}_slope"] = float(np.polyfit(positions, non_null.to_numpy(), 1)[0])
        else:
            values[f"{column}_slope"] = np.nan
    return values


def build_training_windows(
    dataframe: pd.DataFrame,
    cases: Iterable[CaseRecord],
    *,
    config: TrainingWindowConfig | None = None,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    """Create one aggregated training row per eligible case and pre-failure window."""
    config = config or TrainingWindowConfig()
    if timestamp_column not in dataframe.columns:
        raise ValueError(f"Required timestamp column is missing: {timestamp_column}")
    if "well_id" not in dataframe.columns:
        raise ValueError("Required telemetry identity column is missing: well_id")
    windows = tuple(sorted({int(days) for days in config.pre_failure_days if int(days) > 0}))
    if not windows:
        raise ValueError("At least one positive pre-failure window is required.")

    telemetry = dataframe.copy()
    telemetry["well_id"] = telemetry["well_id"].astype("string").str.strip()
    telemetry[timestamp_column] = pd.to_datetime(telemetry[timestamp_column], errors="coerce")
    telemetry = telemetry.dropna(subset=[timestamp_column]).sort_values(timestamp_column)
    numeric_columns = _numeric_columns(telemetry, timestamp_column)
    records: list[dict[str, object]] = []

    for case in cases:
        weight = TRAINING_CONFIRMATION_WEIGHTS.get(case.confirmation_level)
        if weight is None or (not config.include_highly_probable and case.confirmation_level != "A = Confirmed"):
            continue
        if case.failure_date is None:
            continue
        if not case.well_id.strip():
            raise ValueError(f"Case {case.case_id} is missing well_id; telemetry cannot be associated safely.")
        case_telemetry = telemetry[telemetry["well_id"] == case.well_id.strip()]
        if case_telemetry.empty:
            continue
        failure_timestamp = pd.Timestamp(case.failure_date)
        for days in windows:
            window_start = failure_timestamp - pd.Timedelta(days=days)
            selection = case_telemetry[
                case_telemetry[timestamp_column].ge(window_start)
                & case_telemetry[timestamp_column].le(failure_timestamp)
            ]
            if len(selection) < config.minimum_rows:
                continue
            record: dict[str, object] = {
                "case_id": case.case_id,
                "well_name": case.well_name,
                "root_cause": case.root_cause,
                "confirmation_level": case.confirmation_level,
                "sample_weight": weight,
                "window_days": days,
                "window_start": window_start,
                "window_end": failure_timestamp,
                "row_count": len(selection),
                "missing_rate": float(selection[numeric_columns].isna().mean().mean()) if numeric_columns else 1.0,
            }
            record.update(_aggregate_window(selection, numeric_columns))
            records.append(record)

    if not records:
        return pd.DataFrame(
            columns=[
                "case_id",
                "well_name",
                "root_cause",
                "confirmation_level",
                "sample_weight",
                "window_days",
                "window_start",
                "window_end",
                "row_count",
                "missing_rate",
            ]
        )
    return pd.DataFrame(records).sort_values(["case_id", "window_days"]).reset_index(drop=True)
