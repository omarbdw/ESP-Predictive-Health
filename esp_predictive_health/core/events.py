"""Auditable ESP event detection and event-derived features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EventDetectionConfig:
    """Configurable operating thresholds for event detection."""

    running_frequency_hz: float = 1.0
    frequency_change_hz: float = 2.0
    underload_current_a: float | None = None
    overload_current_a: float | None = None
    rapid_drawdown_psi_per_hour: float = -20.0
    pressure_buildup_psi_per_hour: float = 20.0
    sensor_dropout_minutes: float = 30.0


@dataclass(frozen=True)
class EventDetectionResult:
    """Detected events plus row-aligned features derived from those events."""

    events: pd.DataFrame
    features: pd.DataFrame


def _numeric(dataframe: pd.DataFrame, field_name: str) -> pd.Series:
    if field_name not in dataframe.columns:
        return pd.Series(np.nan, index=dataframe.index, dtype="float64")
    return pd.to_numeric(dataframe[field_name], errors="coerce")


def _running_mask(dataframe: pd.DataFrame, config: EventDetectionConfig) -> pd.Series:
    frequency = _numeric(dataframe, "frequency_hz")
    if "esp_status" not in dataframe.columns:
        return frequency.ge(config.running_frequency_hz).fillna(False)
    status = dataframe["esp_status"].astype("string").str.strip().str.upper()
    status_running = status.isin({"ON", "RUN", "RUNNING", "STARTED", "1", "TRUE"})
    return status_running | frequency.ge(config.running_frequency_hz).fillna(False)


def _append_event(
    events: list[dict[str, Any]],
    timestamp: pd.Timestamp,
    event_type: str,
    details: str,
    value: float | str | None = None,
) -> None:
    events.append(
        {
            "timestamp": timestamp,
            "event_type": event_type,
            "details": details,
            "value": value,
        }
    )


def detect_events(
    dataframe: pd.DataFrame,
    *,
    timestamp_column: str = "timestamp",
    config: EventDetectionConfig | None = None,
) -> EventDetectionResult:
    """Detect operational events without assigning a root-cause diagnosis."""
    config = config or EventDetectionConfig()
    if timestamp_column not in dataframe.columns:
        raise ValueError(f"Required timestamp column is missing: {timestamp_column}")

    working = dataframe.copy()
    working[timestamp_column] = pd.to_datetime(working[timestamp_column], errors="coerce")
    working = working.dropna(subset=[timestamp_column])
    working = working.sort_values(timestamp_column, kind="stable").reset_index(drop=True)
    timestamps = working[timestamp_column]
    running = _running_mask(working, config)
    previous_running = running.shift(fill_value=False)
    frequency = _numeric(working, "frequency_hz")
    current = _numeric(working, "motor_current_a")
    intake_pressure = _numeric(working, "intake_pressure_psi")
    events: list[dict[str, Any]] = []

    for index, timestamp in timestamps.items():
        if running.iloc[index] and not previous_running.iloc[index]:
            _append_event(events, timestamp, "START", "ESP transitioned to running")
        if not running.iloc[index] and previous_running.iloc[index]:
            _append_event(events, timestamp, "SHUTDOWN", "ESP transitioned to stopped")
        if index > 0 and running.iloc[index] and not running.iloc[index - 1]:
            if any(event["event_type"] == "SHUTDOWN" for event in events):
                _append_event(events, timestamp, "RESTART", "ESP started after a shutdown")

        if "trip_reason" in working.columns:
            reason = str(working.loc[index, "trip_reason"]).strip()
            if reason and reason.lower() not in {"nan", "none", "normal", "no trip"}:
                _append_event(events, timestamp, "TRIP", reason)

        if index > 0 and pd.notna(frequency.iloc[index]) and pd.notna(frequency.iloc[index - 1]):
            change = float(frequency.iloc[index] - frequency.iloc[index - 1])
            if abs(change) >= config.frequency_change_hz:
                _append_event(events, timestamp, "FREQUENCY_CHANGE", "Frequency changed", change)

        if running.iloc[index] and pd.notna(current.iloc[index]):
            if config.underload_current_a is not None and current.iloc[index] < config.underload_current_a:
                _append_event(events, timestamp, "UNDERLOAD", "Current below configured threshold", float(current.iloc[index]))
            if config.overload_current_a is not None and current.iloc[index] > config.overload_current_a:
                _append_event(events, timestamp, "OVERLOAD", "Current above configured threshold", float(current.iloc[index]))

        if index > 0 and pd.notna(intake_pressure.iloc[index]) and pd.notna(intake_pressure.iloc[index - 1]):
            elapsed_hours = (timestamps.iloc[index] - timestamps.iloc[index - 1]).total_seconds() / 3600
            if elapsed_hours > 0:
                pressure_rate = float(intake_pressure.iloc[index] - intake_pressure.iloc[index - 1]) / elapsed_hours
                if pressure_rate <= config.rapid_drawdown_psi_per_hour:
                    _append_event(events, timestamp, "RAPID_DRAWDOWN", "Intake pressure falling rapidly", pressure_rate)
                if not running.iloc[index] and pressure_rate >= config.pressure_buildup_psi_per_hour:
                    _append_event(events, timestamp, "PRESSURE_BUILDUP", "Intake pressure recovering while ESP is stopped", pressure_rate)

    dropout_fields = [field for field in ("frequency_hz", "motor_current_a", "intake_pressure_psi") if field in working.columns]
    if dropout_fields:
        dropout_mask = working[dropout_fields].isna().all(axis=1)
        for index in working.index[dropout_mask]:
            if index == 0:
                continue
            gap_minutes = (timestamps.iloc[index] - timestamps.iloc[index - 1]).total_seconds() / 60
            if gap_minutes >= config.sensor_dropout_minutes:
                _append_event(events, timestamps.iloc[index], "SENSOR_DROPOUT", "Required signals unavailable after a long gap", gap_minutes)

    event_frame = pd.DataFrame(events, columns=["timestamp", "event_type", "details", "value"])
    if event_frame.empty:
        event_frame = pd.DataFrame(columns=["timestamp", "event_type", "details", "value"])
    else:
        event_frame = event_frame.sort_values("timestamp").reset_index(drop=True)

    event_features = _event_features(working, event_frame, timestamp_column)
    return EventDetectionResult(events=event_frame, features=event_features)


def _event_features(
    dataframe: pd.DataFrame,
    events: pd.DataFrame,
    timestamp_column: str,
) -> pd.DataFrame:
    """Attach rolling event counts to every telemetry row."""
    result = dataframe.copy()
    event_timestamps = pd.to_datetime(events["timestamp"], errors="coerce") if not events.empty else pd.Series(dtype="datetime64[ns]")
    event_types = events["event_type"] if not events.empty else pd.Series(dtype="string")
    for event_type, column_name in (
        ("TRIP", "trips"),
        ("RESTART", "restarts"),
        ("UNDERLOAD", "underloads"),
        ("OVERLOAD", "overloads"),
    ):
        type_times = event_timestamps[event_types == event_type]
        for window_minutes in (60, 1440, 10080):
            counts = []
            for timestamp in pd.to_datetime(result[timestamp_column]):
                counts.append(int(((type_times <= timestamp) & (type_times >= timestamp - pd.Timedelta(minutes=window_minutes))).sum()))
            result[f"{column_name}_last_{window_minutes // 60}h"] = counts
    return result
