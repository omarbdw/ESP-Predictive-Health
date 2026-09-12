"""Time-aware ESP feature engineering for standardized telemetry."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

DEFAULT_ROLLING_WINDOWS_MINUTES: tuple[int, ...] = (30, 60, 360, 1440, 4320, 10080)
ROLLING_FIELDS: tuple[str, ...] = (
    "motor_current_a",
    "flow_rate_bpd",
    "intake_pressure_psi",
    "discharge_pressure_psi",
    "motor_temperature_c",
    "vibration",
)


def _numeric(dataframe: pd.DataFrame, field_name: str) -> pd.Series:
    if field_name not in dataframe.columns:
        return pd.Series(np.nan, index=dataframe.index, dtype="float64")
    return pd.to_numeric(dataframe[field_name], errors="coerce")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.where(denominator.ne(0))
    return numerator.divide(denominator)


def _slope(values: np.ndarray, elapsed_hours: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(elapsed_hours)
    if valid.sum() < 2 or np.ptp(elapsed_hours[valid]) == 0:
        return np.nan
    return float(np.polyfit(elapsed_hours[valid], values[valid], 1)[0])


def _time_window_slopes(
    timestamps: pd.Series,
    values: pd.Series,
    windows_minutes: Iterable[int],
) -> dict[str, pd.Series]:
    timestamp_values = pd.to_datetime(timestamps).reset_index(drop=True)
    numeric_values = pd.to_numeric(values, errors="coerce").reset_index(drop=True)
    result: dict[str, pd.Series] = {}
    for window_minutes in windows_minutes:
        slopes = np.full(len(numeric_values), np.nan, dtype="float64")
        window_delta = pd.Timedelta(minutes=window_minutes)
        for position, timestamp in enumerate(timestamp_values):
            if pd.isna(timestamp):
                continue
            start = timestamp - window_delta
            window_timestamps = timestamp_values.iloc[: position + 1]
            mask = window_timestamps.ge(start) & window_timestamps.le(timestamp)
            window_values = numeric_values.iloc[: position + 1][mask]
            window_time = window_timestamps[mask]
            if len(window_values) >= 2:
                elapsed_hours = (window_time - timestamp).dt.total_seconds().to_numpy() / 3600.0
                slopes[position] = _slope(window_values.to_numpy(dtype="float64"), elapsed_hours)
        result[f"slope_{window_minutes}m"] = pd.Series(slopes)
    return result


def add_engineered_features(
    dataframe: pd.DataFrame,
    *,
    timestamp_column: str = "timestamp",
    rolling_windows_minutes: Iterable[int] = DEFAULT_ROLLING_WINDOWS_MINUTES,
) -> pd.DataFrame:
    """Return chronological telemetry with derived and time-aware rolling features."""
    if timestamp_column not in dataframe.columns:
        raise ValueError(f"Required timestamp column is missing: {timestamp_column}")

    windows = tuple(sorted(set(int(window) for window in rolling_windows_minutes if int(window) > 0)))
    if not windows:
        raise ValueError("At least one positive rolling window is required.")

    features = dataframe.copy()
    features[timestamp_column] = pd.to_datetime(features[timestamp_column], errors="coerce")
    features = features.dropna(subset=[timestamp_column])
    features = features.sort_values(timestamp_column, kind="stable").reset_index(drop=True)

    frequency = _numeric(features, "frequency_hz")
    current = _numeric(features, "motor_current_a")
    flow = _numeric(features, "flow_rate_bpd")
    intake_pressure = _numeric(features, "intake_pressure_psi")
    discharge_pressure = _numeric(features, "discharge_pressure_psi")

    features["pump_dp_psi"] = discharge_pressure - intake_pressure
    features["current_per_hz"] = _safe_divide(current, frequency)
    features["flow_per_hz"] = _safe_divide(flow, frequency)
    features["pump_dp_per_hz2"] = _safe_divide(features["pump_dp_psi"], frequency.pow(2))
    features["frequency_change_hz"] = frequency.diff()
    features["data_missing_signal_count"] = features.isna().sum(axis=1)

    indexed = features.set_index(timestamp_column, drop=False)
    for field_name in ROLLING_FIELDS:
        values = _numeric(indexed, field_name)
        if values.empty:
            continue
        for window_minutes in windows:
            window = values.rolling(f"{window_minutes}min", min_periods=2)
            suffix = f"{window_minutes}m"
            features[f"{field_name}_mean_{suffix}"] = window.mean().to_numpy()
            features[f"{field_name}_std_{suffix}"] = window.std().to_numpy()
            features[f"{field_name}_min_{suffix}"] = window.min().to_numpy()
            features[f"{field_name}_max_{suffix}"] = window.max().to_numpy()
            features[f"{field_name}_range_{suffix}"] = (window.max() - window.min()).to_numpy()
            features[f"{field_name}_cv_{suffix}"] = _safe_divide(window.std(), window.mean().abs()).to_numpy()

        slopes = _time_window_slopes(features[timestamp_column], values.reset_index(drop=True), windows)
        for window_minutes in windows:
            features[f"{field_name}_slope_{window_minutes}m"] = slopes[
                f"slope_{window_minutes}m"
            ].to_numpy()

    return features
