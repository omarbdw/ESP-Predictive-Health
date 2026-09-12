"""Data-quality checks and conservative cleaning for standardized ESP data."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

NUMERIC_FIELDS: tuple[str, ...] = (
    "frequency_hz",
    "motor_current_a",
    "motor_voltage_v",
    "intake_pressure_psi",
    "discharge_pressure_psi",
    "wellhead_pressure_psi",
    "motor_temperature_c",
    "intake_temperature_c",
    "flow_rate_bpd",
    "oil_rate_bopd",
    "water_rate_bwpd",
    "water_cut_pct",
    "vibration",
)

PHYSICAL_RANGES: dict[str, tuple[float | None, float | None]] = {
    "frequency_hz": (0, 1000),
    "motor_current_a": (0, None),
    "motor_voltage_v": (0, None),
    "intake_pressure_psi": (0, None),
    "discharge_pressure_psi": (0, None),
    "wellhead_pressure_psi": (0, None),
    "motor_temperature_c": (-100, 300),
    "intake_temperature_c": (-100, 300),
    "flow_rate_bpd": (0, None),
    "oil_rate_bopd": (0, None),
    "water_rate_bwpd": (0, None),
    "water_cut_pct": (0, 100),
    "vibration": (0, None),
}


@dataclass(frozen=True)
class QualityReport:
    """Serializable summary of quality findings for one dataframe."""

    score: int
    row_count: int
    timestamp_invalid: int
    duplicate_timestamps: int
    non_monotonic_timestamps: int
    sampling_interval_minutes: float | None
    missing_samples: int
    missing_sample_rate: float
    irregular_intervals: int
    long_missing_intervals: int
    invalid_numeric: dict[str, int] = field(default_factory=dict)
    impossible_values: dict[str, int] = field(default_factory=dict)
    frozen_sensors: dict[str, int] = field(default_factory=dict)
    spikes: dict[str, int] = field(default_factory=dict)
    zero_periods: dict[str, int] = field(default_factory=dict)
    issues: tuple[str, ...] = ()


def _numeric_series(dataframe: pd.DataFrame, field_name: str) -> pd.Series:
    if field_name not in dataframe.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(dataframe[field_name], errors="coerce")


def _invalid_numeric_counts(dataframe: pd.DataFrame) -> dict[str, int]:
    invalid: dict[str, int] = {}
    for field_name in NUMERIC_FIELDS:
        if field_name not in dataframe.columns:
            continue
        source = dataframe[field_name]
        numeric = pd.to_numeric(source, errors="coerce")
        count = int((source.notna() & numeric.isna()).sum())
        if count:
            invalid[field_name] = count
    return invalid


def _frozen_run_count(series: pd.Series, minimum_points: int) -> int:
    values = series.dropna()
    if values.empty:
        return 0
    same_as_previous = values.eq(values.shift())
    groups = (~same_as_previous).cumsum()
    run_lengths = same_as_previous.groupby(groups).sum()
    return int((run_lengths >= minimum_points - 1).sum())


def _spike_count(series: pd.Series) -> int:
    values = series.dropna()
    if len(values) < 5:
        return 0
    changes = values.diff().abs().dropna()
    first_quartile = changes.quantile(0.25)
    third_quartile = changes.quantile(0.75)
    threshold = third_quartile + 3 * (third_quartile - first_quartile)
    if threshold <= 0:
        return 0
    return int((changes > threshold).sum())


def _physical_violations(dataframe: pd.DataFrame) -> dict[str, int]:
    violations: dict[str, int] = {}
    for field_name, (minimum, maximum) in PHYSICAL_RANGES.items():
        values = _numeric_series(dataframe, field_name)
        if values.empty:
            continue
        invalid = pd.Series(False, index=values.index)
        if minimum is not None:
            invalid |= values < minimum
        if maximum is not None:
            invalid |= values > maximum
        count = int(invalid.sum())
        if count:
            violations[field_name] = count
    return violations


def analyze_quality(
    dataframe: pd.DataFrame,
    *,
    timestamp_column: str = "timestamp",
    frozen_minimum_points: int = 3,
    long_gap_multiplier: float = 6.0,
) -> QualityReport:
    """Analyze timestamps, sampling, sensor behavior, and physical ranges."""
    row_count = len(dataframe)
    issues: list[str] = []
    if timestamp_column not in dataframe.columns:
        return QualityReport(
            score=0,
            row_count=row_count,
            timestamp_invalid=row_count,
            duplicate_timestamps=0,
            non_monotonic_timestamps=0,
            sampling_interval_minutes=None,
            missing_samples=0,
            missing_sample_rate=0.0,
            irregular_intervals=0,
            long_missing_intervals=0,
            issues=("Timestamp column is missing.",),
        )

    timestamps = pd.to_datetime(dataframe[timestamp_column], errors="coerce")
    timestamp_invalid = int(timestamps.isna().sum())
    valid_timestamps = timestamps.dropna()
    duplicate_timestamps = int(valid_timestamps.duplicated().sum())
    source_order_deltas = valid_timestamps.diff().dt.total_seconds().dropna()
    non_monotonic_timestamps = int((source_order_deltas < 0).sum())
    unique_timestamps = valid_timestamps.drop_duplicates().sort_values()
    date_only_timestamps = bool(
        not unique_timestamps.empty
        and (unique_timestamps.dt.hour == 0).all()
        and (unique_timestamps.dt.minute == 0).all()
        and (unique_timestamps.dt.second == 0).all()
    )
    deltas = unique_timestamps.diff().dt.total_seconds().dropna()
    positive_deltas = deltas[deltas > 0]

    sampling_interval_minutes: float | None = None
    missing_samples = 0
    missing_sample_rate = 0.0
    irregular_intervals = 0
    long_missing_intervals = 0
    if not positive_deltas.empty and not date_only_timestamps:
        median_interval = float(positive_deltas.median())
        sampling_interval_minutes = median_interval / 60.0
        missing_samples = int(
            sum(max(round(delta / median_interval) - 1, 0) for delta in positive_deltas)
        )
        missing_sample_rate = missing_samples / max(len(valid_timestamps) + missing_samples, 1)
        irregular_intervals = int(
            ((positive_deltas < median_interval * 0.8) | (positive_deltas > median_interval * 1.2)).sum()
        )
        long_missing_intervals = int(
            (positive_deltas > median_interval * long_gap_multiplier).sum()
        )

    invalid_numeric = _invalid_numeric_counts(dataframe)
    impossible_values = _physical_violations(dataframe)
    frozen_sensors = {
        field_name: _frozen_run_count(_numeric_series(dataframe, field_name), frozen_minimum_points)
        for field_name in NUMERIC_FIELDS
        if _frozen_run_count(_numeric_series(dataframe, field_name), frozen_minimum_points)
    }
    spikes = {
        field_name: _spike_count(_numeric_series(dataframe, field_name))
        for field_name in NUMERIC_FIELDS
        if _spike_count(_numeric_series(dataframe, field_name))
    }
    zero_periods = {
        field_name: int((_numeric_series(dataframe, field_name) == 0).sum())
        for field_name in NUMERIC_FIELDS
        if field_name in dataframe.columns and int((_numeric_series(dataframe, field_name) == 0).sum())
    }

    if timestamp_invalid:
        issues.append(f"{timestamp_invalid} invalid timestamp value(s).")
    if duplicate_timestamps:
        issues.append(f"{duplicate_timestamps} duplicate timestamp value(s).")
    if non_monotonic_timestamps:
        issues.append(f"{non_monotonic_timestamps} non-monotonic timestamp interval(s).")
    if missing_samples:
        issues.append(f"{missing_samples} estimated missing sample(s).")
    if long_missing_intervals:
        issues.append(f"{long_missing_intervals} long missing interval(s) detected.")
    if date_only_timestamps:
        issues.append("Timestamps contain dates only; telemetry sampling interval is unavailable.")
    if invalid_numeric:
        issues.append("Non-numeric sensor readings detected.")
    if impossible_values:
        issues.append("Impossible physical values detected.")
    if frozen_sensors:
        issues.append("Frozen sensor intervals detected.")
    if spikes:
        issues.append("Potential sensor spikes detected.")

    penalty = min(timestamp_invalid / max(row_count, 1) * 30, 30)
    penalty += 10 if duplicate_timestamps else 0
    penalty += 10 if non_monotonic_timestamps else 0
    penalty += min(missing_sample_rate * 30, 30)
    penalty += min(irregular_intervals / max(len(positive_deltas), 1) * 10, 10)
    penalty += min(sum(impossible_values.values()) / max(row_count, 1) * 20, 20)
    penalty += min(sum(invalid_numeric.values()) / max(row_count, 1) * 20, 20)
    penalty += 10 if frozen_sensors else 0
    penalty += min(sum(spikes.values()) / max(row_count, 1) * 10, 10)
    score = max(0, round(100 - penalty))

    return QualityReport(
        score=score,
        row_count=row_count,
        timestamp_invalid=timestamp_invalid,
        duplicate_timestamps=duplicate_timestamps,
        non_monotonic_timestamps=non_monotonic_timestamps,
        sampling_interval_minutes=sampling_interval_minutes,
        missing_samples=missing_samples,
        missing_sample_rate=missing_sample_rate,
        irregular_intervals=irregular_intervals,
        long_missing_intervals=long_missing_intervals,
        invalid_numeric=invalid_numeric,
        impossible_values=impossible_values,
        frozen_sensors=frozen_sensors,
        spikes=spikes,
        zero_periods=zero_periods,
        issues=tuple(issues),
    )


def clean_dataframe(
    dataframe: pd.DataFrame,
    *,
    timestamp_column: str = "timestamp",
    interpolate_short_gaps: bool = False,
    max_gap_samples: int = 2,
) -> pd.DataFrame:
    """Return a cleaned copy, optionally interpolating only short numeric gaps."""
    cleaned = dataframe.copy()
    if timestamp_column in cleaned.columns:
        cleaned[timestamp_column] = pd.to_datetime(cleaned[timestamp_column], errors="coerce")
        cleaned = cleaned.dropna(subset=[timestamp_column])
        cleaned = cleaned.drop_duplicates(subset=[timestamp_column], keep="first")
        cleaned = cleaned.sort_values(timestamp_column, kind="stable").reset_index(drop=True)

    for field_name, (minimum, maximum) in PHYSICAL_RANGES.items():
        if field_name not in cleaned.columns:
            continue
        numeric = pd.to_numeric(cleaned[field_name], errors="coerce")
        invalid = pd.Series(False, index=cleaned.index)
        if minimum is not None:
            invalid |= numeric < minimum
        if maximum is not None:
            invalid |= numeric > maximum
        cleaned[field_name] = numeric.mask(invalid)

    if interpolate_short_gaps:
        for field_name in NUMERIC_FIELDS:
            if field_name not in cleaned.columns:
                continue
            numeric = pd.to_numeric(cleaned[field_name], errors="coerce")
            cleaned[field_name] = numeric.interpolate(
                limit=max_gap_samples,
                limit_area="inside",
            )
    return cleaned
