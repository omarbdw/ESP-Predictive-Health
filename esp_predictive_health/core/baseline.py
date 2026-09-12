"""Frequency-aware healthy-well baseline calculations for ESP telemetry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BASELINE_FIELDS: tuple[str, ...] = (
    "motor_current_a",
    "intake_pressure_psi",
    "discharge_pressure_psi",
    "pump_dp_psi",
    "flow_rate_bpd",
    "motor_temperature_c",
    "vibration",
)


@dataclass(frozen=True)
class BaselineConfig:
    """Configuration for frequency-conditioned baseline construction."""

    frequency_bin_hz: float = 1.0
    minimum_frequency_hz: float = 1.0
    minimum_samples_per_bin: int = 3
    matching_tolerance_hz: float = 2.0


def _numeric(dataframe: pd.DataFrame, field_name: str) -> pd.Series:
    if field_name not in dataframe.columns:
        return pd.Series(np.nan, index=dataframe.index, dtype="float64")
    return pd.to_numeric(dataframe[field_name], errors="coerce")


def _running_mask(dataframe: pd.DataFrame, minimum_frequency_hz: float) -> pd.Series:
    frequency = _numeric(dataframe, "frequency_hz")
    running = frequency.ge(minimum_frequency_hz).fillna(False)
    if "esp_status" in dataframe.columns:
        status = dataframe["esp_status"].astype("string").str.upper().str.strip()
        stopped = status.isin({"OFF", "STOP", "STOPPED", "TRIP", "FAULT"})
        usable_status = status.notna() & status.ne("") & ~status.isin({"NAN", "NONE", "<NA>"})
        if usable_status.any():
            running &= ~stopped
    return running


def _with_derived_fields(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()
    if "pump_dp_psi" not in result.columns:
        result["pump_dp_psi"] = _numeric(result, "discharge_pressure_psi") - _numeric(
            result, "intake_pressure_psi"
        )
    return result


def build_baseline(
    dataframe: pd.DataFrame,
    *,
    healthy_mask: pd.Series | None = None,
    config: BaselineConfig | None = None,
) -> pd.DataFrame:
    """Build median and variability profiles conditioned on pump frequency."""
    config = config or BaselineConfig()
    if "frequency_hz" not in dataframe.columns:
        raise ValueError("Frequency is required to build a baseline.")
    working = _with_derived_fields(dataframe)
    frequency = _numeric(working, "frequency_hz")
    eligible = _running_mask(working, config.minimum_frequency_hz)
    if healthy_mask is not None:
        eligible &= healthy_mask.reindex(working.index, fill_value=False).astype(bool)
    eligible_data = working.loc[eligible].copy()
    if eligible_data.empty:
        frequency = _numeric(working, "frequency_hz")
        available = int(frequency.notna().sum())
        maximum = frequency.max()
        raise ValueError(
            "No eligible running rows are available for the baseline. "
            f"Frequency values available: {available:,}; maximum: "
            f"{maximum if pd.notna(maximum) else 'missing'}. "
            "Map the vendor speed/frequency column to frequency_hz and ensure running values exceed the minimum threshold."
        )

    eligible_data["frequency_hz"] = _numeric(eligible_data, "frequency_hz")
    eligible_data["frequency_bin_hz"] = (
        eligible_data["frequency_hz"] / config.frequency_bin_hz
    ).round() * config.frequency_bin_hz
    grouped = eligible_data.groupby("frequency_bin_hz", sort=True)
    rows: list[dict[str, float | int]] = []
    for frequency_bin, group in grouped:
        if len(group) < config.minimum_samples_per_bin:
            continue
        row: dict[str, float | int] = {
            "frequency_bin_hz": float(frequency_bin),
            "sample_count": int(len(group)),
        }
        for field_name in BASELINE_FIELDS:
            values = _numeric(group, field_name).dropna()
            if values.empty:
                row[f"{field_name}_expected"] = np.nan
                row[f"{field_name}_lower"] = np.nan
                row[f"{field_name}_upper"] = np.nan
            else:
                row[f"{field_name}_expected"] = float(values.median())
                row[f"{field_name}_lower"] = float(values.quantile(0.1))
                row[f"{field_name}_upper"] = float(values.quantile(0.9))
        rows.append(row)
    if not rows:
        raise ValueError("No frequency bin contains enough healthy samples.")
    return pd.DataFrame(rows).sort_values("frequency_bin_hz").reset_index(drop=True)


def apply_baseline(
    dataframe: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    config: BaselineConfig | None = None,
) -> pd.DataFrame:
    """Attach nearest-frequency expectations and deviations to telemetry rows."""
    config = config or BaselineConfig()
    if "frequency_hz" not in dataframe.columns:
        raise ValueError("Frequency is required to apply a baseline.")
    working = _with_derived_fields(dataframe).copy()
    working["frequency_hz"] = _numeric(working, "frequency_hz")
    working["_row_order"] = np.arange(len(working))
    working = working.sort_values("frequency_hz")
    profile = baseline.sort_values("frequency_bin_hz")
    result = pd.merge_asof(
        working,
        profile,
        left_on="frequency_hz",
        right_on="frequency_bin_hz",
        direction="nearest",
        tolerance=config.matching_tolerance_hz,
    )
    for field_name in BASELINE_FIELDS:
        actual = _numeric(result, field_name)
        expected = _numeric(result, f"{field_name}_expected")
        result[f"{field_name}_deviation"] = actual - expected
        result[f"{field_name}_deviation_pct"] = actual.divide(expected.abs().where(expected.ne(0))) * 100
    return result.sort_values("_row_order").drop(columns=["_row_order"]).reset_index(drop=True)
