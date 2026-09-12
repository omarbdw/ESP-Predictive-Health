"""Unit detection and conversion into the canonical ESP unit system."""

from __future__ import annotations

import re
from collections.abc import Mapping

import pandas as pd

CANONICAL_UNITS: dict[str, str] = {
    "well_id": "native",
    "timestamp": "datetime",
    "frequency_hz": "Hz",
    "motor_current_a": "A",
    "motor_voltage_v": "V",
    "intake_pressure_psi": "psi",
    "discharge_pressure_psi": "psi",
    "wellhead_pressure_psi": "psi",
    "motor_temperature_c": "C",
    "intake_temperature_c": "C",
    "flow_rate_bpd": "bpd",
    "oil_rate_bopd": "bopd",
    "water_rate_bwpd": "bwpd",
    "water_cut_pct": "%",
    "vibration": "native",
    "esp_status": "native",
    "trip_reason": "native",
}

UNIT_OPTIONS: dict[str, tuple[str, ...]] = {
    "well_id": ("native",),
    "timestamp": ("datetime",),
    "frequency_hz": ("Hz", "rpm"),
    "motor_current_a": ("A", "mA", "kA"),
    "motor_voltage_v": ("V", "kV"),
    "intake_pressure_psi": ("psi", "kPa", "bar", "MPa"),
    "discharge_pressure_psi": ("psi", "kPa", "bar", "MPa"),
    "wellhead_pressure_psi": ("psi", "kPa", "bar", "MPa"),
    "motor_temperature_c": ("C", "F", "K"),
    "intake_temperature_c": ("C", "F", "K"),
    "flow_rate_bpd": ("bpd", "m3/d", "L/s"),
    "oil_rate_bopd": ("bopd", "m3/d", "L/s"),
    "water_rate_bwpd": ("bwpd", "m3/d", "L/s"),
    "water_cut_pct": ("%", "fraction"),
    "vibration": ("native", "mm/s", "in/s", "g"),
    "esp_status": ("native",),
    "trip_reason": ("native",),
}

_UNIT_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"(?:°|deg(?:rees)?\s*)?f(?:ahrenheit)?\b", "F"),
    (r"(?:°|deg(?:rees)?\s*)?c(?:elsius)?\b", "C"),
    (r"\bkelvin\b|\bk\b", "K"),
    (r"\bmpa\b", "MPa"),
    (r"\bkpa\b", "kPa"),
    (r"\bbar\b", "bar"),
    (r"\bpsi\b|\bpsig\b", "psi"),
    (r"\brpm\b", "rpm"),
    (r"\bkhz\b", "kHz"),
    (r"\bhz\b", "Hz"),
    (r"\bka\b", "kA"),
    (r"\bma\b", "mA"),
    (r"\ba\b|\bamp(?:s|ere)?\b", "A"),
    (r"\bkv\b", "kV"),
    (r"\bv\b|\bvolt(?:s)?\b", "V"),
    (r"\bm3\s*/?\s*d\b|\bsm3\s*/?\s*d\b", "m3/d"),
    (r"\bl\s*/?\s*s\b", "L/s"),
    (r"\bbpd\b|\bbbl\s*/?\s*d\b", "bpd"),
    (r"\bmm\s*/?\s*s\b", "mm/s"),
    (r"\bin\s*/?\s*s\b", "in/s"),
    (r"\bg\b|\baccel(?:eration)?\b", "g"),
    (r"\bpercent\b|%", "%"),
)


def detect_unit(source_column: str, canonical_field: str) -> str:
    """Infer a source unit from a header, falling back to the canonical unit."""
    normalized = str(source_column).lower().replace("μ", "u").replace("µ", "u")
    for pattern, unit in _UNIT_PATTERNS:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            if unit in UNIT_OPTIONS.get(canonical_field, ("native",)):
                return unit
    return CANONICAL_UNITS.get(canonical_field, "native")


def convert_series(series: pd.Series, source_unit: str, canonical_field: str) -> pd.Series:
    """Convert one mapped series into its canonical unit."""
    target_unit = CANONICAL_UNITS.get(canonical_field, "native")
    if canonical_field == "timestamp":
        return pd.to_datetime(series, errors="coerce")
    if source_unit == "native" or source_unit == target_unit:
        return series
    if canonical_field == "vibration" and source_unit == "g":
        return series

    numeric = pd.to_numeric(series, errors="coerce")
    if canonical_field == "frequency_hz" and source_unit == "rpm":
        return numeric / 60.0
    if canonical_field == "motor_current_a" and source_unit == "mA":
        return numeric / 1000.0
    if canonical_field == "motor_current_a" and source_unit == "kA":
        return numeric * 1000.0
    if canonical_field == "motor_voltage_v" and source_unit == "kV":
        return numeric * 1000.0
    if canonical_field in {"intake_pressure_psi", "discharge_pressure_psi", "wellhead_pressure_psi"}:
        return numeric * {"kPa": 0.1450377377, "bar": 14.5037738, "MPa": 145.037738}[source_unit]
    if canonical_field in {"motor_temperature_c", "intake_temperature_c"}:
        if source_unit == "F":
            return (numeric - 32.0) * (5.0 / 9.0)
        if source_unit == "K":
            return numeric - 273.15
    if canonical_field in {"flow_rate_bpd", "oil_rate_bopd", "water_rate_bwpd"}:
        if source_unit == "m3/d":
            return numeric * 6.28981077
        if source_unit == "L/s":
            return numeric * 543.43965
    if canonical_field == "water_cut_pct" and source_unit == "fraction":
        return numeric * 100.0
    if canonical_field == "vibration" and source_unit == "in/s":
        return numeric * 25.4
    raise ValueError(f"Unsupported conversion from {source_unit} to {target_unit} for {canonical_field}.")


def apply_unit_conversions(
    dataframe: pd.DataFrame,
    source_to_canonical: Mapping[str, str],
    unit_overrides: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Convert mapped columns while leaving unmapped columns unchanged."""
    converted = dataframe.copy()
    overrides = unit_overrides or {}
    for source_column, canonical_field in source_to_canonical.items():
        if canonical_field not in converted.columns:
            continue
        source_unit = overrides.get(source_column) or detect_unit(source_column, canonical_field)
        converted[canonical_field] = convert_series(
            converted[canonical_field], source_unit, canonical_field
        )
    return converted
