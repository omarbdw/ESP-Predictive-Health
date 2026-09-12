"""Canonical ESP fields and explainable source-column matching."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from math import exp

CANONICAL_FIELDS: tuple[str, ...] = (
    "timestamp",
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
    "esp_status",
    "trip_reason",
)

FIELD_LABELS: dict[str, str] = {
    "timestamp": "Timestamp",
    "frequency_hz": "Frequency (Hz)",
    "motor_current_a": "Motor current (A)",
    "motor_voltage_v": "Motor voltage (V)",
    "intake_pressure_psi": "Intake pressure (psi)",
    "discharge_pressure_psi": "Discharge pressure (psi)",
    "wellhead_pressure_psi": "Wellhead pressure (psi)",
    "motor_temperature_c": "Motor temperature (C)",
    "intake_temperature_c": "Intake temperature (C)",
    "flow_rate_bpd": "Flow rate (bpd)",
    "oil_rate_bopd": "Oil rate (bopd)",
    "water_rate_bwpd": "Water rate (bwpd)",
    "water_cut_pct": "Water cut (%)",
    "vibration": "Vibration",
    "esp_status": "ESP status",
    "trip_reason": "Trip reason",
}

ALIASES: dict[str, tuple[str, ...]] = {
    "timestamp": ("timestamp", "time", "date time", "datetime", "event time"),
    "frequency_hz": (
        "frequency",
        "frequency hz",
        "freq",
        "speed hz",
        "speed rpm",
        "pump speed",
        "pump speed rpm",
        "hz",
    ),
    "motor_current_a": (
        "amps",
        "motor amps",
        "motor current",
        "current",
        "i motor",
        "motor current a",
        "ama",
        "ama a",
    ),
    "motor_voltage_v": ("volts", "motor volts", "motor voltage", "voltage", "motor voltage v"),
    "intake_pressure_psi": (
        "intake pressure",
        "pip",
        "pi",
        "pump intake pressure",
        "intake pressure psi",
        "pi psi",
    ),
    "discharge_pressure_psi": (
        "discharge pressure",
        "pd",
        "pump discharge pressure",
        "discharge pressure psi",
        "pd psi",
    ),
    "wellhead_pressure_psi": ("wellhead pressure", "whp", "well head pressure", "wellhead pressure psi"),
    "motor_temperature_c": (
        "motor temperature",
        "motor temp",
        "motor temperature c",
        "tm",
        "tm f",
    ),
    "intake_temperature_c": ("intake temperature", "intake temp", "intake temperature c"),
    "flow_rate_bpd": ("flow rate", "total flow", "liquid rate", "flow bpd", "flow rate bpd"),
    "oil_rate_bopd": ("oil rate", "oil production", "oil rate bopd"),
    "water_rate_bwpd": ("water rate", "water production", "water rate bwpd"),
    "water_cut_pct": ("water cut", "watercut", "water cut percent", "water cut pct"),
    "vibration": ("vibration", "vibration level", "vibration mm s", "vx", "vx g"),
    "esp_status": ("esp status", "pump status", "run status", "status"),
    "trip_reason": ("trip reason", "shutdown reason", "trip code", "fault reason"),
}


def normalize_column_name(value: str) -> str:
    """Normalize a vendor column name for comparison."""
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


@dataclass(frozen=True)
class ColumnMatch:
    """Explain how one source column maps to a canonical field."""

    source_column: str
    canonical_field: str | None
    confidence: float
    method: str
    alternatives: tuple[tuple[str, float], ...] = ()


def _similarity(source: str, alias: str) -> float:
    source_tokens = set(source.split())
    alias_tokens = set(alias.split())
    token_score = len(source_tokens & alias_tokens) / max(len(alias_tokens), 1)
    sequence_score = SequenceMatcher(None, source, alias).ratio()
    return max(token_score, sequence_score)


def _rank_candidates(normalized_source: str) -> list[tuple[str, float, str]]:
    scores: dict[str, tuple[float, str]] = {}
    for canonical_field, aliases in ALIASES.items():
        best_score = 0.0
        best_method = "similar alias"
        for alias in aliases:
            normalized_alias = normalize_column_name(alias)
            score = _similarity(normalized_source, normalized_alias)
            method = "exact alias" if normalized_source == normalized_alias else "similar alias"
            if score > best_score:
                best_score = score
                best_method = method
        if best_score >= 0.45:
            scores[canonical_field] = (best_score, best_method)

    if not scores:
        return []
    highest = max(score for score, _ in scores.values())
    weights = {
        field: exp((score - highest) / 0.08)
        for field, (score, _) in scores.items()
    }
    total_weight = sum(weights.values())
    return sorted(
        [
            (field, round(weights[field] / total_weight, 3), method)
            for field, (_, method) in scores.items()
        ],
        key=lambda item: item[1],
        reverse=True,
    )


def match_column(source_column: str) -> ColumnMatch:
    """Return a ranked mapping suggestion without silently forcing ambiguity."""
    normalized_source = normalize_column_name(source_column)
    if not normalized_source:
        return ColumnMatch(source_column, None, 0.0, "empty source name")

    ranked = _rank_candidates(normalized_source)
    if not ranked:
        return ColumnMatch(source_column, None, 0.0, "no confident match")

    top_field, top_probability, method = ranked[0]
    second_probability = ranked[1][1] if len(ranked) > 1 else 0.0
    top_score = max(_similarity(normalized_source, alias) for alias in ALIASES[top_field])
    is_clear = top_probability >= 0.65 and top_probability - second_probability >= 0.1
    alternatives = tuple((field, probability) for field, probability, _ in ranked[:3])
    if top_score < 0.72 or not is_clear:
        return ColumnMatch(
            source_column,
            None,
            top_probability,
            "ambiguous or low-confidence match",
            alternatives,
        )
    return ColumnMatch(source_column, top_field, top_probability, method, alternatives)


def detect_column_mapping(source_columns: list[str]) -> list[ColumnMatch]:
    """Match source columns while preventing duplicate canonical assignments."""
    matches = sorted(
        (match_column(column) for column in source_columns),
        key=lambda item: (item.canonical_field is None, -item.confidence, item.source_column),
    )
    used_fields: set[str] = set()
    resolved: list[ColumnMatch] = []
    for match in matches:
        if match.canonical_field in used_fields:
            resolved.append(ColumnMatch(match.source_column, None, 0.0, "duplicate canonical field"))
        else:
            if match.canonical_field:
                used_fields.add(match.canonical_field)
            resolved.append(match)
    return sorted(resolved, key=lambda item: source_columns.index(item.source_column))
