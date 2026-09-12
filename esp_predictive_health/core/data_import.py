"""CSV/Excel loading and canonical dataframe construction."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from collections.abc import Mapping
from typing import BinaryIO

import pandas as pd

from esp_predictive_health.core.column_mapping import CANONICAL_FIELDS, ColumnMatch
from esp_predictive_health.core.units import apply_unit_conversions

SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".xls"}


def read_uploaded_data(file_data: bytes | BinaryIO, filename: str) -> pd.DataFrame:
    """Read a CSV or Excel upload without modifying the original bytes."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError("Supported file types are CSV, XLSX, and XLS.")
    stream = BytesIO(file_data) if isinstance(file_data, bytes) else file_data
    if suffix == ".csv":
        return pd.read_csv(stream, sep=None, engine="python")
    return pd.read_excel(stream)


def standardize_dataframe(
    dataframe: pd.DataFrame,
    matches: list[ColumnMatch],
    unit_overrides: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Return canonical fields with unit conversions and preserve unmapped data."""
    rename_map = {
        match.source_column: match.canonical_field
        for match in matches
        if match.canonical_field and match.source_column in dataframe.columns
    }
    standardized = dataframe.rename(columns=rename_map).copy()
    standardized = apply_unit_conversions(
        standardized,
        {
            match.source_column: match.canonical_field
            for match in matches
            if match.canonical_field
        },
        unit_overrides=unit_overrides,
    )
    for field in CANONICAL_FIELDS:
        if field not in standardized.columns:
            standardized[field] = pd.NA
    ordered_fields = [field for field in CANONICAL_FIELDS if field in standardized.columns]
    other_fields = [column for column in standardized.columns if column not in CANONICAL_FIELDS]
    return standardized[ordered_fields + other_fields]
