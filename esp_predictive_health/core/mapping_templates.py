"""Small JSON-backed mapping template store for recurring data sources."""

from __future__ import annotations

import json
from pathlib import Path

from esp_predictive_health.database.db import PROJECT_ROOT

TEMPLATE_DIRECTORY = PROJECT_ROOT / "data" / "mapping_templates"


def save_template(name: str, mapping: dict[str, str]) -> Path:
    """Save a source-column to canonical-field mapping under a safe filename."""
    clean_name = "_".join(name.strip().lower().split())
    if not clean_name:
        raise ValueError("Template name is required.")
    TEMPLATE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = TEMPLATE_DIRECTORY / f"{clean_name}.json"
    path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    return path


def list_templates() -> list[str]:
    """Return available mapping template names."""
    TEMPLATE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    return sorted(path.stem for path in TEMPLATE_DIRECTORY.glob("*.json"))


def load_template(name: str) -> dict[str, str]:
    """Load a saved mapping template."""
    path = TEMPLATE_DIRECTORY / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Mapping template not found: {name}")
    return json.loads(path.read_text(encoding="utf-8"))
