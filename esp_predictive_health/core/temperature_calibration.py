"""Leakage-aware probability calibration and engineering-review flags."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class ConfidenceConfig:
    """Thresholds for deciding whether a prediction needs engineering review."""

    minimum_probability: float = 0.55
    maximum_normalized_entropy: float = 0.85
    minimum_margin: float = 0.10


def _normalize(probabilities: np.ndarray) -> np.ndarray:
    totals = probabilities.sum(axis=1, keepdims=True)
    return probabilities / np.where(totals == 0, 1, totals)


def calibrate_temperature(probabilities: pd.DataFrame, labels: pd.Series) -> float:
    """Fit a scalar temperature on holdout probabilities only."""
    if len(probabilities) != len(labels) or len(probabilities) < 4:
        raise ValueError("At least four holdout probability rows are required for calibration.")
    classes = list(probabilities.columns)
    label_values = labels.astype(str).to_numpy()
    if not set(label_values).issubset(classes):
        raise ValueError("Calibration labels must be present in the probability columns.")
    values = np.clip(_normalize(probabilities.to_numpy(dtype=float)), 1e-8, 1.0)
    target = np.array([classes.index(label) for label in label_values])

    def loss(log_temperature: float) -> float:
        temperature = np.exp(log_temperature)
        logits = np.log(values) / temperature
        logits -= logits.max(axis=1, keepdims=True)
        scaled = np.exp(logits)
        scaled /= scaled.sum(axis=1, keepdims=True)
        return float(-np.log(scaled[np.arange(len(target)), target]).mean())

    result = minimize_scalar(loss, bounds=(-2.0, 2.0), method="bounded")
    if not result.success:
        raise ValueError("Probability calibration did not converge.")
    return float(np.exp(result.x))


def apply_temperature(probabilities: pd.DataFrame, temperature: float) -> pd.DataFrame:
    """Apply a fitted scalar temperature to a probability distribution."""
    if temperature <= 0:
        raise ValueError("Calibration temperature must be positive.")
    values = np.clip(_normalize(probabilities.to_numpy(dtype=float)), 1e-8, 1.0)
    logits = np.log(values) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    scaled = np.exp(logits)
    scaled /= scaled.sum(axis=1, keepdims=True)
    return pd.DataFrame(scaled, columns=probabilities.columns, index=probabilities.index)


def assess_confidence(
    probabilities: pd.DataFrame,
    *,
    config: ConfidenceConfig | None = None,
    required_missing: pd.Series | None = None,
) -> pd.DataFrame:
    """Return top diagnosis, margin, entropy, and engineering-review decision."""
    config = config or ConfidenceConfig()
    values = _normalize(probabilities.to_numpy(dtype=float))
    order = np.argsort(values, axis=1)
    top_index = order[:, -1]
    second_index = order[:, -2] if values.shape[1] > 1 else top_index
    top_probability = values[np.arange(len(values)), top_index]
    second_probability = values[np.arange(len(values)), second_index]
    margin = top_probability - second_probability
    normalized_entropy = -(values * np.log(np.clip(values, 1e-8, 1.0))).sum(axis=1) / np.log(values.shape[1])
    missing = required_missing.fillna(0).astype(bool).to_numpy() if required_missing is not None else np.zeros(len(values), dtype=bool)
    review = (
        (top_probability < config.minimum_probability)
        | (normalized_entropy > config.maximum_normalized_entropy)
        | (margin < config.minimum_margin)
        | missing
    )
    diagnosis = np.array(probabilities.columns, dtype=object)[top_index]
    diagnosis[review] = "UNKNOWN"
    return pd.DataFrame(
        {
            "most_likely_diagnosis": diagnosis,
            "top_probability": top_probability,
            "second_probability": second_probability,
            "probability_margin": margin,
            "normalized_entropy": normalized_entropy,
            "engineering_review_required": review,
        },
        index=probabilities.index,
    )
