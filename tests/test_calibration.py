import pandas as pd
import pytest

from esp_predictive_health.core.temperature_calibration import (
    ConfidenceConfig,
    apply_temperature,
    assess_confidence,
    calibrate_temperature,
)


def test_temperature_calibration_preserves_probability_distribution():
    probabilities = pd.DataFrame(
        {
            "GAS": [0.90, 0.75, 0.20, 0.60],
            "SOLIDS": [0.10, 0.25, 0.80, 0.40],
        }
    )
    labels = pd.Series(["GAS", "GAS", "SOLIDS", "SOLIDS"])

    temperature = calibrate_temperature(probabilities, labels)
    calibrated = apply_temperature(probabilities, temperature)

    assert temperature > 0
    assert calibrated.sum(axis=1).round(8).eq(1).all()


def test_confidence_flags_low_margin_as_unknown():
    probabilities = pd.DataFrame({"GAS": [0.51], "SOLIDS": [0.49]})

    assessed = assess_confidence(
        probabilities,
        config=ConfidenceConfig(minimum_probability=0.55, minimum_margin=0.10),
    )

    assert assessed.loc[0, "most_likely_diagnosis"] == "UNKNOWN"
    assert bool(assessed.loc[0, "engineering_review_required"])


def test_calibration_requires_holdout_rows():
    with pytest.raises(ValueError, match="four holdout"):
        calibrate_temperature(pd.DataFrame({"GAS": [1.0]}), pd.Series(["GAS"]))
