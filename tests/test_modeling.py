import pandas as pd
import pytest

from esp_predictive_health.core.modeling import (
    fit_classifier,
    predict_probabilities,
    validate_grouped,
)


def training_data():
    return pd.DataFrame(
        {
            "case_id": ["A1", "A2", "B1", "B2", "C1", "C2", "D1", "D2"],
            "well_name": ["W1", "W1", "W2", "W2", "W3", "W3", "W4", "W4"],
            "root_cause": ["GAS", "GAS", "GAS", "GAS", "SOLIDS", "SOLIDS", "SOLIDS", "SOLIDS"],
            "sample_weight": [1.0, 1.0, 1.0, 0.7, 1.0, 1.0, 1.0, 0.7],
            "window_days": [1, 3, 1, 3, 1, 3, 1, 3],
            "motor_current_a_mean": [10.0, 11.0, 12.0, 13.0, 30.0, 31.0, 32.0, 33.0],
            "flow_rate_bpd_mean": [1000.0, 990.0, 980.0, 970.0, 500.0, 490.0, 480.0, 470.0],
        }
    )


def test_model_returns_complete_probability_distribution():
    data = training_data()
    artifact = fit_classifier(data)
    probabilities = predict_probabilities(artifact, data)

    assert set(probabilities.columns) == {"GAS", "SOLIDS"}
    assert (probabilities.sum(axis=1).round(8) == 1).all()


def test_grouped_validation_reports_well_holdout_metrics():
    metrics = validate_grouped(training_data(), n_splits=2)

    assert set(metrics["test_wells"]) == {2}
    assert "balanced_accuracy" in metrics.columns
    assert "weighted_f1" in metrics.columns


def test_model_rejects_single_well_training_data():
    data = training_data().assign(well_name="ONE")

    with pytest.raises(ValueError, match="two wells"):
        fit_classifier(data)
