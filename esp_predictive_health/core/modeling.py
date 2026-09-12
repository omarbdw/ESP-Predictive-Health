"""Interpretable, grouped-validation models for ESP case diagnosis."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

METADATA_COLUMNS = {
    "case_id",
    "well_name",
    "root_cause",
    "confirmation_level",
    "sample_weight",
    "window_days",
    "window_start",
    "window_end",
    "row_count",
    "missing_rate",
}


@dataclass(frozen=True)
class ModelArtifact:
    """Fitted classifier and the exact feature contract used to train it."""

    pipeline: Pipeline
    feature_columns: tuple[str, ...]
    classes: tuple[str, ...]


def _feature_columns(dataframe: pd.DataFrame) -> list[str]:
    return [
        column
        for column in dataframe.columns
        if column not in METADATA_COLUMNS and pd.api.types.is_numeric_dtype(dataframe[column])
    ]


def _validate_training_data(dataframe: pd.DataFrame) -> tuple[list[str], pd.Series, pd.Series]:
    required = {"root_cause", "well_name", "sample_weight"}
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(f"Training data is missing required columns: {', '.join(sorted(missing))}")
    features = _feature_columns(dataframe)
    if not features:
        raise ValueError("No numeric training features are available.")
    labels = dataframe["root_cause"].astype(str)
    groups = dataframe["well_name"].astype(str)
    if labels.nunique() < 2:
        raise ValueError("At least two root-cause classes are required for training.")
    if groups.nunique() < 2:
        raise ValueError("At least two wells are required for grouped validation.")
    return features, labels, groups


def _new_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )


def validate_grouped(
    dataframe: pd.DataFrame,
    *,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Evaluate grouped folds by well, never splitting rows from one well."""
    features, labels, groups = _validate_training_data(dataframe)
    split_count = min(max(2, n_splits), groups.nunique())
    rows: list[dict[str, object]] = []
    splitter = GroupKFold(n_splits=split_count)
    values = dataframe[features]
    weights = pd.to_numeric(dataframe["sample_weight"], errors="coerce").fillna(1.0)
    for fold, (train_index, test_index) in enumerate(splitter.split(values, labels, groups), start=1):
        train_labels = labels.iloc[train_index]
        if train_labels.nunique() < 2:
            continue
        pipeline = _new_pipeline()
        pipeline.fit(values.iloc[train_index], train_labels, classifier__sample_weight=weights.iloc[train_index])
        predicted = pipeline.predict(values.iloc[test_index])
        rows.append(
            {
                "fold": fold,
                "train_wells": groups.iloc[train_index].nunique(),
                "test_wells": groups.iloc[test_index].nunique(),
                "test_rows": len(test_index),
                "balanced_accuracy": balanced_accuracy_score(labels.iloc[test_index], predicted),
                "weighted_f1": f1_score(labels.iloc[test_index], predicted, average="weighted", zero_division=0),
            }
        )
    if not rows:
        raise ValueError("No grouped validation fold contained at least two training classes.")
    return pd.DataFrame(rows)


def grouped_oof_probabilities(
    dataframe: pd.DataFrame,
    *,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return well-held-out probabilities suitable for calibration."""
    features, labels, groups = _validate_training_data(dataframe)
    split_count = min(max(2, n_splits), groups.nunique())
    classes = tuple(sorted(labels.unique()))
    probabilities = pd.DataFrame(float("nan"), index=dataframe.index, columns=classes)
    splitter = GroupKFold(n_splits=split_count)
    for train_index, test_index in splitter.split(dataframe[features], labels, groups):
        train_labels = labels.iloc[train_index]
        if train_labels.nunique() < 2:
            continue
        pipeline = _new_pipeline()
        pipeline.fit(
            dataframe.iloc[train_index][features],
            train_labels,
            classifier__sample_weight=pd.to_numeric(
                dataframe.iloc[train_index]["sample_weight"], errors="coerce"
            ).fillna(1.0),
        )
        fold_probabilities = pd.DataFrame(
            pipeline.predict_proba(dataframe.iloc[test_index][features]),
            index=dataframe.index[test_index],
            columns=[str(value) for value in pipeline.named_steps["classifier"].classes_],
        )
        probabilities.loc[fold_probabilities.index, fold_probabilities.columns] = fold_probabilities
    valid = probabilities.notna().all(axis=1)
    if valid.sum() < 4:
        raise ValueError("At least four valid grouped holdout rows are required for calibration.")
    return probabilities.loc[valid], labels.loc[valid]


def fit_classifier(dataframe: pd.DataFrame) -> ModelArtifact:
    """Fit the full weighted classifier after validating its grouped-data contract."""
    features, labels, _ = _validate_training_data(dataframe)
    pipeline = _new_pipeline()
    weights = pd.to_numeric(dataframe["sample_weight"], errors="coerce").fillna(1.0)
    pipeline.fit(dataframe[features], labels, classifier__sample_weight=weights)
    classes = tuple(str(value) for value in pipeline.named_steps["classifier"].classes_)
    return ModelArtifact(pipeline=pipeline, feature_columns=tuple(features), classes=classes)


def predict_probabilities(artifact: ModelArtifact, dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return one probability column for every supported class."""
    missing = set(artifact.feature_columns) - set(dataframe.columns)
    if missing:
        raise ValueError(f"Prediction data is missing features: {', '.join(sorted(missing))}")
    probabilities = artifact.pipeline.predict_proba(dataframe[list(artifact.feature_columns)])
    return pd.DataFrame(probabilities, columns=list(artifact.classes), index=dataframe.index)
