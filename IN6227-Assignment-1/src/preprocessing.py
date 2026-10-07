"""Variant 1 preprocessing. Requires pandas and scikit-learn >= 1.2.

Fit the returned Pipeline inside cross-validation; do not transform the full
training set before cross-validation. Unknown/Other are valid categories.
"""
if __package__:
    from .paths import resolve_path
else:
    from paths import resolve_path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

NUMERIC_COLUMNS = [
    "aptitude_score", "performance_score", "stability_index", "load_ratio",
    "index_weight", "activity_duration", "composite_rank",
]
CATEGORICAL_COLUMNS = [
    "geological_era", "weather_pattern", "region", "personal_interest",
    "brew_preference", "instrument", "species", "mineral_type",
]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
TARGET_COLUMN = "label"
LABEL_MAPPING = {"no": 0, "yes": 1}


def load_dataset(path):
    """Read known fields; only blank/whitespace cells count as missing.

    Relative paths are resolved from the project root.
    Invalid numeric values, infinities, unexpected labels and schema changes
    raise errors instead of silently altering the data. Row indices remain
    the original zero-based record positions (not CSV physical line numbers).
    """
    frame = pd.read_csv(resolve_path(path), dtype=str, keep_default_na=False)
    expected = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = expected - set(frame.columns)
    extra = set(frame.columns) - expected
    if missing or extra:
        raise ValueError(f"Schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    # Deterministic normalization needs no fitted statistics and is safe before CV.
    for col in frame.columns:
        frame[col] = frame[col].str.strip().replace("", np.nan)
    for col in NUMERIC_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="raise").astype(float)
        if np.isinf(frame[col].to_numpy()).any():
            raise ValueError(f"Infinite numeric values in {col}")
    invalid = set(frame[TARGET_COLUMN].dropna()) - set(LABEL_MAPPING)
    if invalid:
        raise ValueError(f"Unexpected labels: {sorted(invalid)}")
    for col in CATEGORICAL_COLUMNS:
        if frame[col].eq("__MISSING__").any():
            raise ValueError(f"Reserved missing-value token already exists in {col}")
    return frame


def split_features_target(frame, drop_missing_labels=True):
    """Return X and nullable integer y, preserving original row indices.

    Training: use the default and exclude missing labels.
    Evaluation: pass False; predict ALL rows, score only y.notna() rows.
    Never include label in X or impute a missing label.
    """
    invalid = set(frame[TARGET_COLUMN].dropna()) - set(LABEL_MAPPING)
    if invalid:
        raise ValueError(f"Unexpected labels: {sorted(invalid)}")
    selected = frame.loc[frame[TARGET_COLUMN].notna()] if drop_missing_labels else frame
    X = selected.loc[:, FEATURE_COLUMNS].copy()
    y = selected[TARGET_COLUMN].map(LABEL_MAPPING).astype("Int64")
    return X, y


def build_preprocessor(scale_numeric=False):
    """Return a NEW, unfitted ColumnTransformer.

    Decision tree and random forest: scale_numeric=False.
    Empty numerical columns, if any in a fold, remain present (filled with 0).
    Unseen validation/test categories produce all-zero one-hot blocks.
    """
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical = Pipeline([
        ("imputer", SimpleImputer(
            strategy="constant", fill_value="__MISSING__", keep_empty_features=True,
        )),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    return ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), NUMERIC_COLUMNS),
        ("categorical", categorical, CATEGORICAL_COLUMNS),
    ], remainder="drop")


def build_model_pipeline(classifier, scale_numeric=False):
    """Combine preprocessing and a fresh estimator for GridSearchCV.

    Parameter grid names use model__, for example model__min_samples_leaf or model__max_depth.
    Save the fitted Pipeline, not just the classifier.
    """
    return Pipeline([
        ("preprocessor", build_preprocessor(scale_numeric)),
        ("model", classifier),
    ])
