"""Audit Variant 1 data without fitting preprocessing or changing CSV files.

Run from repository root:
    python -m src.inspect_data
    python -m src.inspect_data --train PATH --output outputs/data_audit.json
    python -m src.inspect_data --include-test-audit

Optional config.json keys: train_path, test_path, output_dir.
Default CSV locations: data/train.csv and data/test.csv.
CLI relative paths are resolved relative to the project root.
Relative paths explicitly set in config are resolved relative to that file.
Omitted config keys use the project data/ and outputs/ directories.
Test audit is explicitly opt-in and only reports integrity, not label balance
or feature distributions. Modeling decisions must use training data only.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .paths import PROJECT_ROOT, TRAIN_PATH, TEST_PATH, OUTPUT_DIR, resolve_path
    from .preprocessing import (
        CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS,
        TARGET_COLUMN, load_dataset,
    )
else:
    from paths import PROJECT_ROOT, TRAIN_PATH, TEST_PATH, OUTPUT_DIR, resolve_path
    from preprocessing import (
        CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS,
        TARGET_COLUMN, load_dataset,
    )


def integrity_summary(frame):
    """Counts refer to all raw records after deterministic whitespace cleanup."""
    missing_target = frame[TARGET_COLUMN].isna()
    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "feature_count": len(FEATURE_COLUMNS),
        "numeric_feature_count": len(NUMERIC_COLUMNS),
        "categorical_feature_count": len(CATEGORICAL_COLUMNS),
        "missing_by_column": frame.isna().sum().to_dict(),
        "missing_feature_cells": int(frame[FEATURE_COLUMNS].isna().sum().sum()),
        "rows_with_missing_features": int(frame[FEATURE_COLUMNS].isna().any(axis=1).sum()),
        "missing_label_rows": int(missing_target.sum()),
        "labeled_rows": int((~missing_target).sum()),
        "missing_label_row_indices": frame.index[missing_target].tolist(),
        "duplicate_rows_beyond_first": int(frame.duplicated().sum()),
        "duplicate_feature_rows_beyond_first": int(frame[FEATURE_COLUMNS].duplicated().sum()),
    }


def training_summary(frame):
    result = integrity_summary(frame)
    labeled = frame.loc[frame[TARGET_COLUMN].notna()]
    labels = labeled[TARGET_COLUMN].value_counts()
    result["label_counts"] = labels.to_dict()
    result["label_proportions"] = (labels / len(labeled)).to_dict()
    # EDA here uses all training records, including records without a label.
    result["numeric_summary_all_train_rows"] = (
        frame[NUMERIC_COLUMNS].describe().to_dict()
    )
    result["numeric_skewness"] = frame[NUMERIC_COLUMNS].skew().to_dict()
    result["numeric_zero_counts"] = frame[NUMERIC_COLUMNS].eq(0).sum().to_dict()
    result["categorical_counts"] = {
        col: frame[col].dropna().value_counts().to_dict()
        for col in CATEGORICAL_COLUMNS
    }
    result["iqr_diagnostics"] = {}
    for col in NUMERIC_COLUMNS:
        values = frame[col].dropna()
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        result["iqr_diagnostics"][col] = {
            "lower": lower, "upper": upper,
            "flagged_rows": int(((values < lower) | (values > upper)).sum()),
        }
    corr = frame[NUMERIC_COLUMNS].corr(method="pearson")
    result["numeric_pearson_correlation"] = corr.to_dict()
    result["high_correlation_pairs"] = [
        {"feature_1": a, "feature_2": b, "correlation": float(corr.loc[a, b])}
        for i, a in enumerate(NUMERIC_COLUMNS)
        for b in NUMERIC_COLUMNS[i + 1:]
        if abs(corr.loc[a, b]) >= 0.8
    ]
    return result


def json_safe(value):
    """Write strict JSON: NumPy scalars become Python values, NaN becomes null."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-test-audit", action="store_true")
    args = parser.parse_args()
    config = {}
    base = PROJECT_ROOT

    if args.config:
        config_path = resolve_path(args.config, PROJECT_ROOT)
        with config_path.open(encoding="utf-8") as handle:
            config = json.load(handle)
        # Relative paths inside config.json are resolved relative to config.json.
        base = config_path.parent

    if args.train:
        train_path = resolve_path(args.train, PROJECT_ROOT)
    else:
        train_path = resolve_path(config.get("train_path", TRAIN_PATH), base)

    if args.test:
        test_path = resolve_path(args.test, PROJECT_ROOT)
    else:
        test_path = resolve_path(config.get("test_path", TEST_PATH), base)

    if args.output:
        output = resolve_path(args.output, PROJECT_ROOT)
    else:
        output = resolve_path(config.get("output_dir", OUTPUT_DIR), base) / "data_audit.json"
    train = load_dataset(train_path)
    audit = {
        "notes": [
            "Row indices are zero-based CSV record positions.",
            "Counts follow whitespace trimming; blanks alone are missing.",
            "Unknown and Other remain ordinary categories.",
            "IQR flags and correlations are diagnostic; no records/features are removed.",
            "No imputers, encoders or scalers are fitted by this script.",
        ],
        "train_path": str(train_path.resolve()),
        "train": training_summary(train),
    }
    if args.include_test_audit:
        test = load_dataset(test_path)
        audit["test_path"] = str(test_path.resolve())
        audit["test_integrity"] = integrity_summary(test)
        # Exact matching on all 15 features; NaNs match NaNs in pandas merge.
        unique_train = train[FEATURE_COLUMNS].drop_duplicates()
        matches = test[FEATURE_COLUMNS].merge(
            unique_train, how="left", on=FEATURE_COLUMNS,
            indicator=True, validate="many_to_one",
        )
        audit["test_rows_matching_train_features"] = int(matches["_merge"].eq("both").sum())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(audit), handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Training rows: {len(train):,}; labeled: {audit['train']['labeled_rows']:,}")
    print(f"Audit saved to: {output.resolve()}")


if __name__ == "__main__":
    main()
