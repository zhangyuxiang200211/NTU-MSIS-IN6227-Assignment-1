"""Decision tree and random forest baselines for the supplied dataset.

Run from the project root:
    python -B -m src.baseline
    python -B src/baseline.py --folds 3 --n-jobs -1
    python -B -m src.baseline --evaluate-test

Models: DecisionTreeClassifier and RandomForestClassifier (100 trees).
Defaults: five shuffled stratified folds, seed 42, no hyperparameter tuning.
Both tree models use imputation and one-hot encoding without numeric scaling.
All models use identical folds; preprocessing is fitted inside each fold.
Mean validation average precision (AP) selects the winner. AP is not the
trapezoidal area under the precision-recall curve. Class weights are left at
their defaults; threshold metrics use each classifier's default prediction.
The test set is read only with --evaluate-test, after model selection.
No reports, models, or predictions are written to disk by this module.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score, make_scorer,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

if __package__:
    from .paths import TRAIN_PATH, TEST_PATH, resolve_path
    from .preprocessing import (
        build_model_pipeline, load_dataset, split_features_target,
    )
else:
    from paths import TRAIN_PATH, TEST_PATH, resolve_path
    from preprocessing import (
        build_model_pipeline, load_dataset, split_features_target,
    )


SCORING = {
    "average_precision": "average_precision",
    "roc_auc": "roc_auc",
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
}


def build_baselines(random_state=42):
    """Return fresh, unfitted pipelines with fixed baseline parameters."""
    return {
        "decision_tree": build_model_pipeline(
            DecisionTreeClassifier(random_state=random_state),
            scale_numeric=False,
        ),
        "random_forest": build_model_pipeline(
            # Parallelize CV only, avoiding nested worker pools.
            RandomForestClassifier(
                n_estimators=100, random_state=random_state, n_jobs=1,
            ),
            scale_numeric=False,
        ),
    }


def compare_baselines(X, y, folds=5, random_state=42, n_jobs=1):
    """Return unfitted candidates, per-fold scores and aggregate CV scores.

    y must contain both mapped labels (0=no, 1=yes), without missing values.
    The reported standard deviation is across folds, not a confidence interval.
    """
    if folds < 2:
        raise ValueError("folds must be at least 2")
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0")
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows")
    if y.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError("Training requires both labels 0 and 1, without missing labels")
    y = y.astype(int)
    if int(y.value_counts().min()) < folds:
        raise ValueError("Each class must have at least as many rows as CV folds")

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    splits = list(cv.split(X, y))
    models = build_baselines(random_state)
    rows = []
    for name, pipeline in models.items():
        print(f"Evaluating {name} ({folds} folds)...", flush=True)
        result = cross_validate(
            pipeline, X, y, cv=splits, scoring=SCORING,
            n_jobs=n_jobs, error_score="raise",
        )
        for fold in range(folds):
            rows.append({
                "model": name,
                "fold": fold + 1,
                **{metric: float(result[f"test_{metric}"][fold])
                   for metric in SCORING},
                "fit_time_seconds": float(result["fit_time"][fold]),
            })
    fold_scores = pd.DataFrame(rows)
    summary = fold_scores.groupby("model", sort=False)[list(SCORING)].agg(
        ["mean", "std"]
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.sort_values(
        "average_precision_mean", ascending=False, kind="stable",
    )
    return models, fold_scores, summary


def evaluate_test(pipeline, X_train, y_train, test_path):
    """Refit a CV-selected model; predict all test rows and score known labels.

    Returns the fitted pipeline, row-aligned predictions and metrics. Missing
    test targets are never imputed; undefined ranking metrics are None.
    """
    test = load_dataset(test_path)
    X_test, y_test = split_features_target(test, drop_missing_labels=False)
    if X_test.empty:
        raise ValueError("The test set contains no rows")
    fitted = clone(pipeline).fit(X_train, y_train.astype(int))
    positive_column = int(np.flatnonzero(fitted.classes_ == 1)[0])
    predictions = pd.DataFrame({
        "true_label": y_test,
        "predicted_label": fitted.predict(X_test),
        "probability_yes": fitted.predict_proba(X_test)[:, positive_column],
    }, index=X_test.index)
    predictions.index.name = "row_index"
    known = y_test.notna()
    metrics = {
        "test_rows": len(test), "scored_rows": int(known.sum()),
        "missing_label_rows": int((~known).sum()),
    }
    if known.any():
        truth = y_test.loc[known].astype(int)
        predicted = predictions.loc[known, "predicted_label"]
        probability = predictions.loc[known, "probability_yes"]
        both_classes = truth.nunique() == 2
        metrics.update({
            "accuracy": float(accuracy_score(truth, predicted)),
            "balanced_accuracy": (
                float(balanced_accuracy_score(truth, predicted)) if both_classes else None
            ),
            "precision": float(precision_score(truth, predicted, zero_division=0)),
            "recall": float(recall_score(truth, predicted, zero_division=0)),
            "f1": float(f1_score(truth, predicted, zero_division=0)),
            "average_precision": (
                float(average_precision_score(truth, probability)) if both_classes else None
            ),
            "roc_auc": float(roc_auc_score(truth, probability)) if both_classes else None,
            "confusion_matrix": confusion_matrix(truth, predicted, labels=[0, 1]).tolist(),
            "classification_report": classification_report(
                truth, predicted, labels=[0, 1], target_names=["no", "yes"],
                zero_division=0, output_dict=True,
            ),
        })
    return fitted, predictions, metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--test", type=Path, default=TEST_PATH)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--evaluate-test", action="store_true",
                        help="Refit the CV winner and evaluate the reserved test set")
    args = parser.parse_args(argv)
    if args.folds < 2 or args.n_jobs == 0:
        parser.error("--folds must be at least 2 and --n-jobs must not be 0")
    train_path = resolve_path(args.train).resolve()
    test_path = resolve_path(args.test).resolve()
    if args.evaluate_test and train_path == test_path:
        parser.error("Training and test paths must differ")
    train = load_dataset(train_path)
    X, y = split_features_target(train)
    print(f"Training rows: {len(train):,}; labeled: {len(y):,}; "
          f"excluded missing targets: {len(train) - len(y):,}")
    print(f"Label counts (0=no, 1=yes): {y.value_counts().sort_index().to_dict()}")
    models, _, summary = compare_baselines(
        X, y, folds=args.folds, random_state=args.seed, n_jobs=args.n_jobs,
    )
    print("\nCross-validation scores (mean and sample standard deviation):")
    print(summary.to_string(float_format=lambda value: f"{value:.4f}"))
    winner = summary.index[0]
    print(f"\nBest baseline by mean validation AP: {winner}")
    if args.evaluate_test:
        import json

        _, _, metrics = evaluate_test(models[winner], X, y, test_path)
        print("\nFinal test evaluation (confusion matrix order: no, yes):")
        print(json.dumps(metrics, indent=2, allow_nan=False))
    else:
        print("Test set was not read. Use --evaluate-test only for final evaluation.")


if __name__ == "__main__":
    main()
