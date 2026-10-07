"""Shared, reproducible train-only hyperparameter search and artifact export."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate

if __package__:
    from .evaluate import METRIC_NAMES, SELECTION_METRIC, evaluate_model, score_estimator
    from .paths import TRAIN_PATH, TEST_PATH, OUTPUT_DIR, resolve_path
    from .preprocessing import load_dataset, split_features_target
else:
    from evaluate import METRIC_NAMES, SELECTION_METRIC, evaluate_model, score_estimator
    from paths import TRAIN_PATH, TEST_PATH, OUTPUT_DIR, resolve_path
    from preprocessing import load_dataset, split_features_target


def make_splits(X, y, folds=5, random_state=42):
    """Both models receive identical positional folds for identical data/seed."""
    if folds < 2:
        raise ValueError("folds must be at least 2")
    if len(X) != len(y) or not X.index.equals(y.index):
        raise ValueError("X and y must be aligned")
    if y.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError("Training requires both labels 0 and 1 without missing targets")
    if int(y.value_counts().min()) < folds:
        raise ValueError("Each class must have at least as many rows as CV folds")
    return list(StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=random_state,
    ).split(X, y.astype(int)))


def cross_validate_model(pipeline, X, y, folds=5, random_state=42, n_jobs=1):
    """Evaluate fixed parameters; retained for programmatic comparisons."""
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0")
    splits = make_splits(X, y, folds, random_state)
    result = cross_validate(
        pipeline, X, y.astype(int), cv=splits, scoring=score_estimator,
        n_jobs=n_jobs, error_score="raise",
    )
    scores = pd.DataFrame({name: result[f"test_{name}"] for name in METRIC_NAMES},
                          index=pd.RangeIndex(1, folds + 1, name="fold"))
    summary = scores.agg(["mean", "std"]).T
    scores["fit_time_seconds"] = result["fit_time"]
    return scores, summary


def tune_model(pipeline, parameter_space, X, y, folds=5, random_state=42,
               n_jobs=1, n_iter=24, verbose=1):
    """Tune on training folds only and refit the highest-AP candidate on all X.

    CV scores are selection scores, not an unbiased estimate after tuning.
    A held-out test set is used only by the explicitly requested final step.
    """
    if n_jobs == 0 or n_iter < 1:
        raise ValueError("n_jobs must not be 0 and n_iter must be positive")
    splits = make_splits(X, y, folds, random_state)
    search = RandomizedSearchCV(
        pipeline, param_distributions=parameter_space, n_iter=n_iter,
        scoring=score_estimator, refit=SELECTION_METRIC, cv=splits,
        random_state=random_state, n_jobs=n_jobs, pre_dispatch="n_jobs",
        return_train_score=True, error_score="raise", verbose=verbose,
    )
    search.fit(X, y.astype(int))
    return search, splits


def write_json(path, payload):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_cli(model_name, builder, parameter_space, argv=None):
    parser = argparse.ArgumentParser(
        description=f"Formal {model_name} experiment: train-only randomized search by AP.",
    )
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--test", type=Path, default=TEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / model_name)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--n-iter", type=int, default=24)
    parser.add_argument("--evaluate-test", action="store_true",
                        help="Evaluate the refitted model after train-only selection")
    args = parser.parse_args(argv)
    if args.folds < 2 or args.n_jobs == 0 or args.n_iter < 1:
        parser.error("Require --folds >= 2, --n-jobs != 0 and --n-iter >= 1")
    train_path = resolve_path(args.train).resolve()
    test_path = resolve_path(args.test).resolve()
    if args.evaluate_test and (train_path == test_path or train_path.samefile(test_path)):
        parser.error("Training and test paths must differ")
    train = load_dataset(train_path)
    X, y = split_features_target(train)
    # A new directory per run prevents mixing old test outputs with a new search.
    output_base = resolve_path(args.output_dir).resolve()
    output = output_base / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    print(f"Training rows: {len(train)}; labeled: {len(y)}; "
          f"excluded missing targets: {len(train) - len(y)}", flush=True)
    print(f"Output: {output}", flush=True)
    search, splits = tune_model(
        builder(random_state=args.seed), parameter_space, X, y,
        folds=args.folds, random_state=args.seed, n_jobs=args.n_jobs, n_iter=args.n_iter,
    )
    results = pd.DataFrame(search.cv_results_)
    results.to_csv(output / "cv_results.csv", index=False)
    best = search.best_index_
    fold_scores = pd.DataFrame({
        name: [float(search.cv_results_[f"split{fold}_test_{name}"][best])
               for fold in range(args.folds)] for name in METRIC_NAMES
    }, index=pd.RangeIndex(1, args.folds + 1, name="fold"))
    fold_scores.to_csv(output / "best_cv_folds.csv")
    summary = fold_scores.agg(["mean", "std"]).T
    summary.index.name = "metric"
    summary.to_csv(output / "best_cv_summary.csv")
    fold_assignment = pd.Series(index=X.index, dtype="Int64", name="validation_fold")
    for fold, (_, validation) in enumerate(splits, start=1):
        fold_assignment.iloc[validation] = fold
    fold_assignment.index.name = "row_index"
    fold_assignment.to_csv(output / "cv_fold_assignments.csv")
    joblib.dump(search.best_estimator_, output / "best_pipeline.joblib")
    write_json(output / "best_params.json", search.best_params_)
    metadata = {
        "model": model_name, "seed": args.seed, "folds": args.folds,
        "n_jobs": args.n_jobs, "requested_candidates": args.n_iter,
        "evaluated_candidates": len(results), "selection_metric": SELECTION_METRIC,
        "parameter_space": parameter_space, "best_params": search.best_params_,
        "best_cv_average_precision": float(search.best_score_),
        "cv_note": "Selection CV; not an unbiased post-tuning performance estimate. "
                   "Summary std uses ddof=1; sklearn cv_results std uses ddof=0.",
        "prediction_rule": "estimator.predict; probability ties select class 0",
        "positive_class": "yes=1", "confusion_matrix_order": ["no", "yes"],
        "train_path": str(train_path), "train_sha256": file_digest(train_path),
        "train_rows": len(train), "labeled_train_rows": len(y),
        "missing_train_targets_excluded": len(train) - len(y),
        "test_evaluated": False,
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__,
                     "joblib": joblib.__version__},
    }
    write_json(output / "experiment.json", metadata)
    print("Best parameters:", search.best_params_)
    print("Selection CV (mean and sample std):")
    print(summary.to_string(float_format=lambda value: f"{value:.4f}"))
    if args.evaluate_test:
        test = load_dataset(test_path)
        X_test, y_test = split_features_target(test, drop_missing_labels=False)
        metrics, predictions = evaluate_model(search.best_estimator_, X_test, y_test)
        predictions.to_csv(output / "test_predictions.csv")
        write_json(output / "test_metrics.json", metrics)
        metadata.update({"test_evaluated": True, "test_path": str(test_path),
                         "test_sha256": file_digest(test_path)})
        write_json(output / "experiment.json", metadata)
        print("Final test evaluation:")
        print(json.dumps(metrics, indent=2, allow_nan=False))
    else:
        print("Test set was not read. Add --evaluate-test for final evaluation.")
    return search
