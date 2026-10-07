"""EDA-driven feature ablation; run python -B -m src.feature_engineering.

No test data is read. All fitted preprocessing stays inside each CV fold.
Existing formal model entry points retain their original baseline.
"""
import argparse
from pathlib import Path
import sys

# Direct execution puts src/, rather than the project root, on sys.path.
# Resolve from this file so IDE launchers and other working directories work.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datetime import datetime, timezone
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted

from src.preprocessing import (FEATURE_COLUMNS, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS,
                            load_dataset, split_features_target, build_model_pipeline)
from src.experiment import make_splits, file_digest, write_json
from src.evaluate import score_estimator
from src.paths import TRAIN_PATH, OUTPUT_DIR, resolve_path

VARIANTS = ("baseline", "reduced", "engineered", "reduced_engineered")
DERIVED_COLUMNS = ["missing_feature_count", "performance_stability_both_zero",
                   "load_duration_both_zero"]
ZERO_PAIRS = [("performance_score", "stability_index"),
              ("load_ratio", "activity_duration")]


class AuditFeatures(TransformerMixin, BaseEstimator):
    """Deterministic transformations; no label-derived statistics or thresholds.

    Joint-zero flags remain missing when either source value is unobserved.
    In the reduced_engineered variant stability_index is still required as a
    source for its joint-zero flag; only its continuous representation is dropped.
    """
    def __init__(self, variant="engineered"):
        self.variant = variant

    def fit(self, X, y=None):
        if self.variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {self.variant}")
        self._validate(X)
        self.feature_names_in_ = np.asarray(FEATURE_COLUMNS, dtype=object)
        self.n_features_in_ = len(FEATURE_COLUMNS)
        self.output_columns_ = list(FEATURE_COLUMNS)
        if "reduced" in self.variant:
            self.output_columns_.remove("stability_index")
        if "engineered" in self.variant:
            self.output_columns_ += DERIVED_COLUMNS
        return self

    @staticmethod
    def _validate(X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("AuditFeatures requires a named pandas DataFrame")
        missing = set(FEATURE_COLUMNS) - set(X.columns)
        if missing:
            raise ValueError(f"Missing input features: {sorted(missing)}")

    def transform(self, X):
        check_is_fitted(self, "output_columns_")
        self._validate(X)
        result = X.loc[:, FEATURE_COLUMNS].copy()
        if "engineered" in self.variant:
            result[DERIVED_COLUMNS[0]] = result.isna().sum(axis=1).astype(float)
            for name, (left, right) in zip(DERIVED_COLUMNS[1:], ZERO_PAIRS):
                known = result[[left, right]].notna().all(axis=1)
                result[name] = (result[left].eq(0) & result[right].eq(0)).astype(float)
                result.loc[~known, name] = np.nan
        return result.loc[:, self.output_columns_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "output_columns_")
        return np.asarray(self.output_columns_, dtype=object)


def build_feature_pipeline(classifier, variant="engineered"):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    if variant == "baseline":
        return build_model_pipeline(classifier)
    numeric = [c for c in NUMERIC_COLUMNS
               if not ("reduced" in variant and c == "stability_index")]
    if "engineered" in variant:
        numeric += DERIVED_COLUMNS
    preprocessor = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__",
                                      keep_empty_features=True)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]), CATEGORICAL_COLUMNS),
    ], remainder="drop")
    return Pipeline([("features", AuditFeatures(variant)),
                     ("preprocessor", preprocessor), ("model", classifier)])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default=TRAIN_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR / "feature_engineering")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    train_path = resolve_path(args.train)
    frame = load_dataset(train_path)
    X, y = split_features_target(frame)
    splits = make_splits(X, y, args.folds, args.seed)
    output = resolve_path(args.output_dir) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    # Fixed settings isolate feature changes. These are not tuned optima.
    models = {
        "decision_tree": DecisionTreeClassifier(max_depth=12, min_samples_leaf=10,
                                                random_state=args.seed),
        "random_forest": RandomForestClassifier(n_estimators=100, min_samples_leaf=3,
                                                 random_state=args.seed, n_jobs=1),
    }
    records = []
    assignments = pd.Series(index=X.index, dtype="Int64", name="validation_fold")
    for fold, (_, valid) in enumerate(splits, 1):
        assignments.iloc[valid] = fold
    assignments.index.name = "row_index"
    assignments.to_csv(output / "cv_fold_assignments.csv")
    for model_name, model in models.items():
        for variant in VARIANTS:
            for fold, (train, valid) in enumerate(splits, 1):
                fitted = build_feature_pipeline(clone(model), variant)
                fitted.fit(X.iloc[train], y.iloc[train].astype(int))
                scores = score_estimator(fitted, X.iloc[valid], y.iloc[valid].astype(int))
                records.append(dict(model=model_name, variant=variant, fold=fold, **scores))
            print(f"Completed {model_name}: {variant}", flush=True)
    results = pd.DataFrame(records)
    results.to_csv(output / "cv_folds.csv", index=False)
    metrics = [c for c in results if c not in ["model", "variant", "fold"]]
    summary = results.groupby(["model", "variant"], sort=False)[metrics].agg(["mean", "std"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    summary.reset_index().to_csv(output / "cv_summary.csv", index=False)
    means = summary["average_precision_mean"].unstack("model").reindex(VARIANTS)
    # A shared feature recipe keeps the two classifiers comparable. Exact ties
    # prefer the first declared variant (baseline first); no test score is used.
    shared_scores = means.mean(axis=1)
    selected = str(shared_scores.idxmax())
    shared_scores.rename("mean_ap_across_models").to_csv(output / "selection_scores.csv")
    manifest = {}
    for model_name, model in models.items():
        fitted = build_feature_pipeline(clone(model), selected).fit(X, y.astype(int))
        joblib.dump(fitted, output / f"{model_name}_pipeline.joblib")
        names = fitted.named_steps["preprocessor"].get_feature_names_out().tolist()
        manifest[model_name] = names
    write_json(output / "feature_names.json", manifest)
    write_json(output / "selection.json", {
        "selected_variant": selected, "selection_metric": "mean AP across the two fixed models",
        "variants": list(VARIANTS), "seed": args.seed, "folds": args.folds,
        "train_sha256": file_digest(train_path), "labeled_rows": len(y),
        "excluded_missing_labels": len(frame) - len(y), "test_data_used": False,
        "model_parameters": {name: model.get_params() for name, model in models.items()},
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
        "caveat": "Exploratory feature selection on already explored training data. "
                  "CV scores are selection scores, not unbiased generalization estimates. "
                  "Feature gains can depend on model hyperparameters; tune after selection.",
    })
    print(summary.to_string())
    print(f"Selected shared recipe: {selected}; saved to {output}")
    return output


if __name__ == "__main__":
    # Use an importable class path in joblib artifacts, not __main__.AuditFeatures.
    from src.feature_engineering import main as run_main
    run_main()
