"""Formal random-forest experiment with train-only hyperparameter selection.

    python -B -m src.random_forest
    python -B -m src.random_forest --evaluate-test --n-jobs -1

Uses exactly the same folds, preprocessing and evaluation as decision_tree.py.
"""
from sklearn.ensemble import RandomForestClassifier

if __package__:
    from .experiment import run_cli, tune_model
    from .preprocessing import build_model_pipeline
else:
    from experiment import run_cli, tune_model
    from preprocessing import build_model_pipeline

PARAMETER_SPACE = {
    "model__n_estimators": [100, 200, 400],
    "model__max_depth": [None, 8, 16, 24],
    "model__min_samples_split": [2, 10],
    "model__min_samples_leaf": [1, 3, 10],
    "model__max_features": ["sqrt", "log2", 0.5],
    "model__class_weight": [None, "balanced", "balanced_subsample"],
}


def build_random_forest(random_state=42, n_estimators=100, max_depth=None,
                        min_samples_leaf=1, min_samples_split=2,
                        max_features="sqrt", class_weight=None):
    """Fresh unscaled pipeline. Parallelize search, not individual forests."""
    return build_model_pipeline(
        RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
            max_features=max_features, class_weight=class_weight,
            random_state=random_state, n_jobs=1,
        ), scale_numeric=False,
    )


def run_experiment(X, y, folds=5, random_state=42, n_jobs=1, n_iter=24):
    """Return the fitted search and folds; never reads or scores the test set."""
    return tune_model(build_random_forest(random_state), PARAMETER_SPACE, X, y,
                      folds, random_state, n_jobs, n_iter)


def main(argv=None):
    return run_cli("random_forest", build_random_forest, PARAMETER_SPACE, argv)


if __name__ == "__main__":
    main()
