"""Formal decision-tree experiment with train-only hyperparameter selection.

    python -B -m src.decision_tree
    python -B -m src.decision_tree --evaluate-test --n-jobs -1

See EXPERIMENTS.md for the shared protocol and saved artifacts.
"""
from sklearn.tree import DecisionTreeClassifier

if __package__:
    from .experiment import cross_validate_model, run_cli, tune_model
    from .preprocessing import build_model_pipeline
else:
    from experiment import cross_validate_model, run_cli, tune_model
    from preprocessing import build_model_pipeline

# Finite, declared search space; 24 candidates are sampled by default.
PARAMETER_SPACE = {
    "model__criterion": ["gini", "entropy"],
    "model__max_depth": [None, 4, 8, 12, 20],
    "model__min_samples_split": [2, 10, 30],
    "model__min_samples_leaf": [1, 5, 10, 20],
    "model__class_weight": [None, "balanced"],
    "model__ccp_alpha": [0.0, 0.0001, 0.001],
}


def build_decision_tree(random_state=42, max_depth=None, min_samples_leaf=1,
                        min_samples_split=2, criterion="gini", class_weight=None):
    """Return an unfitted decision tree pipeline without numeric scaling."""
    return build_model_pipeline(
        DecisionTreeClassifier(
            random_state=random_state,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            criterion=criterion,
            class_weight=class_weight,
        ),
        scale_numeric=False,
    )



def cross_validate_tree(pipeline, X, y, folds=5, random_state=42, n_jobs=1):
    """Compatibility wrapper for fixed-parameter CV using shared evaluation."""
    return cross_validate_model(pipeline, X, y, folds, random_state, n_jobs)


def run_experiment(X, y, folds=5, random_state=42, n_jobs=1, n_iter=24):
    """Return the fitted search and folds; never reads or scores the test set."""
    return tune_model(build_decision_tree(random_state), PARAMETER_SPACE, X, y,
                      folds, random_state, n_jobs, n_iter)


def main(argv=None):
    return run_cli("decision_tree", build_decision_tree, PARAMETER_SPACE, argv)


if __name__ == "__main__":
    main()
