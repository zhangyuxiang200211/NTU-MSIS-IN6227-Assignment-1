"""Shared binary-classification evaluation for both formal experiments.

Positive class: yes=1. AP means average precision, not trapezoidal PR-AUC.
Threshold metrics use estimator.predict (ties follow sklearn's class order).
Missing targets are excluded from scoring, but all rows receive predictions.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)

METRIC_NAMES = (
    "average_precision", "roc_auc", "accuracy", "balanced_accuracy",
    "precision", "recall", "f1",
)
SELECTION_METRIC = "average_precision"


def evaluate_predictions(y_true, y_pred, probability_yes):
    """Evaluate aligned, one-dimensional arrays; undefined metrics are None.

    Inputs are positional. Use evaluate_model for indexed pandas data.
    Confusion matrix rows=true, columns=predicted, both ordered [no, yes].
    """
    truth = pd.Series(y_true).reset_index(drop=True)
    predicted = np.asarray(y_pred)
    probability = np.asarray(probability_yes, dtype=float)
    if predicted.ndim != 1 or probability.ndim != 1:
        raise ValueError("Predictions and probabilities must be one-dimensional")
    if len(truth) != len(predicted) or len(truth) != len(probability):
        raise ValueError("Targets, predictions and probabilities must have equal lengths")
    known = truth.notna().to_numpy()
    if not truth.loc[known].isin([0, 1]).all():
        raise ValueError("Known targets must be 0=no or 1=yes")
    if not np.isin(predicted, [0, 1]).all():
        raise ValueError("Predictions must be 0=no or 1=yes")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Probabilities must be finite values in [0, 1]")
    result = {
        "total_rows": len(truth), "scored_rows": int(known.sum()),
        "missing_label_rows": int((~known).sum()),
        **{name: None for name in METRIC_NAMES},
        "confusion_matrix": None, "classification_report": None,
    }
    if not known.any():
        return result
    actual = truth.loc[known].astype(int).to_numpy()
    pred, prob = predicted[known], probability[known]
    result.update({
        "accuracy": float(accuracy_score(actual, pred)),
        "precision": float(precision_score(actual, pred, zero_division=0)),
        "recall": float(recall_score(actual, pred, zero_division=0)),
        "f1": float(f1_score(actual, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(actual, pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            actual, pred, labels=[0, 1], target_names=["no", "yes"],
            zero_division=0, output_dict=True,
        ),
    })
    if np.unique(actual).size == 2:
        result.update({
            "balanced_accuracy": float(balanced_accuracy_score(actual, pred)),
            "average_precision": float(average_precision_score(actual, prob)),
            "roc_auc": float(roc_auc_score(actual, prob)),
        })
    return result


def evaluate_model(estimator, X, y):
    """Score an already fitted model without refitting it; return metrics, rows."""
    if len(X) != len(y) or len(X) == 0:
        raise ValueError("X and y must have the same nonzero number of rows")
    if isinstance(y, pd.Series) and hasattr(X, "index") and not X.index.equals(y.index):
        raise ValueError("X and y row indices must match in order")
    classes = np.asarray(estimator.classes_)
    if set(classes.tolist()) != {0, 1}:
        raise ValueError("The fitted estimator must contain classes 0 and 1")
    predicted = estimator.predict(X)
    probability = estimator.predict_proba(X)[:, int(np.flatnonzero(classes == 1)[0])]
    metrics = evaluate_predictions(y, predicted, probability)
    index = X.index if hasattr(X, "index") else pd.RangeIndex(len(X))
    rows = pd.DataFrame({
        "true_label": pd.array(y, dtype="Int64"),
        "predicted_label": predicted,
        "probability_yes": probability,
    }, index=index)
    rows.index.name = "row_index"
    return metrics, rows


def score_estimator(estimator, X, y):
    """Multi-metric sklearn scorer using the exact same evaluation function."""
    metrics, _ = evaluate_model(estimator, X, y)
    if any(metrics[name] is None for name in METRIC_NAMES):
        raise ValueError("Cross-validation scoring requires both target classes")
    return {name: metrics[name] for name in METRIC_NAMES}
