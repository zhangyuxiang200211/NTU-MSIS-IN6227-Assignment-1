"""Evaluate a saved experiment pipeline without training or parameter search.

python -B -m src.evaluate_saved --run-dir outputs/random_forest/<run>
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn

if __package__:
    from .evaluate import evaluate_model
    from .experiment import file_digest, write_json
    from .paths import TEST_PATH, resolve_path
    from .preprocessing import load_dataset, split_features_target
else:
    from evaluate import evaluate_model
    from experiment import file_digest, write_json
    from paths import TEST_PATH, resolve_path
    from preprocessing import load_dataset, split_features_target


def evaluate_saved(run_dir, test_path=TEST_PATH, output_dir=None):
    """Load a trusted project experiment and save a separate test evaluation.

    The original training metadata and model are never modified. Each call
    creates a new output subdirectory, preserving earlier evaluation results.
    """
    run_dir = resolve_path(run_dir).resolve()
    test_path = resolve_path(test_path).resolve()
    model_path = run_dir / "best_pipeline.joblib"
    metadata_path = run_dir / "experiment.json"
    for path in (model_path, metadata_path, test_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    test_hash = file_digest(test_path)
    train_path = resolve_path(metadata["train_path"]).resolve()
    if (test_path == train_path
            or (train_path.exists() and test_path.samefile(train_path))
            or test_hash == metadata["train_sha256"]):
        raise ValueError("The test file must not be the training file or an identical copy")

    print(f"Loading saved model: {model_path}", flush=True)
    pipeline = joblib.load(model_path)
    test = load_dataset(test_path)
    X_test, y_test = split_features_target(test, drop_missing_labels=False)
    metrics, predictions = evaluate_model(pipeline, X_test, y_test)

    output_base = (resolve_path(output_dir).resolve() if output_dir is not None
                   else run_dir / "test_evaluations")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = output_base / timestamp
    output.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(output / "test_predictions.csv")
    write_json(output / "test_metrics.json", metrics)
    write_json(output / "evaluation.json", {
        "model": metadata["model"], "source_run": str(run_dir),
        "model_path": str(model_path), "model_sha256": file_digest(model_path),
        "source_experiment_sha256": file_digest(metadata_path),
        "best_params": metadata["best_params"],
        "training_versions": metadata.get("versions", {}),
        "test_path": str(test_path), "test_sha256": test_hash,
        "test_evaluated": True, "refitted": False,
        "evaluated_at_utc": timestamp,
        "positive_class": "yes=1", "confusion_matrix_order": ["no", "yes"],
        "prediction_rule": "estimator.predict; probability ties select class 0",
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__,
                     "joblib": joblib.__version__},
    })
    print("Final test evaluation (no training or search performed):")
    print(json.dumps(metrics, indent=2, allow_nan=False))
    print(f"Output: {output}", flush=True)
    return metrics, predictions, output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Saved run containing best_pipeline.joblib and experiment.json")
    parser.add_argument("--test", type=Path, default=TEST_PATH)
    parser.add_argument("--output-dir", type=Path,
                        help="Output parent; defaults to <run-dir>/test_evaluations")
    args = parser.parse_args(argv)
    return evaluate_saved(args.run_dir, args.test, args.output_dir)


if __name__ == "__main__":
    main()
