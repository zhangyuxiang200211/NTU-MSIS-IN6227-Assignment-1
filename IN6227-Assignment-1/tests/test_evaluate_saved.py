"""Saved-model evaluation must reuse fitted preprocessing and never refit."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.decision_tree import build_decision_tree
from src.random_forest import build_random_forest
from src.evaluate import evaluate_model
from src.evaluate_saved import main
from src.experiment import file_digest, write_json
from src.preprocessing import (
    NUMERIC_COLUMNS, CATEGORICAL_COLUMNS, load_dataset, split_features_target,
)


class SavedEvaluationTests(unittest.TestCase):
    def test_saved_models_without_refitting_and_preserve_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            frame = pd.DataFrame({c: np.arange(20, dtype=float) for c in NUMERIC_COLUMNS})
            for column in CATEGORICAL_COLUMNS:
                frame[column] = ["A", "B"] * 10
            frame["label"] = ["no", "yes"] * 10
            train_path, test_path = base / "train.csv", base / "test.csv"
            frame.to_csv(train_path, index=False)
            X, y = split_features_target(load_dataset(train_path))
            frame.loc[0, "label"] = None
            frame.loc[1, NUMERIC_COLUMNS[0]] = np.nan
            frame.loc[2, CATEGORICAL_COLUMNS[0]] = "unseen"
            frame.to_csv(test_path, index=False)
            X_test, y_test = split_features_target(load_dataset(test_path), False)
            for name, pipeline in [
                ("decision_tree", build_decision_tree(max_depth=2)),
                ("random_forest", build_random_forest(n_estimators=3, max_depth=2)),
            ]:
                run = base / name
                run.mkdir()
                pipeline.fit(X, y.astype(int))
                joblib.dump(pipeline, run / "best_pipeline.joblib")
                write_json(run / "experiment.json", {
                    "model": name, "train_path": str(train_path),
                    "train_sha256": file_digest(train_path), "best_params": {},
                    "test_evaluated": False,
                })
                before = {p.name: p.read_bytes() for p in run.iterdir()}
                expected, _ = evaluate_model(pipeline, X_test, y_test)
                with patch.object(Pipeline, "fit", side_effect=AssertionError("Unexpected refit")), \
                        contextlib.redirect_stdout(io.StringIO()):
                    metrics, rows, output = main([
                        "--run-dir", str(run), "--test", str(test_path),
                    ])
                    _, _, second_output = main([
                        "--run-dir", str(run), "--test", str(test_path),
                    ])
                self.assertEqual(metrics, expected)
                self.assertEqual(len(rows), 20)
                self.assertEqual(metrics["scored_rows"], 19)
                self.assertTrue(pd.isna(rows.iloc[0]["true_label"]))
                self.assertNotEqual(output, second_output)
                self.assertEqual(json.loads((output / "test_metrics.json").read_text()), expected)
                self.assertFalse(json.loads((output / "evaluation.json").read_text())["refitted"])
                self.assertEqual(len(pd.read_csv(output / "test_predictions.csv")), 20)
                for filename, original in before.items():
                    self.assertEqual((run / filename).read_bytes(), original)
                # Reject both the training file and a renamed identical copy.
                copy_path = base / "train_copy.csv"
                copy_path.write_bytes(train_path.read_bytes())
                for invalid in [train_path, copy_path]:
                    with self.assertRaises(ValueError):
                        main(["--run-dir", str(run), "--test", str(invalid)])
                with self.assertRaises(FileNotFoundError):
                    main(["--run-dir", str(base / "missing"), "--test", str(test_path)])


if __name__ == "__main__":
    unittest.main()
