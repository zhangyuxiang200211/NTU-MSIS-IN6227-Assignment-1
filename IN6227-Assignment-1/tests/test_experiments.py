"""Run: python -B -m unittest discover -s tests -v"""
import tempfile
import unittest
from pathlib import Path

import joblib
import json
import numpy as np
import pandas as pd

from src.decision_tree import build_decision_tree
from src.random_forest import build_random_forest
from src.evaluate import evaluate_predictions, evaluate_model, score_estimator
from src.experiment import make_splits, tune_model, run_cli
from src.preprocessing import NUMERIC_COLUMNS, CATEGORICAL_COLUMNS


class EvaluationTests(unittest.TestCase):
    def test_known_metrics_and_missing_target(self):
        result = evaluate_predictions(
            [0, 0, 1, 1, pd.NA], [0, 1, 0, 1, 1], [0.1, 0.8, 0.2, 0.9, 0.4],
        )
        self.assertEqual(result["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(result["scored_rows"], 4)
        self.assertEqual(result["missing_label_rows"], 1)
        for name in ["accuracy", "balanced_accuracy", "precision", "recall", "f1"]:
            self.assertAlmostEqual(result[name], 0.5)
        self.assertAlmostEqual(result["roc_auc"], 0.75)
        self.assertAlmostEqual(result["average_precision"], 5 / 6)

    def test_undefined_metrics(self):
        for truth in [[0, 0], [1, 1], [pd.NA, pd.NA]]:
            result = evaluate_predictions(truth, [0, 1], [0.2, 0.8])
            for name in ["roc_auc", "average_precision", "balanced_accuracy"]:
                self.assertIsNone(result[name])
        self.assertIsNone(result["accuracy"])

    def test_invalid_inputs(self):
        for truth, pred, prob in [([0], [0, 1], [0.2]), ([2], [0], [0.2]),
                                  ([0], [0], [float("nan")]), ([0], [0], [1.1])]:
            with self.assertRaises(ValueError):
                evaluate_predictions(truth, pred, prob)


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame({name: np.arange(24, dtype=float) for name in NUMERIC_COLUMNS})
        for name in CATEGORICAL_COLUMNS:
            self.X[name] = ["A", "B"] * 12
        self.X.loc[0, NUMERIC_COLUMNS[0]] = np.nan
        self.X.loc[1, CATEGORICAL_COLUMNS[0]] = np.nan
        self.X.index = pd.RangeIndex(100, 124)
        self.y = pd.Series([0, 1] * 12, index=self.X.index, dtype="Int64")

    def test_shared_folds_search_and_evaluation(self):
        splits = make_splits(self.X, self.y, folds=2)
        for builder in [build_decision_tree, build_random_forest]:
            pipeline = builder()
            space = {"model__max_depth": [2, 3]}
            if builder is build_random_forest:
                space["model__n_estimators"] = [3]
            search, actual = tune_model(pipeline, space, self.X, self.y,
                                        folds=2, n_iter=2, verbose=0)
            for expected_fold, actual_fold in zip(splits, actual):
                np.testing.assert_array_equal(expected_fold[1], actual_fold[1])
            test_X = self.X.copy()
            test_X.loc[100, CATEGORICAL_COLUMNS[0]] = "unseen category"
            truth = self.y.copy()
            truth.iloc[0] = pd.NA
            metrics, rows = evaluate_model(search.best_estimator_, test_X, truth)
            self.assertEqual(metrics["scored_rows"], 23)
            self.assertTrue(rows.index.equals(test_X.index))
            self.assertTrue(pd.isna(rows.iloc[0]["true_label"]))
            complete, _ = evaluate_model(search.best_estimator_, test_X, self.y)
            scores = score_estimator(search.best_estimator_, test_X, self.y)
            self.assertEqual(scores, {key: complete[key] for key in scores})
            with self.assertRaises(ValueError):
                evaluate_model(search.best_estimator_, test_X, self.y.iloc[::-1])

    def test_cli_saves_artifacts_without_reading_test(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            frame = self.X.copy()
            frame["label"] = self.y.map({0: "no", 1: "yes"})
            train = base / "train.csv"
            frame.to_csv(train, index=False)
            run_cli("smoke", build_decision_tree, {"model__max_depth": [2]}, [
                "--train", str(train), "--test", str(base / "does_not_exist.csv"),
                "--output-dir", str(base / "out"), "--folds", "2", "--n-iter", "1",
            ])
            runs = list((base / "out").iterdir())
            self.assertEqual(len(runs), 1)
            self.assertTrue((runs[0] / "best_pipeline.joblib").is_file())
            self.assertTrue((runs[0] / "experiment.json").is_file())
            self.assertFalse((runs[0] / "test_metrics.json").exists())


    def test_final_test_exports_and_serialization(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            frame = self.X.copy()
            frame["label"] = self.y.map({0: "no", 1: "yes"})
            train, test = base / "train.csv", base / "test.csv"
            frame.to_csv(train, index=False)
            frame.loc[100, "label"] = None
            frame.to_csv(test, index=False)
            for builder in [build_decision_tree, build_random_forest]:
                space = {"model__max_depth": [2]}
                if builder is build_random_forest:
                    space["model__n_estimators"] = [3]
                output = base / builder.__name__
                search = run_cli("smoke", builder, space, [
                    "--train", str(train), "--test", str(test),
                    "--output-dir", str(output), "--folds", "2",
                    "--n-iter", "1", "--evaluate-test",
                ])
                run = next(output.iterdir())
                metrics = json.loads((run / "test_metrics.json").read_text())
                metadata = json.loads((run / "experiment.json").read_text())
                self.assertEqual(metrics["scored_rows"], 23)
                self.assertEqual(metrics["missing_label_rows"], 1)
                self.assertTrue(metadata["test_evaluated"])
                rows = pd.read_csv(run / "test_predictions.csv")
                self.assertEqual(len(rows), 24)
                self.assertTrue(pd.isna(rows.loc[0, "true_label"]))
                restored = joblib.load(run / "best_pipeline.joblib")
                np.testing.assert_allclose(restored.predict_proba(self.X),
                                           search.best_estimator_.predict_proba(self.X))


if __name__ == "__main__":
    unittest.main()
