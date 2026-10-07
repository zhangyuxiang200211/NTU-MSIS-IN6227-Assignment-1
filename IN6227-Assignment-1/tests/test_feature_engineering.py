"""Behavior checks for EDA-driven features and fold-local preprocessing."""
import unittest
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from src.preprocessing import FEATURE_COLUMNS, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS
from src.feature_engineering import AuditFeatures, build_feature_pipeline, VARIANTS


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame({c: [0., 1., 2., 3., 4., 5.] for c in NUMERIC_COLUMNS},
                              index=[10, 11, 12, 13, 14, 15])
        for c in CATEGORICAL_COLUMNS:
            self.X[c] = ["Unknown", "Other", "A", "B", "A", "B"]
        self.X.loc[11, "performance_score"] = np.nan
        self.y = [0, 0, 1, 0, 1, 1]

    def test_missing_is_not_zero_and_index_is_preserved(self):
        original = self.X.copy(deep=True)
        result = AuditFeatures("reduced_engineered").fit_transform(self.X)
        self.assertTrue(result.index.equals(self.X.index))
        self.assertNotIn("stability_index", result)
        self.assertEqual(result.loc[10, "performance_stability_both_zero"], 1.)
        self.assertTrue(pd.isna(result.loc[11, "performance_stability_both_zero"]))
        self.assertEqual(result.loc[11, "missing_feature_count"], 1.)
        self.assertEqual(result.loc[12, "performance_stability_both_zero"], 0.)
        pd.testing.assert_frame_equal(self.X, original)

    def test_fold_statistics_unseen_categories_and_cloning(self):
        for variant in VARIANTS:
            pipeline = clone(build_feature_pipeline(DecisionTreeClassifier(random_state=42), variant))
            pipeline.fit(self.X, self.y)
            pre = pipeline.named_steps["preprocessor"]
            numeric = pre.named_transformers_["numeric"]
            imputer = numeric.named_steps["imputer"] if variant == "baseline" else numeric
            self.assertEqual(imputer.statistics_[0], 2.5)
            heldout = self.X.iloc[:2].copy()
            heldout["aptitude_score"] = [np.nan, 100000.]
            heldout["species"] = "unseen"
            self.assertEqual(pipeline.predict_proba(heldout).shape, (2, 2))
            self.assertEqual(imputer.statistics_[0], 2.5)
            categories = pre.named_transformers_["categorical"].named_steps["encoder"].categories_
            self.assertIn("Unknown", categories[0])
            self.assertIn("Other", categories[0])
            self.assertNotIn("unseen", categories[CATEGORICAL_COLUMNS.index("species")])

    def test_cli_entry_points(self):
        project = Path(__file__).resolve().parents[1]
        commands = [(["-m", "src.feature_engineering", "--help"], project)]
        with tempfile.TemporaryDirectory() as outside_project:
            commands.append(([str(project / "src" / "feature_engineering.py"), "--help"], outside_project))
            for arguments, directory in commands:
                result = subprocess.run([sys.executable, "-B", *arguments],
                                        cwd=directory, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--folds", result.stdout)

    def test_schema_and_feature_names(self):
        transformer = AuditFeatures("engineered").fit(self.X)
        self.assertEqual(len(transformer.get_feature_names_out()), len(FEATURE_COLUMNS) + 3)
        with self.assertRaises(ValueError):
            transformer.transform(self.X.drop(columns="performance_score"))
        with self.assertRaises(ValueError):
            AuditFeatures("invalid").fit(self.X)


if __name__ == "__main__":
    unittest.main()
