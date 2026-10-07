# IN6227 Assignment 1: Binary Classification and Evaluation

[中文文档](README.md)

This project uses the course-provided Variant 1 dataset to predict the target field `label` (`no` / `yes`) and compare decision tree and random forest classifiers. It includes data auditing, exploratory data analysis (EDA), preprocessing, baseline comparison, hyperparameter search, held-out test evaluation, and controlled feature engineering experiments.

## Project Structure

```text
IN6227-Assignment-1/
├── data/
│   ├── train.csv                 # Training data
│   └── test.csv                  # Held-out test data
├── notebooks/
│   └── EDA.ipynb                 # Training data exploration and visualization
├── src/
│   ├── paths.py                  # Project path management
│   ├── inspect_data.py           # Data integrity and distribution audit
│   ├── preprocessing.py          # Schema validation, imputation, and encoding
│   ├── baseline.py               # Comparison of untuned models
│   ├── experiment.py             # Shared cross-validation, search, and export
│   ├── decision_tree.py          # Decision tree tuning entry point
│   ├── random_forest.py          # Random forest tuning entry point
│   ├── evaluate.py               # Shared evaluation metrics
│   ├── evaluate_saved.py         # Evaluation of saved models
│   └── feature_engineering.py    # Controlled feature comparison
├── tests/                        # unittest tests
├── outputs/                      # Audits, figures, models, and experiment records
├── requirements.txt
├── README.md                     # Chinese documentation
└── README_EN.md                  # English documentation
```

## Environment Setup

The saved experiments used Python 3.10.1, scikit-learn 1.7.2, pandas 2.3.3, and NumPy 2.2.6. Dependencies are listed in `requirements.txt`, which requires scikit-learn 1.2 or later. Dependency versions are not fully pinned, so a fresh installation may use different versions and produce different results.

Run the following commands from the project root:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

In the commands below, `python` should refer to this virtual environment's interpreter. You can also replace it directly with `.venv\Scripts\python.exe`. To run the notebook, install and launch Jupyter separately:

```powershell
.venv\Scripts\python.exe -m pip install jupyterlab
.venv\Scripts\python.exe -m jupyterlab notebooks/EDA.ipynb
```

## Data and Preprocessing

Each input CSV must contain the following 15 features and `label`. The loader checks for missing or extra columns.

| Type | Fields |
| --- | --- |
| Numerical features (7) | `aptitude_score`, `performance_score`, `stability_index`, `load_ratio`, `index_weight`, `activity_duration`, `composite_rank` |
| Categorical features (8) | `geological_era`, `weather_pattern`, `region`, `personal_interest`, `brew_preference`, `instrument`, `species`, `mineral_type` |
| Target | `label`: `no → 0`, `yes → 1`; `yes` is the positive class |

According to the saved experiment records, the training set contains 31,112 rows, of which 31,109 are labeled. The test set contains 13,334 rows, of which 13,333 are labeled.

- Leading and trailing whitespace is removed. Only empty or whitespace-only cells are treated as missing; `Unknown` and `Other` remain ordinary categories.
- Rows with missing targets are excluded from training. Target labels are never imputed. During test evaluation, predictions are generated for all rows, but metrics use only labeled rows.
- Missing numerical values are imputed using the training fold's median. Missing categorical values are replaced with `__MISSING__`, followed by one-hot encoding that ignores unseen categories during prediction.
- Numerical scaling is not applied to the decision tree or random forest. Imputation and encoding are fitted inside each training fold through a Pipeline to prevent validation data leakage.
- Invalid numerical values, infinities, and unexpected labels raise errors. The audit does not automatically remove outliers or duplicate records.

## Workflow

Run all commands below from the project root. Relative data and output paths supplied to the scripts are resolved against the project root by default.

### 1. Data Audit and EDA

```powershell
python -m src.inspect_data
```

By default, this reads only the training set and writes missing-value counts, class distributions, duplicate counts, correlations, and IQR outlier diagnostics to `outputs/data_audit.json`. To additionally check test data integrity and records whose features match training records, explicitly run:

```powershell
python -m src.inspect_data --include-test-audit
```

Open `notebooks/EDA.ipynb` in Jupyter and run its cells in order. Figures and statistical tables are saved to `outputs/eda/`. If the notebook cannot locate the data, check `PROJECT_ROOT_OVERRIDE` in its path configuration cell.

### 2. Baseline Comparison

```powershell
python -m src.baseline
```

The baselines compare a default decision tree and a random forest with 100 trees using identical stratified cross-validation splits. The winner is selected by mean validation AP. This entry point prints results without automatically saving models or reports, and does not read the test set by default.

### 3. Hyperparameter Search

```powershell
python -m src.decision_tree --n-jobs -1
python -m src.random_forest --n-jobs -1
```

Both entry points use `RandomizedSearchCV`. Defaults are a random seed of 42, five stratified folds, and 24 sampled parameter combinations. Parameters are selected by mean validation Average Precision (AP), and the best Pipeline is refitted on all labeled training data.

The decision tree search covers the split criterion, depth, minimum samples for splits and leaves, class weights, and pruning parameter. The random forest search covers the number of trees, depth, minimum samples for splits and leaves, feature sampling, and class weights.

| Argument | Default | Description |
| --- | --- | --- |
| `--train` | `data/train.csv` | Training data |
| `--test` | `data/test.csv` | Final evaluation data |
| `--folds` | `5` | Number of cross-validation folds |
| `--seed` | `42` | Random seed |
| `--n-iter` | `24` | Number of sampled search candidates |
| `--n-jobs` | `1` | Parallel jobs; `-1` uses all available CPUs |
| `--output-dir` | `outputs/<model_name>` | Parent directory for experiment outputs |
| `--evaluate-test` | Disabled | Evaluate the test set after the search |

For a quick check that the workflow runs, reduce the search size. These results are not directly comparable to a full experiment:

```powershell
python -m src.decision_tree --folds 3 --n-iter 2
```

### 4. Final Test Evaluation

Once the model configuration is finalized, load a saved formal experiment to evaluate it without retraining. The following commands use model directories already present in the project:

```powershell
python -m src.evaluate_saved --run-dir outputs/decision_tree/20261005T063807_780755Z
python -m src.evaluate_saved --run-dir outputs/random_forest/20261005T061626_884619Z
```

Results are saved under the selected run's `test_evaluations/<UTC_timestamp>/` directory, including `test_metrics.json`, `test_predictions.csv`, and `evaluation.json`. After retraining, replace `--run-dir` with the new experiment directory.

Alternatively, append `--evaluate-test` to a training command to perform both search and evaluation. This retrains the model and saves test outputs directly in that experiment directory. Use the test set only for final evaluation, not for selecting parameters, features, or thresholds.

### 5. Controlled Feature Engineering Experiment

```powershell
python -m src.feature_engineering
```

This experiment holds model parameters fixed and compares four feature variants using identical cross-validation splits:

| Variant | Transformation |
| --- | --- |
| `baseline` | Original 15 features |
| `reduced` | Remove the continuous numerical representation of `stability_index` |
| `engineered` | Add three features: the number of missing feature values, whether `performance_score` and `stability_index` are both zero, and whether `load_ratio` and `activity_duration` are both zero |
| `reduced_engineered` | Combine feature removal and derived features |

A shared feature variant is selected by averaging validation AP across the two fixed models. The saved experiment selected `reduced_engineered`. This variant still requires `stability_index` as an input for the joint-zero feature. If either source value is missing, its joint-zero flag also remains missing.

This experiment is separate from the formal tuning entry points. `decision_tree.py` and `random_forest.py` continue to use the original feature preprocessing and do not automatically adopt the selected variant. Feature experiments save `<model_name>_pipeline.joblib` and `selection.json`; they cannot be passed directly to `evaluate_saved`, which expects `best_pipeline.joblib` and `experiment.json`.

## Saved Experiment Results

The following values come from existing project outputs and are rounded to four decimal places. Models were not retrained to produce this README.

| Model | Validation AP (mean ± SD) | Test AP | Test ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Decision tree | 0.6622 ± 0.0111 | 0.6535 | 0.8736 | 0.8271 | 0.6397 | 0.6237 | 0.6316 |
| Random forest | 0.7155 ± 0.0198 | 0.7085 | 0.8903 | 0.8393 | 0.7212 | 0.5275 | 0.6093 |

Result sources:

- Decision tree: [validation summary](outputs/decision_tree/20261005T063807_780755Z/best_cv_summary.csv), [test metrics](outputs/decision_tree/20261005T063807_780755Z/test_evaluations/20261007T073952_132486Z/test_metrics.json).
- Random forest: [validation summary](outputs/random_forest/20261005T061626_884619Z/best_cv_summary.csv), [test metrics](outputs/random_forest/20261005T061626_884619Z/test_evaluations/20261007T074201_753309Z/test_metrics.json).

The random forest performs better on AP, the primary selection metric. The decision tree achieves higher positive-class recall and F1 under the current default prediction rule. AP means Average Precision, not trapezoidal PR-AUC. Precision, recall, and F1 refer to the positive class `yes`. Confusion matrices use the order `[no, yes]`, with true classes in rows and predicted classes in columns.

Cross-validation scores after tuning are model selection scores and should not be treated as unbiased estimates of generalization. The reported standard deviations are sample standard deviations across folds, not confidence intervals.

## Output Files

Each formal training run creates a separate directory named with a UTC timestamp under `outputs/decision_tree/` or `outputs/random_forest/`.

| File | Contents |
| --- | --- |
| `best_pipeline.joblib` | Complete fitted preprocessing and model Pipeline |
| `best_params.json` | Best hyperparameters |
| `cv_results.csv` | Training and validation results for all search candidates |
| `best_cv_folds.csv` | Per-fold validation metrics for the best candidate |
| `best_cv_summary.csv` | Metric means and sample standard deviations for the best candidate |
| `cv_fold_assignments.csv` | Validation fold assignment for each original record |
| `experiment.json` | Metadata including seed, search space, data hash, and dependency versions |
| `test_metrics.json` | Metrics generated when test evaluation is explicitly requested |
| `test_predictions.csv` | Original row index, true label, predicted label, and positive-class probability |

`row_index` is the zero-based original CSV record index, not the physical file line number. Feature engineering outputs are stored separately in `outputs/feature_engineering/`. EDA figures and tables are stored in `outputs/eda/figures/` and `outputs/eda/tables/`, respectively.

## Tests

```powershell
python -B -m unittest discover -s tests -v
```

The existing tests cover evaluation metrics, cross-validation and search workflows, feature transformations, and independent evaluation of saved models. When reproducing experiments, use the recorded data SHA-256 hashes, dependency versions, random seed, and fold assignments to verify the run conditions.
