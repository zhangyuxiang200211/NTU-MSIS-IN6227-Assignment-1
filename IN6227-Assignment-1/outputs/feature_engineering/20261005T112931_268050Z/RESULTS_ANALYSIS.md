# Feature engineering results analysis

Source run: 20261005T112931_268050Z. Files inspected: selection.json, selection_scores.csv, cv_summary.csv, cv_folds.csv and feature_names.json. The run used 31,109 labeled training records, seed 42 and five stratified folds; three missing-label records were excluded. No test data was used. Results below describe fixed model settings, not tuned model optima.

## Main comparison

| Recipe | Decision tree mean AP | Change vs baseline | Random forest mean AP | Change vs baseline | Mean AP across models |
|---|---:|---:|---:|---:|---:|
| baseline | 0.633165 | 0 | 0.712547 | 0 | 0.672856 |
| reduced | 0.631459 | -0.001707 | 0.712932 | +0.000385 | 0.672195 |
| engineered | 0.632443 | -0.000723 | 0.712041 | -0.000506 | 0.672242 |
| reduced_engineered | 0.631653 | -0.001512 | 0.714192 | +0.001645 | 0.672923 |

The declared rule selected reduced_engineered because its mean AP across the two models was highest. Its advantage over baseline was only 0.00006651 AP (0.006651 percentage points on a percentage scale). This is a numerical winner under the specified rule, not evidence of a meaningful overall gain.

## Paired-fold evidence

Comparisons below subtract the baseline AP on the same fold, rather than comparing independent fold distributions.

| Model | Candidate | Folds with higher AP | Mean paired change | Sample SD of paired changes |
|---|---|---:|---:|---:|
| Decision tree | reduced | 1/5 | -0.001707 | 0.002747 |
| Decision tree | engineered | 3/5 | -0.000723 | 0.001700 |
| Decision tree | reduced_engineered | 0/5 | -0.001512 | 0.001081 |
| Random forest | reduced | 3/5 | +0.000385 | 0.001193 |
| Random forest | engineered | 2/5 | -0.000506 | 0.002275 |
| Random forest | reduced_engineered | 4/5 | +0.001645 | 0.001727 |

The decision tree favors the baseline in mean AP; the combined recipe loses in all five folds. The random forest shows a small, fairly consistent AP benefit from the combined recipe, although fold 5 decreases. The cross-model paired gain is positive in only three folds, with a sample SD of 0.001101, much larger than its mean of 0.00006651. These are descriptive comparisons; overlapping training folds, candidate selection and prior EDA prevent treating them as independent confirmatory trials. No statistical significance claim is made.

## Other metrics: random forest

| Metric | baseline | reduced_engineered | Absolute change |
|---|---:|---:|---:|
| AP | 0.712547 | 0.714192 | +0.001645 |
| ROC-AUC | 0.890228 | 0.890749 | +0.000521 |
| Accuracy | 0.838761 | 0.839821 | +0.001061 |
| Precision | 0.707307 | 0.713222 | +0.005915 |
| Recall | 0.559622 | 0.556138 | -0.003484 |
| F1 | 0.624694 | 0.624874 | +0.000180 |
| Balanced accuracy | 0.743249 | 0.742755 | -0.000494 |

At the existing prediction rule, increased precision is accompanied by reduced recall; F1 is almost unchanged. The engineered-only forest has the highest F1 of the four recipes (0.625915), whereas the combined recipe has the highest AP. These optimize different properties; selection should follow the declared AP objective rather than switching metrics after seeing results.

## Interpretation of feature decisions

Removing the continuous stability_index feature alone barely benefits the forest and lowers the tree's AP. Near-perfect rank correlation with performance_score therefore supports investigating redundancy, but not assuming deletion improves all models. Differences may reflect missing values, numerical rounding and changes in tree construction.

Adding all three engineered features alone lowers mean AP in both models. Consequently there is no empirical support here for a general improvement from missing counts and joint-zero flags. The trees can already learn zero thresholds, and the missing and joint-zero patterns are relatively sparse; these are plausible explanations, not experimentally isolated causes.

The combined recipe behaves differently from either individual change, particularly for the forest. Its sqrt feature sampling can respond to changes in feature count and representation. The experiment does not isolate the contribution of each of the three engineered features, so no individual feature can be credited with the observed gain.

The automatically selected reduced_engineered recipe has 17 pre-encoding features and 92 columns after fitting the encoder on all labeled training records: six original numeric features, three engineered features, and 83 one-hot columns. Both saved feature-name lists agree. stability_index remains necessary as an input to the joint-zero indicator, so the selected recipe does not remove the need to collect that raw variable. The two saved pipelines use this shared recipe, even though the decision tree performs better with baseline features.

## Practical conclusion

The final feature policy is to retain all 15 original predictors and use the baseline configuration for both the decision tree and random forest. Although the automated maximum-mean-AP rule selected reduced_engineered, its average AP across the two models exceeded the baseline by only 0.00006651. This small aggregate gain does not provide a compelling practical reason to add three engineered features, particularly because decision-tree AP decreased in all five folds. The random forest did benefit modestly from the combined configuration, improving AP in four of five folds, but this was accompanied by lower recall and almost no change in F1. Retaining the original predictors therefore provides a simpler common feature scheme for comparing the two classifiers, while preserving potentially complementary information in stability_index rather than removing it solely because of its strong correlation with performance_score. Standard preprocessing remains in place: missing numerical values are imputed, missing categorical values receive a separate token, and categorical predictors are one-hot encoded within each training fold. Retaining 15 original predictors refers to the input variables before encoding, not to the number of columns in the encoded model matrix.

This final choice is a documented practical decision made after reviewing the feature experiments and considering the preference to retain all 15 original predictors. It is not the outcome of the original maximum-mean-AP rule, and it does not imply that baseline achieved the highest AP across both models. The recorded selection.json and the saved pipelines from this experiment still correspond to reduced_engineered; they are retained as the historical experimental outputs and should not be described as baseline models. Subsequent model tuning and final evaluation should use the baseline feature policy. The existing decision_tree and random_forest entry points already construct baseline pipelines. No test-set results informed this decision, and final generalization performance must still be assessed on the reserved test set.

The forest exceeds the tree's AP by approximately 0.08 under these fixed settings, far more than any feature-recipe effect in this run. Model choice appears to have the larger practical impact in this experiment, without establishing that one model universally dominates after tuning.

## Report-ready paragraph

Four feature configurations were evaluated using identical five-fold stratified cross-validation splits. The combined reduced-and-engineered configuration achieved the highest mean average precision (AP) across the decision tree and random forest, at 0.672923 compared with 0.672856 for the baseline, representing an improvement of only 0.00006651. Its effect was model-dependent: random-forest AP increased from 0.712547 to 0.714192, with improvements in four of five folds, whereas decision-tree AP decreased from 0.633165 to 0.631653, with decreases in all five folds. Adding the engineered features alone did not improve mean AP for either classifier. Furthermore, the combined random-forest configuration increased precision but reduced recall, leaving F1 almost unchanged. After reviewing these results, we chose to retain all 15 original predictors and adopt the baseline feature configuration for both classifiers. This choice reflects our preference to preserve the original feature set and maintain a simpler, consistent preprocessing scheme, as the negligible aggregate AP improvement did not provide a compelling justification for the additional engineered features. The highly correlated stability_index predictor was retained because removing it did not consistently improve performance. Numerical imputation and categorical one-hot encoding were retained and fitted within each cross-validation training fold. Although the automated maximum-mean-AP rule selected the combined configuration, the final baseline choice was a separate, explicitly documented practical decision and should not be interpreted as the numerical winner under that rule. These cross-validation results were used for feature-policy selection; final generalization performance remains to be evaluated on the reserved test set.
