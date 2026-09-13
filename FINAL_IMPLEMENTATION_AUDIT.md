# Final Implementation Audit

## Audit Scope

This audit verifies the current executable implementation of the thesis prototype against the source code, Streamlit page paths, knowledge-base JSON files, configuration, tests, `PROJECT_TECHNICAL_DOCUMENTATION.md`, and `PROJECT_REVIEW_REPORT.md`.

No implementation files were changed for this audit. The current automated test run completed with **159 passed, 1 warning**. The warning is an existing scikit-learn `UndefinedMetricWarning` from a small regression report test.

## 1. End-to-End Workflow

### Finding

The complete workflow is connected and executable through the Streamlit application:

`CSV upload -> profiling -> target selection/task detection -> knowledge-based recommendation -> Top 5 display -> pipeline selection -> pipeline building -> train/test split -> preprocessing -> training -> evaluation -> explainability -> Markdown report`

### Executable path

- CSV upload: `app/views/upload_dataset.py::render()` calls `src.data_loader.load_csv()`, stores the DataFrame in Streamlit session state, and resets earlier workflow results.
- Profiling: `app/views/dataset_profile.py::render()` calls `src.data_profiler.profile_dataset()` and stores `profile_result`.
- Target selection and task detection: `app/views/task_detection.py::render()` stores `target_column` and calls `src.task_detector.detect_task_type()`.
- Recommendation: `app/views/recommendations.py::render()` calls `src.recommendation_engine.recommend_pipelines()`.
- Selection and construction: the selected recommendation is stored in session state; `app/views/training.py::render()` calls `src.pipeline_builder.build_pipeline()`.
- Training: the same page calls `src.model_trainer.train_pipeline()`.
- Evaluation: `app/views/evaluation.py::render()` calls `src.model_evaluator.evaluate_model()`.
- Explainability: `app/views/explainability.py::render()` calls `src.explainability.explain_model()` and optionally `generate_shap_explanation()`.
- Reporting: `app/views/report.py::render()` calls `src.report_generator.generate_report()` and downloads Markdown.

The application shell in `app/streamlit_app.py` defines the ordered pages, shared session state, page prerequisites, and navigation transitions.

### Important discrepancies

1. **PDF is documented as optional output but is not currently generated.** `src.report_generator.generate_report()` accepts `include_pdf` only for backward compatibility, explicitly ignores it, and always returns `pdf_bytes = None`. The Streamlit report page offers Markdown only.
2. **The report does not preserve the user-selected pipeline as a distinct report input.** `PipelineRecommendationResult` has no `selected_pipeline` field. `src.report_generator._build_selected_pipeline_summary()` falls back to `top_pipelines[0]`, so the report can describe the first-ranked pipeline rather than a different pipeline selected by the user.
3. The technical documentation describes the broader Markdown/PDF architecture and optional PDF fallback, but the current executable report path is Markdown-only.

## 2. Recommendation Engine

### Knowledge-base execution

Knowledge-based reasoning is genuinely executed. `src.knowledge_base.KnowledgeBase.load()` reads and validates JSON metadata, categories, pipeline archetypes, and rule files. `src.recommendation_engine.RecommendationEngine.recommend()` then:

1. Builds profile and task contexts.
2. Selects archetypes matching the detected task.
3. Selects a representative model family for each archetype.
4. Calls `_match_rules()` against all loaded rules.
5. Builds supporting recommendation items from matched rules.
6. Builds a pipeline recommendation with explanations, evidence, preprocessing strategy, and confidence.
7. Sorts and truncates the candidates to `TOP_PIPELINE_COUNT = 5`.

The current JSON inventory contains **10 pipeline archetypes** and **21 declarative rules**:

| Knowledge-base file | Rules |
| --- | ---: |
| `classification_model_rules.json` | 3 |
| `cross_validation_rules.json` | 2 |
| `dimensionality_rules.json` | 2 |
| `encoding_rules.json` | 2 |
| `evaluation_metric_rules.json` | 3 |
| `imbalance_rules.json` | 2 |
| `imputation_rules.json` | 2 |
| `regression_model_rules.json` | 3 |
| `scaling_rules.json` | 2 |
| **Total** | **21** |

### Actual archetypes

The 10 archetypes in `knowledge_base/pipeline_archetypes.json` are five classification archetypes and five regression archetypes:

- Balanced Linear Classification Pipeline
- Robust Tree Classification Pipeline
- Imbalanced Classification Pipeline
- KNN Classification Pipeline
- Naive Bayes Classification Pipeline
- Balanced Linear Regression Pipeline
- Support Vector Regression Pipeline
- Random Forest Regression Pipeline
- Robust Tree Regression Pipeline
- Regularized Regression Pipeline

Because the engine filters archetypes by task and then truncates to five, the normal successful path produces five classification or five regression recommendations. Fewer than five are possible if the knowledge base has fewer than five valid archetypes for a task, if loading/filtering is changed, or if recommendation computation fails. The current JSON has five for each task, so the current valid built-in task paths produce five.

### Representative actual rules

1. `knowledge_base/rules/imbalance_rules.json`, rule `IMB-C-002`: when `class_imbalance_ratio < 0.2`, consider SMOTE inside the training pipeline. It carries confidence weight `0.9`, high severity, and priority `1`.
2. `knowledge_base/rules/imputation_rules.json`, rule `IMP-A-001`: when `missing_values.total_count > 0`, use imputation. It carries confidence weight `0.95`.
3. `knowledge_base/rules/dimensionality_rules.json`, rule `DIM-A-001`: when `high_correlation_pairs.count > 0`, consider PCA to reduce redundancy. It carries confidence weight `0.84`.

### Dataset evidence used

`_build_profile_context()` and `_build_task_context()` expose row count, column count, numeric and categorical counts, high-cardinality count, high-correlation count, missingness totals, class imbalance, target metadata, task type, target uniqueness, and task confidence. These values are consumed by the rule matcher and by the archetype-fit scoring logic.

### Rule matching and confidence

Rules match only when every condition passes. Supported operators are equality, numeric `>`, `>=`, `<`, `<=`, and membership `in`.

For a matched rule, rule confidence is:

`clamp(confidence_weight * mean(condition_strengths))`

Each satisfied equality or membership condition has strength `1.0`; numeric conditions also currently resolve to `1.0` when satisfied and `0.0` otherwise. Pipeline confidence is:

`clamp(0.40 * task_confidence + 0.35 * supporting_rule_average + 0.25 * archetype_fit_score)`

If no supporting rules match, the supporting-rule average defaults to `0.5`. Pipelines are ranked by descending confidence and then pipeline name, and ranks/priorities are assigned after truncation.

### Explanations and evidence

Each pipeline receives:

- a human-readable explanation containing task, row count, numeric/categorical counts, missingness, optional class imbalance, and matched-rule count;
- an evidence list containing dataset/task facts and up to five matched rule reason templates;
- supporting recommendation items containing rule ID, category, recommendation, reason, confidence, severity, priority, evidence, advantages, and limitations.

The Recommendations page displays the top five, confidence, rank, model name, rationale, advantages, limitations, and preprocessing strategy. This is implemented behaviour, not documentation-only behaviour.

## 3. Preprocessing Implementation Audit

| Functionality | Classification | Evidence |
| --- | --- | --- |
| Numeric missing-value imputation | **AUTOMATICALLY EXECUTED** | `src/pipeline_builder.py::_build_preprocessor()` adds `SimpleImputer(strategy="median")` for numeric columns when numeric columns exist. Features above 50% missingness are excluded before transformer construction by `_columns_to_drop_by_missingness()`. |
| Categorical/boolean missing-value imputation | **AUTOMATICALLY EXECUTED** | `_build_preprocessor()` adds `SimpleImputer(strategy="most_frequent")` for categorical and boolean columns. |
| One-hot encoding | **AUTOMATICALLY EXECUTED** | `_build_preprocessor()` adds `OneHotEncoder(handle_unknown="ignore")` after categorical imputation when categorical columns exist. |
| StandardScaler | **AUTOMATICALLY EXECUTED** when selected | `_build_scaler()` returns `StandardScaler()` for a standard/scaling strategy. The current recommendation strategy normally selects standard scaling for non-tree families. |
| MinMaxScaler | **RECOMMENDED ONLY / BUILDER-SUPPORTED** | `_build_scaler()` can construct it, but current recommendation logic does not normally emit a MinMax strategy. |
| RobustScaler | **RECOMMENDED ONLY / BUILDER-SUPPORTED** | `_build_scaler()` can construct it, but current recommendation logic does not normally emit a Robust strategy. |
| Duplicate removal | **NOT IMPLEMENTED** | Profiling reports duplicate rows; no training transformer removes them. |
| Outlier treatment | **DISPLAYED ONLY** | IQR outlier counts are calculated by `src.data_profiler`, but no outlier treatment transformer is added. |
| Feature selection | **RECOMMENDED ONLY** | JSON rules and recommendation text can recommend feature selection, but `src/pipeline_builder.py` contains no feature-selection transformer. |
| PCA | **AUTOMATICALLY EXECUTED** when the recommendation strategy contains PCA | `PipelineBuilder.build()` calls `_should_include_pca()` and adds `PCA(n_components=0.95)`. Current dimensionality logic can return `PCA or feature selection` for non-regularized archetypes with high correlation or at least 10 numeric columns. |
| Class-imbalance detection | **AUTOMATICALLY EXECUTED** | `src.data_profiler._profile_target_column()` computes `class_imbalance_ratio`; `src.recommendation_engine._imbalance_strategy()` uses `< 0.2` and `< 0.4` thresholds. |
| SMOTE | **AUTOMATICALLY EXECUTED** when the selected classification strategy is SMOTE | `_should_use_smote()` detects the recommendation strategy, and `PipelineBuilder.build()` inserts `SMOTE(random_state=42)`. |

The documentation is too broad where it presents feature selection as part of the builder’s executable preprocessing. The current builder executes PCA but not feature selection.

## 4. Detailed SMOTE Verification

### Actual execution path

1. **Imbalance detection:** `src.data_profiler._profile_target_column()` calculates the target class ratio.
2. **Recommendation strategy:** `src.recommendation_engine._imbalance_strategy()` returns `SMOTE inside training folds` only for classification with `class_imbalance_ratio < 0.2`. For ratios from `0.2` to below `0.4`, it returns class weights or threshold tuning, but the current builder has no class-weight transformer; this is a recommendation-level phrase rather than an implemented class-weight step.
3. **Pipeline construction:** `src.pipeline_builder.PipelineBuilder.build()` creates steps in this order: `preprocessor`, optional `pca`, optional `smote`, `model`. When SMOTE is present it uses `imblearn.pipeline.Pipeline`.
4. **Train/test split:** `src.model_trainer.ModelTrainer.train()` removes missing-target rows, calls `train_test_split(test_size=0.2, random_state=42)`, and stores `X_train`, `X_test`, `y_train`, and `y_test`.
5. **Cross-validation:** `cross_val_score()` receives the imbalanced-learn pipeline and only `X_train`, `y_train`. The pipeline is cloned and fitted separately for each fold, so SMOTE is applied inside each training fold by the imbalanced-learn pipeline.
6. **Final fit:** `pipeline.fit(X_train, y_train)` applies preprocessing and SMOTE to training data before fitting the estimator.
7. **Held-out evaluation:** `src.model_evaluator.ModelEvaluator.evaluate()` reuses the stored test split and calls `pipeline.predict(test_features)`. The test data is not passed through SMOTE during prediction.

### Methodological conclusion

SMOTE is genuinely executed when activated. The pipeline ordering is methodologically appropriate: preprocessing and SMOTE occur inside the training pipeline, after the hold-out split, and the test set remains untouched by oversampling. Cross-validation also receives the pipeline rather than pre-resampled data, so SMOTE is fitted separately inside training folds.

No realistic SMOTE data-leakage problem was found in the current execution path. The implementation should still be described accurately as oversampling within training folds, not as a preprocessing operation applied to the full dataset.

### Direct tests

- `tests/test_pipeline_builder.py::test_smote_inclusion_for_classification` verifies an `imblearn` pipeline and a `SMOTE` step.
- `tests/test_model_trainer.py::test_smote_pipeline_trains_for_classification` verifies that a SMOTE pipeline trains successfully with cross-validation enabled by default.
- Regression rejection is tested by `test_smote_rejection_for_regression` in both pipeline-builder and trainer tests.

There is no dedicated assertion that inspects every CV fold or explicitly proves that the held-out test set is untouched, but the executable architecture and direct pipeline/training tests support that conclusion.

## 5. Actual Machine-Learning Models

### A. Implemented in the pipeline builder

`src.pipeline_builder._build_model()` supports:

- Classification: `LogisticRegression`, `RandomForestClassifier`, `DecisionTreeClassifier`, `SVC`, KNN, `GaussianNB`, and optional `XGBClassifier`.
- Regression: `LinearRegression`, `Ridge`, `Lasso`, `RandomForestRegressor`, `DecisionTreeRegressor`, `SVR`, and optional `XGBRegressor`.

### B. Reachable through current recommendations

The current archetype-selection logic in `src.recommendation_engine._select_model_family()` makes these model families reachable through normal recommendations:

- Classification: Logistic Regression, Random Forest, KNN, Gaussian Naive Bayes.
- Regression: Linear Regression, SVR, Random Forest Regressor, Ridge.

The five classification archetypes and five regression archetypes are all candidates, but several archetype preferred-family entries are not selected because `_select_model_family()` overrides them.

### C. Optional models

XGBoost is optional at import time. `xgboost` is listed in `requirements.txt`, and the builder creates `XGBClassifier`/`XGBRegressor` when manually requested and available. However, no current recommendation archetype selection returns the XGBoost model keys.

### D. Implemented but currently unreachable through normal recommendations

- Decision Tree Classifier
- SVC/SVM
- XGBClassifier
- Lasso
- Decision Tree Regressor
- XGBRegressor

These remain builder-supported and can be reached by directly constructing a `PipelineRecommendation` with the corresponding model family, but they are not accurate claims about models currently selected by the recommendation system’s normal built-in path.

### Thesis claim

The thesis can accurately claim that the **pipeline builder supports** the broader model list. It should claim that the **current recommendation system normally recommends** only Logistic Regression, Random Forest, KNN, GaussianNB, Linear Regression, SVR, Random Forest Regressor, and Ridge.

## 6. Training, Cross-Validation, and Evaluation

### Training and validation

- Train/test ratio: 80% / 20%, via `train_test_split(test_size=0.2)`.
- Random state: `42` for the split and relevant estimators/SMOTE.
- Classification stratification: used when at least two classes exist and every class has at least two samples.
- Classification CV: `StratifiedKFold(shuffle=True, random_state=42)`.
- Regression CV: `KFold(shuffle=True, random_state=42)`.
- Fold count: maximum five; reduced for small datasets. Classification folds are limited by the smallest class count. Regression folds are limited by training-set size.
- Classification CV scoring: `f1_weighted`.
- Regression CV scoring: `r2`.
- CV can be disabled through `train_pipeline(..., run_cross_validation=False)`; the Training page uses the default enabled mode.

### Evaluation metrics actually calculated

#### Classification

- Accuracy
- Weighted precision
- Weighted recall
- Weighted F1 in the backend result
- Macro F1 in the Streamlit evaluation table
- Confusion matrix
- ROC-AUC for binary classification when the pipeline exposes probabilities or a decision function
- Classification report with per-class precision, recall, F1-score, and support

ROC-AUC is calculated in the Evaluation page but is not stored in `EvaluationResult.classification_metrics`; the persisted backend result contains accuracy, weighted precision, weighted recall, and weighted F1.

#### Regression

- MAE
- MSE
- RMSE
- R²

Qualitative ratings are implemented in `src.model_evaluator._rate_quality()` using project-defined thresholds on accuracy for classification and R² for regression:

- `>= 0.90`: excellent
- `>= 0.80`: good
- `>= 0.65`: acceptable
- otherwise: weak

These are project heuristics, not academically validated universal performance standards.

## 7. Explainability Audit

### Recommendation-level explainability

Recommendation-level explanations are visible. The Recommendation page shows ranking, confidence, model family, “Why Recommended,” advantages, limitations, preprocessing steps, and selected-pipeline details. The underlying result contains matched supporting recommendation items with rule IDs, categories, reason templates, evidence, severity, priority, advantages, and limitations.

The explanation combines dataset/task evidence and matched rule count in `_build_explanation()`. The evidence list includes dataset characteristics and matched rule reasons. Therefore recommendation-level explainability is implemented, although it is rule/evidence transparency rather than a learned explanation model.

### Model-level explainability

The primary method is `sklearn.inspection.permutation_importance` in `src.explainability.explain_model()`.

- Input data: the cleaned held-out test split stored by training.
- Scoring: `f1_weighted` for classification and `r2` for regression.
- Repeats: 10 normally; reduced to 5 when the sample exceeds 1,000 rows or has more than 20 columns.
- Sampling: capped at 5,000 rows for permutation importance.
- Returned features: top 10, while the result also stores the full original-feature importance mapping.
- Feature identity: importance is mapped back to the original input columns, not the one-hot transformed columns.
- UI path: `app/views/explainability.py` calls `explain_model()`, stores `explainability_result`, and renders a table and chart.

### SHAP

**Status: PARTIALLY IMPLEMENTED.**

- `shap` is listed in `requirements.txt`.
- SHAP is imported defensively inside `generate_shap_explanation()`.
- `shap.TreeExplainer` is used for supported tree-based estimators.
- SHAP values are calculated on a sampled hold-out set, capped at 1,000 rows.
- The Explainability page exposes a button, sample-size control, chart, interpretation, and status messages.
- SHAP is not part of the automatic core explainability run; the user must request it.
- SHAP output does not reliably reach the report. `src.report_generator._build_shap_table()` looks for `explainability_result.shap_result`, but `ExplainabilityResult` has no `shap_result` field and the Streamlit page stores SHAP separately as `shap_explanation_result`.

### Other XAI methods

LIME, PDP, ICE, Anchors, and counterfactual explanations are not implemented or imported. Their absence is consistent with the stated bounded thesis scope and is not a freeze blocker.

## 8. Reporting

### Markdown

Markdown reporting is implemented and connected. `app/views/report.py` calls `generate_report()`, renders the generated Markdown, and exposes a Markdown download button. The report incorporates dataset summary, processing information, task detection, a pipeline summary, training, evaluation, missing-value handling, and explainability summary when available.

### PDF

PDF reporting is **not currently implemented in the executable system**:

- `reportlab` is not listed in `requirements.txt`.
- `generate_report(include_pdf=...)` ignores the argument.
- `pdf_bytes` is always `None`.
- `app/views/report.py` provides no PDF button or PDF fallback path.

This conflicts with the technical documentation’s description of optional PDF output and its statement that the UI handles unavailable ReportLab gracefully. The current report implementation is Markdown-only.

### Report content limitations

Recommendation, training, evaluation, and primary permutation-importance explainability information can reach the Markdown report. Optional SHAP does not reach it through the current state/result contract. Also, the report generator falls back to the first-ranked recommendation because the recommendation result does not carry the UI’s separately selected pipeline.

## 9. Testing Reality Check

Current run: **159 passed, 1 warning, 0 failures, 0 skipped**.

### Coverage by type

- Unit tests: the majority of tests exercise individual profilers, task detection, recommendation helpers, pipeline construction, trainer, evaluator, explainability, report generation, state helpers, and formatting.
- Integration-style tests: several tests connect multiple backend modules, for example recommendation -> pipeline builder -> trainer -> evaluator or explainability. They are useful multi-module tests but are not full browser/application integration tests.
- Genuine automated end-to-end tests: none identified. No test drives the Streamlit application from upload through report using a browser or full session interaction.
- Manual application validation: the repository review documentation mentions starting Streamlit, but the current test suite does not automate a full UI workflow.

### Direct test coverage requested in the audit

- Recommendation ranking/Top 5: yes. `tests/test_recommendation_engine.py` asserts five recommendations, ranks 1 through 5, descending confidence, and priority ordering for classification and regression.
- SMOTE: yes. `tests/test_pipeline_builder.py` directly checks the imbalanced-learn pipeline and SMOTE step; `tests/test_model_trainer.py` checks SMOTE training and regression rejection.
- Training/CV: yes. `tests/test_model_trainer.py` covers successful training, CV scores, mean/std scores, no-CV mode, small-data behavior, missing targets, and training time.
- Evaluation: yes. `tests/test_model_evaluator.py` covers classification/regression success, missing targets, invalid inputs, and result structures.
- Permutation importance: yes. `tests/test_explainability.py` checks the method name, successful classification/regression runs, output mapping, top features, and missing-target handling.
- Report generation: yes. `tests/test_report_generator.py` covers successful Markdown generation, missing target handling, regression/classification output, missing values, invalid inputs, and result fields. It does not test actual PDF generation because no PDF path exists.

The tests are strong at backend unit and composed-function level, but they do not prove a complete Streamlit click-path end to end.

## 10. Thesis Claim Verification Table

| Thesis claim | Status | Code evidence / note |
| --- | --- | --- |
| 1. The system profiles structured CSV datasets. | VERIFIED | `app/views/upload_dataset.py`, `src/data_loader.py`, `src/data_profiler.py`. |
| 2. The system detects classification or regression. | VERIFIED | `src/task_detector.py::TaskDetector.detect()`. |
| 3. The system uses a knowledge-based/rule-based recommendation engine. | VERIFIED | `src/knowledge_base.py` loads JSON; `src/recommendation_engine.py` matches rules and archetypes. |
| 4. Dataset characteristics influence recommendations. | VERIFIED | Profile/task contexts feed rules, preprocessing strategies, archetype fit, explanations, and evidence. |
| 5. The system produces ranked Top-5 pipeline recommendations. | VERIFIED | `TOP_PIPELINE_COUNT = 5`; ranking and tests confirm five for current task archetypes. |
| 6. The system explains why pipelines are recommended. | VERIFIED | Recommendation page and result explanation/evidence fields. |
| 7. Recommendations can be converted into executable ML pipelines. | VERIFIED | `src/pipeline_builder.py::PipelineBuilder.build()`. |
| 8. Missing values are automatically handled during model training. | VERIFIED | Median/most-frequent imputers are placed in the fitted preprocessing pipeline. Features above 50% missingness are excluded by the builder. |
| 9. Categorical features are automatically encoded. | VERIFIED | `OneHotEncoder(handle_unknown="ignore")` in the categorical transformer. |
| 10. Scaling is automatically applied when required by the selected pipeline. | VERIFIED | StandardScaler is inserted for scale-sensitive recommendation strategies; tree strategies make scaling optional. MinMax/Robust are builder-supported but not normal current recommendations. |
| 11. SMOTE is automatically applied when the selected recommendation requires it. | VERIFIED | Strong classification imbalance triggers SMOTE strategy; builder inserts SMOTE in an `imblearn.pipeline.Pipeline`. |
| 12. Feature selection is automatically executed. | INACCURATE | Feature selection is represented in rules/recommendation text, but no selector transformer is built. |
| 13. PCA is automatically executed when recommended. | VERIFIED | Builder adds PCA when the dimensionality strategy contains PCA. |
| 14. Models are trained using a held-out train/test split. | VERIFIED | Trainer uses an 80/20 split and downstream evaluator/explainability reuse stored hold-out data. |
| 15. Cross-validation is supported. | VERIFIED | StratifiedKFold/KFold with safe fold reduction and optional disable flag. |
| 16. Classification and regression metrics are calculated. | VERIFIED | Accuracy, precision, recall, F1, confusion matrix, optional binary ROC-AUC, and regression MAE/MSE/RMSE/R² are calculated across backend/UI paths. |
| 17. Model-level explainability uses permutation importance. | VERIFIED | `sklearn.inspection.permutation_importance` on sampled hold-out data. |
| 18. SHAP is implemented. | PARTIALLY VERIFIED | Optional SHAP calculation and UI exist, but it is not core automatic explainability and is not correctly propagated into the report. |
| 19. Recommendation-level and model-level explainability are both provided. | VERIFIED | Rule/evidence explanations in Recommendations; permutation importance in Explainability. |
| 20. Markdown and PDF reporting are supported. | INACCURATE | Markdown is supported; current report generation is Markdown-only and PDF bytes are always `None`. |

## 11. Documentation Discrepancy Check

Only significant discrepancies are listed here:

1. `PROJECT_TECHNICAL_DOCUMENTATION.md` describes optional PDF rendering through ReportLab and graceful PDF fallback. The current code ignores `include_pdf`, has no ReportLab dependency, and exposes Markdown download only.
2. `PROJECT_TECHNICAL_DOCUMENTATION.md` says the system does not implement SHAP-based explanations in the main workflow. The current Explainability page does provide an optional user-triggered SHAP path. The more accurate statement is that SHAP is optional and not part of the automatic core run.
3. `PROJECT_TECHNICAL_DOCUMENTATION.md` presents feature selection as a builder-supported preprocessing behavior. The current builder executes PCA but does not instantiate a feature-selection transformer; feature selection is recommendation-level text only.
4. `PROJECT_TECHNICAL_DOCUMENTATION.md` lists the full builder model-family inventory without clearly separating builder support from normal recommendation reachability. Several listed models are manually builder-supported but not selected by current archetype logic.
5. `PROJECT_REVIEW_REPORT.md` states that the full suite passed with 131 tests. The current repository contains and passes 159 tests, so that count is outdated.
6. `PROJECT_REVIEW_REPORT.md` describes PDF as a future/core-requirement consideration, which is consistent with current code only if PDF is treated as unavailable. It should not be read as evidence that PDF currently works.

## MUST FIX BEFORE DEVELOPMENT FREEZE

None identified as a blocking workflow, leakage, or thesis-scope defect.

The SMOTE path is correctly structured: it is inserted into an imbalanced-learn pipeline, fitted only after the hold-out split, and applied within CV training folds. The core CSV -> profile -> task -> recommendation -> build -> train -> evaluate -> permutation-explainability -> Markdown-report path is executable and covered by backend tests.

The report’s selected-pipeline mismatch and the SHAP-to-report gap are real implementation limitations, but they are not necessarily freeze blockers if the thesis explicitly claims Markdown-only reporting and describes the report as summarizing the top-ranked recommendation rather than always the user’s selected candidate. If the thesis promises exact selected-pipeline reporting or SHAP report inclusion, those claims must be adjusted or the issues must be fixed before submission.

## THESIS CLAIMS THAT MUST BE ADJUSTED

- State that reporting currently supports Markdown only; do not claim working PDF export.
- Describe SHAP as optional, manually triggered, tree-model-only functionality exposed in the Explainability page, not as the core automatic explainability method and not as a report output.
- Distinguish models implemented by the pipeline builder from models currently reachable through knowledge-base recommendations.
- Do not claim automatic feature selection. The current system can recommend feature selection textually, but it does not execute a selector.
- Describe MinMaxScaler and RobustScaler as builder-supported possibilities rather than normal recommendation outcomes unless the thesis demonstrates direct construction of those strategies.
- Treat quality labels such as excellent/good/acceptable/weak as project-defined heuristics, not validated academic quality standards.
- Describe the test suite as backend unit/composed-module coverage, not as a genuine automated end-to-end Streamlit test suite.

## OPTIONAL / FUTURE WORK

- Correctly carry the user-selected pipeline into `ReportResult` and report generation.
- Propagate optional SHAP results into report generation if SHAP reporting becomes a stated requirement.
- Add a true browser/session-level end-to-end test for the complete Streamlit workflow.
- Implement an actual feature-selection transformer only if it becomes part of the thesis scope.
- Add PDF generation only if PDF is required as a final deliverable, with an explicit dependency and dedicated tests.
- Consider additional XAI methods such as LIME, PDP, ICE, or counterfactual explanations only as future work; their absence does not undermine the current bounded scope.

## FINAL VERDICT

**Can the current system reasonably be considered feature-complete for the stated Bachelor's thesis scope: YES.**

The system has an executable, modular, rule-based recommendation workflow for structured CSV data, real preprocessing and model training, held-out evaluation, recommendation-level explanations, permutation-based model explainability, and Markdown reporting. The implementation is supported by 159 passing automated tests. The thesis must describe PDF, SHAP, feature selection, model reachability, and test scope accurately, but no genuine leakage or core workflow failure was identified that requires further development before freezing the stated prototype scope.