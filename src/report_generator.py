"""Final workflow report generation utilities.

This module combines the completed workflow results into a thesis-style report
using a shared report data model. Markdown is preserved for debugging and
version control, while PDF is the primary end-user export.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import logging
from typing import Any

import pandas as pd
from sklearn.metrics import classification_report as sk_classification_report, confusion_matrix as sk_confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from src.missing_value_analysis import build_missing_value_overview
from src.display_formatting import format_confidence, format_count, format_cv_score, format_metric, format_percentage, format_seconds

logger = logging.getLogger(__name__)

APP_VERSION = "v1.8.0"


@dataclass(slots=True)
class ReportArtifact:
    """Container for a generated report artifact."""

    content: str | bytes | None
    mime_type: str
    file_name: str


@dataclass(slots=True)
class ReportData:
    """Shared report data used by both Markdown and PDF generation."""

    title: str
    application_version: str
    generated_at: str
    dataset_name: str
    learning_task: str
    target_variable: str
    dataset_summary: OrderedDict[str, str]
    processing_summary: list[tuple[str, str]]
    task_detection: OrderedDict[str, str]
    selected_pipeline: OrderedDict[str, str]
    training_summary: list[tuple[str, str]]
    evaluation_summary: OrderedDict[str, str]
    explainability_summary: OrderedDict[str, str]
    overall_summary: str
    classification_report: pd.DataFrame | None
    confusion_matrix: pd.DataFrame | None
    feature_importance_chart: pd.DataFrame | None
    shap_chart: pd.DataFrame | None
    report_sections: dict[str, str]


@dataclass(slots=True)
class ReportResult:
    """Structured result of report generation."""

    success: bool
    markdown_report: str | None
    pdf_bytes: bytes | None
    report_sections: dict[str, str]
    messages: list[str]
    warnings: list[str]
    errors: list[str]



def generate_report(
    profile_result: Any,
    task_result: Any,
    recommendation_result: Any,
    training_result: Any,
    evaluation_result: Any,
    explainability_result: Any | None = None,
    include_pdf: bool = False,
    dataset_name: str | None = None,
) -> ReportResult:
    """Generate a thesis-style report from workflow results.

    Args:
        profile_result: Output of the dataset profiling step.
        task_result: Output of the task detection step.
        recommendation_result: Output of the recommendation step.
        training_result: Output of the model training step.
        evaluation_result: Output of the model evaluation step.
        explainability_result: Optional output of the explainability step.
        include_pdf: Preserved for backward compatibility and ignored.
        dataset_name: Optional uploaded dataset name.

    Returns:
        A structured ReportResult containing Markdown and optional PDF output.
    """

    required_errors = _validate_required_inputs(
        profile_result,
        task_result,
        recommendation_result,
        training_result,
        evaluation_result,
    )
    if required_errors:
        return _failure_result(required_errors)

    report_data = _build_report_data(
        profile_result=profile_result,
        task_result=task_result,
        recommendation_result=recommendation_result,
        training_result=training_result,
        evaluation_result=evaluation_result,
        explainability_result=explainability_result,
        dataset_name=dataset_name,
    )
    markdown_report = _build_markdown_report(report_data)

    warnings: list[str] = []
    if explainability_result is None:
        warnings.append("Explainability result was not available, so the report uses a brief placeholder section.")
    elif not getattr(explainability_result, "success", False):
        warnings.append("Explainability result was not successful, so the report uses a brief placeholder section.")

    pdf_bytes: bytes | None = None
    messages = ["Report generated successfully."]

    return ReportResult(
        success=True,
        markdown_report=markdown_report,
        pdf_bytes=pdf_bytes,
        report_sections=report_data.report_sections,
        messages=messages,
        warnings=warnings,
        errors=[],
    )



def _validate_required_inputs(
    profile_result: Any,
    task_result: Any,
    recommendation_result: Any,
    training_result: Any,
    evaluation_result: Any,
) -> list[str]:
    """Validate required workflow results before report generation."""

    errors: list[str] = []
    checks = [
        (profile_result, "profile_result"),
        (task_result, "task_result"),
        (recommendation_result, "recommendation_result"),
        (training_result, "training_result"),
        (evaluation_result, "evaluation_result"),
    ]

    for value, name in checks:
        if value is None:
            errors.append(f"{name} is required.")
        elif not getattr(value, "success", False):
            errors.append(f"{name} must be successful before generating the report.")

    return errors



def _build_report_data(
    *,
    profile_result: Any,
    task_result: Any,
    recommendation_result: Any,
    training_result: Any,
    evaluation_result: Any,
    explainability_result: Any | None,
    dataset_name: str | None,
) -> ReportData:
    """Build the shared report data model."""

    target_variable = _resolve_target_column(
        training_result=training_result,
        task_result=task_result,
        profile_result=profile_result,
    )
    dataset_name_value = dataset_name or _resolve_dataset_name(profile_result)
    task_type = str(getattr(task_result, "task_type", "N/A") or "N/A")

    dataset_summary = OrderedDict(
        [
            ("Number of rows", str(getattr(profile_result, "row_count", "N/A"))),
            ("Number of columns", str(getattr(profile_result, "column_count", "N/A"))),
            ("Numerical features", ", ".join(getattr(profile_result, "numeric_columns", []) or []) or "None"),
            ("Categorical features", ", ".join(getattr(profile_result, "categorical_columns", []) or []) or "None"),
            ("Missing values", str(sum(getattr(profile_result, "missing_values", {}).values()))),
            ("Duplicate rows", str(getattr(profile_result, "duplicate_row_count", "N/A"))),
            ("Dataset quality rating", _dataset_quality_rating(profile_result)),
        ]
    )

    processing_summary = _build_processing_summary(profile_result, training_result)
    task_detection = OrderedDict(
        [
            ("Target column", target_variable),
            ("Detected learning task", task_type),
            ("Confidence", format_confidence(getattr(task_result, "confidence", None))),
            ("Detection reason", str(getattr(task_result, "detection_reason", "N/A") or "N/A")),
        ]
    )

    selected_pipeline = _build_selected_pipeline_summary(recommendation_result)
    training_summary = _build_training_summary(training_result)
    report_evaluation = _build_report_evaluation(
        profile_result=profile_result,
        training_result=training_result,
        evaluation_result=evaluation_result,
    )
    evaluation_summary = _build_evaluation_summary(evaluation_result, report_evaluation)
    explainability_summary = _build_explainability_summary(explainability_result)
    overall_summary = _build_overall_summary(
        profile_result=profile_result,
        task_result=task_result,
        recommendation_result=recommendation_result,
        training_result=training_result,
        evaluation_result=evaluation_result,
    )
    missing_value_handling = _build_missing_value_handling_summary(profile_result, training_result)

    report_sections = OrderedDict(
        [
            ("Cover Information", _build_cover_section(dataset_name_value, task_type, target_variable)),
            ("Dataset Summary", _format_key_value_lines(dataset_summary)),
            ("Dataset Processing Summary", _format_key_value_lines(OrderedDict(processing_summary))),
            ("Task Detection Summary", _build_task_summary(task_result, target_variable)),
            ("Task Detection", _build_task_summary(task_result, target_variable)),
            ("Selected Pipeline", _format_key_value_lines(OrderedDict(selected_pipeline))),
            ("Training Summary", _format_key_value_lines(OrderedDict(training_summary))),
            ("Evaluation Results", _format_key_value_lines(OrderedDict(evaluation_summary))),
            ("Explainability Summary", _format_key_value_lines(OrderedDict(explainability_summary))),
            ("Missing Value Handling", missing_value_handling),
            ("Overall Summary", overall_summary),
            ("Conclusion", _build_conclusion_summary(evaluation_result)),
        ]
    )

    classification_report = report_evaluation.get("classification_report")
    confusion_matrix = report_evaluation.get("confusion_matrix")

    feature_importance_chart = _build_feature_importance_table(explainability_result)
    shap_chart = _build_shap_table(explainability_result)

    return ReportData(
        title="Explainable Machine Learning Pipeline Recommendation Report",
        application_version=APP_VERSION,
        generated_at=_current_timestamp(),
        dataset_name=dataset_name_value,
        learning_task=task_type,
        target_variable=target_variable,
        dataset_summary=dataset_summary,
        processing_summary=processing_summary,
        task_detection=task_detection,
        selected_pipeline=selected_pipeline,
        training_summary=training_summary,
        evaluation_summary=evaluation_summary,
        explainability_summary=explainability_summary,
        overall_summary=overall_summary,
        classification_report=classification_report,
        confusion_matrix=confusion_matrix,
        feature_importance_chart=feature_importance_chart,
        shap_chart=shap_chart,
        report_sections=report_sections,
    )



def _build_markdown_report(report_data: ReportData) -> str:
    """Combine report sections into a Markdown document."""

    lines = [
        f"# {report_data.title}",
        "",
        f"Application Version: {report_data.application_version}",
        f"Generated: {report_data.generated_at}",
        f"Dataset Name: {report_data.dataset_name}",
        f"Learning Task: {report_data.learning_task}",
        f"Target Variable: {report_data.target_variable}",
        "",
    ]

    for section_name, section_body in report_data.report_sections.items():
        lines.append(f"## {section_name}")
        lines.append(section_body)
        lines.append("")

    if report_data.classification_report is not None:
        lines.append("## Classification Report")
        lines.append(report_data.classification_report.to_markdown(index=False))
        lines.append("")

    if report_data.confusion_matrix is not None:
        lines.append("## Confusion Matrix")
        lines.append(report_data.confusion_matrix.to_markdown())
        lines.append("")

    return "\n".join(lines).strip()



def _build_cover_section(dataset_name: str, task_type: str, target_variable: str) -> str:
    """Build the cover information section."""

    return _format_key_value_lines(
        OrderedDict(
            [
                ("Application Version", APP_VERSION),
                ("Generated Date and Time", _current_timestamp()),
                ("Dataset Name", dataset_name),
                ("Learning Task", task_type),
                ("Target Variable", target_variable),
            ]
        )
    )



def _build_processing_summary(profile_result: Any, training_result: Any) -> list[tuple[str, str]]:
    """Summarize the preprocessing that was actually applied."""

    overview = build_missing_value_overview(profile_result)
    rows: list[tuple[str, str]] = []

    if overview["has_missing_values"]:
        rows.append(("Missing value handling", overview["strategy_summary"]))
        if overview["drop_feature_columns"]:
            rows.append(("Dropped features", ", ".join(overview["drop_feature_columns"])))
        if overview["imputed_feature_columns"]:
            rows.append(("Encoded / imputed features", ", ".join(overview["imputed_feature_columns"])))
    else:
        rows.append(("Missing value handling", "No missing value handling was required."))

    preprocessing_steps = getattr(training_result, "trained_pipeline", None)
    model_steps = getattr(preprocessing_steps, "steps", []) or []
    preprocessing_summary = getattr(training_result, "warnings", []) or []
    if preprocessing_summary:
        rows.append(("Applied preprocessing notes", "; ".join(preprocessing_summary)))

    pipeline_result = getattr(training_result, "trained_pipeline", None)
    if pipeline_result is not None:
        rows.append(("Train/Test split", "80% / 20% held-out split"))

    rows.append(("Feature scaling", "Applied when selected by the recommendation"))
    rows.append(("SMOTE", "Applied only when the selected classification pipeline required imbalance handling"))

    if model_steps:
        rows.append(("Pipeline steps", ", ".join(step_name for step_name, _ in model_steps)))

    return rows



def _build_selected_pipeline_summary(recommendation_result: Any) -> OrderedDict[str, str]:
    """Summarize the selected pipeline."""

    selected_pipeline = getattr(recommendation_result, "selected_pipeline", None) or getattr(recommendation_result, "top_pipelines", [None])[0]
    if selected_pipeline is None:
        return OrderedDict([("Pipeline name", "Not available")])

    preprocessing_steps = getattr(selected_pipeline, "preprocessing_steps", []) or []
    return OrderedDict(
        [
            ("Pipeline name", getattr(selected_pipeline, "pipeline_name", "N/A")),
            ("Machine learning algorithm", _format_model_name(getattr(selected_pipeline, "recommended_model_family", ""))),
            ("Preprocessing pipeline", ", ".join(preprocessing_steps) or "Not specified"),
            ("Why recommended", getattr(selected_pipeline, "explanation", "N/A")),
            ("Advantages", ", ".join(getattr(selected_pipeline, "advantages", []) or []) or "Not specified"),
            ("Limitations", ", ".join(getattr(selected_pipeline, "limitations", []) or []) or "Not specified"),
        ]
    )


def _build_task_summary(task_result: Any, target_column: str) -> str:
    """Summarize task detection output using the legacy plain-text format."""

    return (
        f"Target column: {target_column}\n\n"
        f"Detected learning task: {getattr(task_result, 'task_type', 'N/A')}\n\n"
        f"Confidence: {getattr(task_result, 'confidence', 'N/A')}\n\n"
        f"Detection reason: {getattr(task_result, 'detection_reason', 'N/A')}"
    )



def _build_training_summary(training_result: Any) -> list[tuple[str, str]]:
    """Summarize the training result in a table-friendly format."""

    rows: list[tuple[str, str]] = []
    scores = getattr(training_result, "cross_validation_scores", []) or []
    rows.append(("Cross-validation score(s)", ", ".join(format_cv_score(score) for score in scores) if scores else "Not available"))
    rows.append(("Mean CV score", format_cv_score(getattr(training_result, "mean_cv_score", None))))
    rows.append(("Standard deviation", format_cv_score(getattr(training_result, "std_cv_score", None))))
    rows.append(("Training time", _format_seconds(getattr(training_result, "training_time_seconds", None))))
    rows.append(("Selected hyperparameters", _selected_hyperparameters_text(training_result)))
    return rows



def _build_evaluation_summary(evaluation_result: Any, report_evaluation: dict[str, Any]) -> OrderedDict[str, str]:
    """Summarize evaluation results."""

    summary = OrderedDict()
    metrics = getattr(evaluation_result, "classification_metrics", None)
    if metrics is not None:
        summary["Accuracy"] = format_metric(metrics.get("accuracy"))
        summary["Precision"] = format_metric(metrics.get("precision_weighted"))
        summary["Recall"] = format_metric(metrics.get("recall_weighted"))
        summary["F1-score"] = format_metric(metrics.get("f1_weighted"))
    else:
        regression_metrics = getattr(evaluation_result, "regression_metrics", {}) or {}
        summary["MAE"] = format_metric(regression_metrics.get("mae"))
        summary["MSE"] = format_metric(regression_metrics.get("mse"))
        summary["RMSE"] = format_metric(regression_metrics.get("rmse"))
        summary["R² Score"] = format_metric(regression_metrics.get("r2"))

    summary["Quality rating"] = str(getattr(evaluation_result, "quality_rating", "N/A") or "N/A")
    summary["Quality summary"] = str(getattr(evaluation_result, "quality_summary", "N/A") or "N/A")
    if report_evaluation.get("main_metric") is not None:
        summary["Main evaluation score"] = format_metric(report_evaluation.get("main_metric"))
    return summary



def _build_explainability_summary(explainability_result: Any | None) -> OrderedDict[str, str]:
    """Summarize explainability output."""

    summary = OrderedDict()
    if explainability_result is None:
        summary["Pipeline explanation"] = "Explainability results were not generated for this run."
        summary["Feature importance"] = "Not available"
        summary["Top important features"] = "Not available"
        return summary

    if not getattr(explainability_result, "success", False):
        summary["Pipeline explanation"] = "Explainability was requested, but the result was not successful."
        summary["Feature importance"] = "Not available"
        summary["Top important features"] = "Not available"
        return summary

    top_features = getattr(explainability_result, "top_features", []) or []
    summary["Pipeline explanation"] = str(getattr(explainability_result, "explanation_summary", "N/A") or "N/A")
    summary["Feature importance"] = _feature_importance_text(top_features)
    summary["Top important features"] = ", ".join(feature for feature, _ in top_features[:5]) or "Not available"
    return summary


def _build_missing_value_handling_summary(profile_result: Any, training_result: Any) -> str:
    """Summarize missing-value handling for the report."""

    overview = build_missing_value_overview(profile_result)
    train_rows = int(getattr(training_result, "train_row_count", 0) or 0)
    test_rows = int(getattr(training_result, "test_row_count", 0) or 0)
    final_training_size = train_rows + test_rows
    original_rows = int(getattr(profile_result, "row_count", 0) or 0)
    rows_removed_missing_target = max(original_rows - final_training_size, 0)

    if not overview["has_missing_values"]:
        return (
            "No missing values were detected. No missing value handling was required.\n\n"
            f"Rows removed because of missing target values: {format_count(rows_removed_missing_target)}\n\n"
            f"Final dataset size used for training: {format_count(final_training_size)}"
        )

    analysis_frame = overview["analysis_frame"]
    lines = [
        "Missing values were detected in the dataset.",
        "",
        f"Detected missing values: {format_count(overview['total_missing_values'])}",
        f"Affected columns: {', '.join(overview['columns']) if overview['columns'] else 'Not specified'}",
        f"Features recommended for dropping: {', '.join(overview['drop_feature_columns']) if overview['drop_feature_columns'] else 'None'}",
        f"Features imputed: {', '.join(overview['imputed_feature_columns']) if overview['imputed_feature_columns'] else 'None'}",
        f"Rows removed because of missing target values: {format_count(rows_removed_missing_target)}",
        f"Final dataset size used for training: {format_count(final_training_size)}",
        "",
        "Missing percentages and actions:",
    ]
    for _, row in analysis_frame.iterrows():
        reason = row.get("Reason", "")
        reason_text = f" ({reason})" if reason else ""
        lines.append(
            f"- {row['Column']}: {format_percentage(row['Missing percentage'])} missing, recommended action {row['Recommended action']}, "
            f"applied action {row['Applied action during preprocessing']}{reason_text}"
        )

    lines.extend(
        [
            "",
            f"Recommended strategy: {overview['strategy_summary']}",
            f"Explanation: {overview['explanation']}",
        ]
    )
    return "\n".join(lines)


def _build_conclusion_summary(evaluation_result: Any) -> str:
    """Create a short conclusion based on the completed workflow."""

    return (
        f"The workflow completed with a {getattr(evaluation_result, 'quality_rating', 'N/A')} model result. "
        "The final report combines the dataset profile, task detection, pipeline recommendation, training, evaluation, and explainability outputs into one thesis-friendly summary."
    )


def _build_report_evaluation(profile_result: Any, training_result: Any, evaluation_result: Any) -> dict[str, Any]:
    """Reconstruct evaluation details for the report without changing evaluation logic."""

    task_type = getattr(evaluation_result, "task_type", None) or getattr(training_result, "task_type", None)
    train_features = getattr(training_result, "train_features", None)
    test_features = getattr(training_result, "test_features", None)
    train_target = getattr(training_result, "train_target", None)
    test_target = getattr(training_result, "test_target", None)
    if any(value is None for value in (train_features, test_features, train_target, test_target)):
        raise ValueError("Training result does not contain the cleaned hold-out split required for report generation.")
    if len(test_features) != len(test_target):
        raise ValueError("Stored hold-out features and target values must have matching lengths.")

    pipeline = getattr(training_result, "trained_pipeline", None)
    if pipeline is None:
        return {"main_metric": None, "classification_report": None, "confusion_matrix": None}

    predictions = pipeline.predict(test_features)
    if task_type == "classification":
        y_true_series = pd.Series(test_target).reset_index(drop=True)
        y_pred_series = pd.Series(predictions).reset_index(drop=True)
        valid_mask = y_true_series.notna() & y_pred_series.notna()
        y_true_series = y_true_series.loc[valid_mask].reset_index(drop=True)
        y_pred_series = y_pred_series.loc[valid_mask].reset_index(drop=True)
        labels = sorted(pd.unique(pd.concat([y_true_series, y_pred_series], ignore_index=True).dropna()).tolist(), key=lambda value: str(value))
        report = pd.DataFrame(sk_classification_report(y_true_series, y_pred_series, output_dict=True, zero_division=0)).transpose().reset_index()
        report.rename(columns={"index": "Label"}, inplace=True)
        matrix = pd.DataFrame(sk_confusion_matrix(y_true_series, y_pred_series, labels=labels), index=[str(label) for label in labels], columns=[str(label) for label in labels])
        main_metric = getattr(evaluation_result, "classification_metrics", {}) or {}
        return {
            "main_metric": main_metric.get("accuracy"),
            "classification_report": report,
            "confusion_matrix": matrix,
        }

    y_true = pd.to_numeric(pd.Series(test_target), errors="coerce")
    y_pred = pd.to_numeric(pd.Series(predictions), errors="coerce")
    valid_mask = y_true.notna() & y_pred.notna()
    y_true = y_true.loc[valid_mask].reset_index(drop=True)
    y_pred = y_pred.loc[valid_mask].reset_index(drop=True)
    mse = float(mean_squared_error(y_true, y_pred))
    regression_metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": float(mse ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }
    return {
        "main_metric": regression_metrics.get("r2"),
        "classification_report": None,
        "confusion_matrix": None,
    }



def _build_overall_summary(
    *,
    profile_result: Any,
    task_result: Any,
    recommendation_result: Any,
    training_result: Any,
    evaluation_result: Any,
) -> str:
    """Generate a concise overall summary paragraph."""

    task_type = str(getattr(task_result, "task_type", "unknown") or "unknown")
    selected_pipeline = getattr(recommendation_result, "selected_pipeline", None) or getattr(recommendation_result, "top_pipelines", [None])[0]
    pipeline_name = getattr(selected_pipeline, "pipeline_name", "the selected pipeline") if selected_pipeline is not None else "the selected pipeline"
    model_name = _format_model_name(getattr(selected_pipeline, "recommended_model_family", "")) if selected_pipeline is not None else "the trained model"
    eval_summary = getattr(evaluation_result, "quality_summary", "") or ""
    top_features = getattr(getattr(training_result, "trained_pipeline", None), "steps", [])

    if task_type == "regression":
        performance_phrase = f"The trained model achieved {eval_summary.lower()}" if eval_summary else "The trained model was evaluated on the held-out test set"
    else:
        performance_phrase = f"The trained model achieved {eval_summary.lower()}" if eval_summary else "The trained model was evaluated on the held-out test set"

    return (
        f"The uploaded dataset was identified as a {task_type} problem. Based on the analysed dataset characteristics, {model_name} was recommended through {pipeline_name}. "
        f"{performance_phrase}. Feature importance analysis identified the variables contributing most strongly to the predictions."
    )



def _build_classification_report_table(evaluation_result: Any) -> pd.DataFrame | None:
    """Build a compact classification report table when available."""

    report = getattr(evaluation_result, "classification_report", None)
    if report is None:
        return None
    return pd.DataFrame(report)



def _build_confusion_matrix_table(evaluation_result: Any) -> pd.DataFrame | None:
    """Build a confusion matrix table when available."""

    matrix = getattr(evaluation_result, "confusion_matrix", None)
    if matrix is None:
        return None
    return pd.DataFrame(matrix)



def _build_feature_importance_table(explainability_result: Any | None) -> pd.DataFrame | None:
    """Build a feature importance chart table when available."""

    if explainability_result is None or not getattr(explainability_result, "success", False):
        return None
    top_features = getattr(explainability_result, "top_features", []) or []
    if not top_features:
        return None
    return pd.DataFrame(top_features, columns=["Feature", "Importance"])



def _build_shap_table(explainability_result: Any | None) -> pd.DataFrame | None:
    """Build a SHAP chart table when available."""

    shap_result = getattr(explainability_result, "shap_result", None)
    if shap_result is None or not getattr(shap_result, "success", False):
        return None
    top_features = getattr(shap_result, "top_features", []) or []
    if not top_features:
        return None
    return pd.DataFrame(top_features, columns=["Feature", "SHAP Importance"])



def _selected_hyperparameters_text(training_result: Any) -> str:
    """Extract a readable description of selected hyperparameters."""

    trained_pipeline = getattr(training_result, "trained_pipeline", None)
    pipeline_steps = getattr(trained_pipeline, "steps", []) or []
    if not pipeline_steps:
        return "Not available"

    model = pipeline_steps[-1][1]
    params = model.get_params(deep=False) if hasattr(model, "get_params") else {}
    keys_of_interest = ["C", "penalty", "solver", "max_depth", "n_estimators", "alpha", "kernel", "n_neighbors", "gamma", "max_iter"]
    items = []
    for key in keys_of_interest:
        if key in params:
            items.append(f"{key}={params[key]}")
    return ", ".join(items) if items else "Not available"



def _feature_importance_text(top_features: list[tuple[str, float]]) -> str:
    """Return a concise description of top features."""

    if not top_features:
        return "Feature importance was not available."
    top_names = ", ".join(feature for feature, _ in top_features[:3])
    return f"The most influential features were {top_names}."


def _format_model_name(model_family: str) -> str:
    """Convert a model-family identifier into a readable model name."""

    normalized = model_family.replace("_", " ").strip().lower()
    mapping = {
        "logistic regression": "Logistic Regression",
        "random forest": "Random Forest",
        "decision tree": "Decision Tree",
        "knn": "KNN",
        "naive bayes": "Naive Bayes",
        "svm": "SVM",
        "linear regression": "Linear Regression",
        "ridge": "Ridge",
        "lasso": "Lasso",
        "polynomial regression": "Polynomial Regression",
        "random forest regressor": "Random Forest Regressor",
        "logisticregression": "Logistic Regression",
        "randomforest": "Random Forest",
        "decisiontree": "Decision Tree",
        "kneighborsclassifier": "KNN",
        "gaussiannb": "Naive Bayes",
        "linearregression": "Linear Regression",
        "randomforestregressor": "Random Forest Regressor",
        "decisiontreeregressor": "Decision Tree Regressor",
        "xgbclassifier": "XGBoost",
        "xgbregressor": "XGBoost",
    }
    return mapping.get(normalized, model_family.replace("_", " ").title())



def _format_key_value_lines(values: OrderedDict[str, str]) -> str:
    """Format key-value pairs as readable Markdown text."""

    return "\n\n".join(f"**{key}:** {value}" for key, value in values.items())



def _format_numeric(value: Any, *, digits: int = 3) -> str:
    """Format a numeric value for report display."""

    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return "N/A"



def _format_seconds(value: Any) -> str:
    """Format seconds as a readable string."""

    if isinstance(value, (int, float)):
        return format_seconds(value)
    return "N/A"



def _dataset_quality_rating(profile_result: Any) -> str:
    """Derive a lightweight quality rating from the profiling result."""

    missing_values = sum(getattr(profile_result, "missing_values", {}).values())
    duplicates = int(getattr(profile_result, "duplicate_row_count", 0) or 0)
    rows = int(getattr(profile_result, "row_count", 0) or 0)
    if rows <= 0:
        return "Not available"
    missing_ratio = missing_values / max(rows * int(getattr(profile_result, "column_count", 0) or 1), 1)
    duplicate_ratio = duplicates / rows
    if missing_ratio < 0.05 and duplicate_ratio < 0.02:
        return "Good"
    if missing_ratio < 0.15 and duplicate_ratio < 0.05:
        return "Fair"
    return "Needs review"



def _resolve_target_column(
    *,
    training_result: Any | None = None,
    task_result: Any | None = None,
    profile_result: Any | None = None,
) -> str:
    """Resolve the target column using the configured fallback order."""

    for result in (training_result, task_result, profile_result):
        if result is None:
            continue
        target_column = getattr(result, "target_column", None)
        if target_column:
            return str(target_column)
    return "Not specified"



def _resolve_dataset_name(profile_result: Any) -> str:
    """Resolve a human-friendly dataset name."""

    source_name = getattr(profile_result, "source_name", None)
    if source_name:
        return str(source_name)
    dataframe = getattr(profile_result, "dataframe", None)
    if dataframe is not None:
        return "Uploaded dataset"
    return "Not specified"



def _current_timestamp() -> str:
    """Return the current timestamp as a formatted string."""

    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")



def _markdown_to_plain_text(text: str) -> str:
    """Convert a small subset of Markdown to plain text for PDF output."""

    converted = text.replace("**", "")
    converted = converted.replace("- ", "• ")
    return converted



def _failure_result(errors: list[str]) -> ReportResult:
    """Build a failed report result."""

    return ReportResult(
        success=False,
        markdown_report=None,
        pdf_bytes=None,
        report_sections={},
        messages=[],
        warnings=[],
        errors=errors,
    )
