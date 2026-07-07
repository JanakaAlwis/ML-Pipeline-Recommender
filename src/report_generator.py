"""Final workflow report generation utilities.

This module combines the completed workflow results into a thesis-style report
in Markdown form and, when possible, a simple PDF export.
"""

from __future__ import annotations

from dataclasses import dataclass
import io
import logging
from typing import Any

from src.missing_value_analysis import build_missing_value_overview

logger = logging.getLogger(__name__)


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
    include_pdf: bool = True,
) -> ReportResult:
    """Generate a thesis-style report from workflow results.

    Args:
        profile_result: Output of the dataset profiling step.
        task_result: Output of the task detection step.
        recommendation_result: Output of the recommendation step.
        training_result: Output of the model training step.
        evaluation_result: Output of the model evaluation step.
        explainability_result: Optional output of the explainability step.
        include_pdf: Whether to generate PDF bytes when supported.

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

    report_sections = _build_report_sections(
        profile_result=profile_result,
        task_result=task_result,
        recommendation_result=recommendation_result,
        training_result=training_result,
        evaluation_result=evaluation_result,
        explainability_result=explainability_result,
    )
    markdown_report = _build_markdown_report(report_sections)

    warnings: list[str] = []
    if explainability_result is None:
        warnings.append("Explainability result was not available, so the report uses a brief placeholder section.")
    elif not getattr(explainability_result, "success", False):
        warnings.append("Explainability result was not successful, so the report uses a brief placeholder section.")

    pdf_bytes: bytes | None = None
    if include_pdf:
        try:
            pdf_bytes = _build_pdf_bytes(markdown_report)
        except Exception as exc:  # pragma: no cover - optional dependency path
            logger.warning("PDF generation failed: %s", exc)
            warnings.append("PDF export is currently unavailable. You can still download the Markdown report.")

    messages = ["Report generated successfully."]
    if pdf_bytes is not None:
        messages.append("PDF report is ready for download.")

    return ReportResult(
        success=True,
        markdown_report=markdown_report,
        pdf_bytes=pdf_bytes,
        report_sections=report_sections,
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


def _build_report_sections(
    *,
    profile_result: Any,
    task_result: Any,
    recommendation_result: Any,
    training_result: Any,
    evaluation_result: Any,
    explainability_result: Any | None,
) -> dict[str, str]:
    """Build ordered report sections from workflow outputs."""

    sections: dict[str, str] = {}
    sections["Title"] = "Explainable ML Pipeline Recommendation System\nBachelor's Thesis Prototype"
    sections["Dataset Summary"] = _build_dataset_summary(
        profile_result=profile_result,
        task_result=task_result,
        training_result=training_result,
    )
    sections["Task Detection Summary"] = _build_task_summary(task_result)
    sections["Recommended Pipelines"] = _build_recommendation_summary(recommendation_result)
    sections["Training Results"] = _build_training_summary(
        profile_result=profile_result,
        task_result=task_result,
        training_result=training_result,
    )
    sections["Missing Value Handling"] = _build_missing_value_handling_summary(profile_result, training_result)
    sections["Evaluation Results"] = _build_evaluation_summary(evaluation_result)
    sections["Explainability Summary"] = _build_explainability_summary(explainability_result)
    sections["Limitations"] = _build_limitations_summary()
    sections["Conclusion"] = _build_conclusion_summary(evaluation_result)
    return sections


def _build_markdown_report(report_sections: dict[str, str]) -> str:
    """Combine report sections into a Markdown document."""

    lines = ["# Explainable ML Pipeline Recommendation System", "", "Bachelor's Thesis Prototype", ""]
    ordered_sections = [
        "Dataset Summary",
        "Task Detection Summary",
        "Recommended Pipelines",
        "Training Results",
        "Missing Value Handling",
        "Evaluation Results",
        "Explainability Summary",
        "Limitations",
        "Conclusion",
    ]
    for section_name in ordered_sections:
        lines.append(f"## {section_name}")
        lines.append(report_sections[section_name])
        lines.append("")
    return "\n".join(lines).strip()


def _build_dataset_summary(
    *,
    profile_result: Any,
    task_result: Any,
    training_result: Any,
) -> str:
    """Summarize the profiled dataset."""

    numeric_columns = ", ".join(getattr(profile_result, "numeric_columns", []) or []) or "None"
    categorical_columns = ", ".join(getattr(profile_result, "categorical_columns", []) or []) or "None"
    boolean_columns = ", ".join(getattr(profile_result, "boolean_columns", []) or []) or "None"
    missing_count = sum(getattr(profile_result, "missing_values", {}).values())
    target_column = _resolve_target_column(
        training_result=training_result,
        task_result=task_result,
        profile_result=profile_result,
    )
    return (
        f"Rows: {getattr(profile_result, 'row_count', 'N/A')}\n\n"
        f"Columns: {getattr(profile_result, 'column_count', 'N/A')}\n\n"
        f"Target column: {target_column}\n\n"
        f"Numeric features: {numeric_columns}\n\n"
        f"Categorical features: {categorical_columns}\n\n"
        f"Boolean features: {boolean_columns}\n\n"
        f"Missing values: {missing_count}\n\n"
        f"Duplicate rows: {getattr(profile_result, 'duplicate_row_count', 'N/A')}"
    )


def _build_task_summary(task_result: Any) -> str:
    """Summarize task detection output."""

    target_column = _resolve_target_column(task_result=task_result)
    return (
        f"Target column: {target_column}\n\n"
        f"Task type: {getattr(task_result, 'task_type', 'N/A')}\n\n"
        f"Confidence: {getattr(task_result, 'confidence', 'N/A')}\n\n"
        f"Detection reason: {getattr(task_result, 'detection_reason', 'N/A')}\n\n"
        f"Alternative task type: {getattr(task_result, 'alternative_task_type', 'N/A')}"
    )


def _build_recommendation_summary(recommendation_result: Any) -> str:
    """Summarize the recommended pipelines."""

    pipelines = getattr(recommendation_result, "top_pipelines", []) or []
    if not pipelines:
        return "No pipeline recommendations were available."

    lines: list[str] = []
    for pipeline in pipelines:
        lines.append(
            f"- Rank {getattr(pipeline, 'pipeline_rank', 'N/A')}: {getattr(pipeline, 'pipeline_name', 'N/A')} "
            f"({getattr(pipeline, 'recommended_model_family', 'N/A')}, confidence {getattr(pipeline, 'confidence_score', 'N/A')})"
        )
        preprocessing_steps = getattr(pipeline, "preprocessing_steps", []) or []
        if preprocessing_steps:
            lines.append(f"  - Preprocessing: {', '.join(preprocessing_steps)}")
        evaluation_metrics = getattr(pipeline, "evaluation_metrics", []) or []
        if evaluation_metrics:
            lines.append(f"  - Metrics: {', '.join(evaluation_metrics)}")
    return "\n".join(lines)


def _build_training_summary(
    *,
    profile_result: Any,
    task_result: Any,
    training_result: Any,
) -> str:
    """Summarize the training result."""

    scores = getattr(training_result, "cross_validation_scores", []) or []
    score_text = ", ".join(f"{score:.3f}" for score in scores) if scores else "Not available"
    target_column = _resolve_target_column(
        training_result=training_result,
        task_result=task_result,
        profile_result=profile_result,
    )
    return (
        f"Model: {getattr(training_result, 'model_name', 'N/A')}\n\n"
        f"Task type: {getattr(training_result, 'task_type', 'N/A')}\n\n"
        f"Target column: {target_column}\n\n"
        f"Training time (seconds): {getattr(training_result, 'training_time_seconds', 'N/A')}\n\n"
        f"Scoring metric: {getattr(training_result, 'scoring_metric', 'N/A')}\n\n"
        f"Cross-validation scores: {score_text}\n\n"
        f"Mean CV score: {getattr(training_result, 'mean_cv_score', 'N/A')}\n\n"
        f"Std CV score: {getattr(training_result, 'std_cv_score', 'N/A')}"
    )


def _build_missing_value_handling_summary(profile_result: Any, training_result: Any) -> str:
    """Summarize the missing-value handling used for the workflow."""

    overview = build_missing_value_overview(profile_result)
    train_rows = int(getattr(training_result, "train_row_count", 0) or 0)
    test_rows = int(getattr(training_result, "test_row_count", 0) or 0)
    final_training_size = train_rows + test_rows
    original_rows = int(getattr(profile_result, "row_count", 0) or 0)
    rows_removed_missing_target = max(original_rows - final_training_size, 0)

    if not overview["has_missing_values"]:
        return (
            "No missing values were detected. No missing value handling was required.\n\n"
            f"Rows removed because of missing target values: {rows_removed_missing_target}\n\n"
            f"Final dataset size used for training: {final_training_size}"
        )

    analysis_frame = overview["analysis_frame"]
    lines = [
        "Missing values were detected in the dataset.",
        "",
        f"Detected missing values: {overview['total_missing_values']}",
        f"Affected columns: {', '.join(overview['columns']) if overview['columns'] else 'Not specified'}",
        f"Features recommended for dropping: {', '.join(overview['drop_feature_columns']) if overview['drop_feature_columns'] else 'None'}",
        f"Features imputed: {', '.join(overview['imputed_feature_columns']) if overview['imputed_feature_columns'] else 'None'}",
        f"Rows removed because of missing target values: {rows_removed_missing_target}",
        f"Final dataset size used for training: {final_training_size}",
        "",
        "Missing percentages and actions:",
    ]
    for _, row in analysis_frame.iterrows():
        reason = row.get("Reason", "")
        reason_text = f" ({reason})" if reason else ""
        lines.append(
            f"- {row['Column']}: {row['Missing percentage']} missing, recommended action {row['Recommended action']}, "
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


def _build_evaluation_summary(evaluation_result: Any) -> str:
    """Summarize the evaluation result."""

    if getattr(evaluation_result, "classification_metrics", None) is not None:
        metrics = evaluation_result.classification_metrics
        metrics_text = "\n".join(f"- {name}: {value}" for name, value in metrics.items())
    else:
        metrics = getattr(evaluation_result, "regression_metrics", {}) or {}
        metrics_text = "\n".join(f"- {name}: {value}" for name, value in metrics.items())

    return (
        f"Quality rating: {getattr(evaluation_result, 'quality_rating', 'N/A')}\n\n"
        f"Quality summary: {getattr(evaluation_result, 'quality_summary', 'N/A')}\n\n"
        f"Rows evaluated: {getattr(evaluation_result, 'row_count', 'N/A')}\n\n"
        f"Metrics:\n{metrics_text if metrics_text else '- None'}"
    )


def _build_explainability_summary(explainability_result: Any | None) -> str:
    """Summarize explainability output or note when it is unavailable."""

    if explainability_result is None:
        return "Explainability results were not generated for this run."
    if not getattr(explainability_result, "success", False):
        return "Explainability was requested, but the result was not successful."

    top_features = getattr(explainability_result, "top_features", []) or []
    if top_features:
        feature_lines = [f"- {name}: {score:.6f}" for name, score in top_features]
        feature_text = "\n".join(feature_lines)
    else:
        feature_text = "- No feature importance values were available."

    return (
        f"Method used: {getattr(explainability_result, 'method_used', 'N/A')}\n\n"
        f"Summary: {getattr(explainability_result, 'explanation_summary', 'N/A')}\n\n"
        f"Top features:\n{feature_text}"
    )


def _build_limitations_summary() -> str:
    """Describe the scope limits of the prototype."""

    return (
        "This prototype is designed for structured tabular CSV datasets only. It is not intended for multiple tables in one CSV, free-text documents, image datasets, audio datasets, time-series specific workflows, or graph and network datasets."
    )


def _build_conclusion_summary(evaluation_result: Any) -> str:
    """Create a short conclusion based on the completed workflow."""

    return (
        f"The workflow completed with a {getattr(evaluation_result, 'quality_rating', 'N/A')} model result. "
        "The final report combines the dataset profile, task detection, pipeline recommendation, training, evaluation, and explainability outputs into one thesis-friendly summary."
    )


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


def _build_pdf_bytes(markdown_report: str) -> bytes:
    """Create a simple PDF document from the Markdown report."""

    try:
        from reportlab.lib.pagesizes import A4  # type: ignore
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # type: ignore
        from reportlab.lib.units import mm  # type: ignore
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise ImportError("reportlab is not available") from exc

    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    section_style = styles["Heading2"]
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        leading=14,
        spaceAfter=6,
    )

    story = [Paragraph("Explainable ML Pipeline Recommendation System", title_style), Spacer(1, 6)]
    for line in markdown_report.splitlines():
        stripped = line.strip()
        if not stripped:
            story.append(Spacer(1, 4))
            continue
        if stripped.startswith("# "):
            continue
        if stripped.startswith("## "):
            story.append(Paragraph(stripped[3:], section_style))
            continue
        story.append(Paragraph(_markdown_to_plain_text(stripped), body_style))

    document.build(story)
    return buffer.getvalue()


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