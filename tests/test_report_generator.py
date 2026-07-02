"""Tests for the report generator module."""

from __future__ import annotations

import importlib.util
from dataclasses import fields

import pandas as pd

from src.data_profiler import DatasetProfileResult
from src.explainability import explain_model
from src.model_evaluator import evaluate_model
from src.model_trainer import train_pipeline
from src.pipeline_builder import build_pipeline
from src.recommendation_engine import recommend_pipelines
from src.report_generator import ReportResult, generate_report
from src.task_detector import detect_task_type


def _workflow_data() -> pd.DataFrame:
    """Create a small classification dataset for report tests."""

    return pd.DataFrame(
        {
            "age": [20, 30, 40, 50, 60, 70, 80, 90],
            "income": [100, 200, 300, 400, 500, 600, 700, 800],
            "city": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


def _dataset_profile() -> object:
    """Create a profile result compatible with the dataset."""

    dataframe = _workflow_data()
    return DatasetProfileResult(
        success=True,
        dataframe=dataframe,
        messages=["ok"],
        warnings=[],
        errors=[],
        shape=dataframe.shape,
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        column_names=list(dataframe.columns),
        numeric_columns=["age", "income"],
        categorical_columns=["city"],
        boolean_columns=[],
        datetime_columns=[],
        missing_values={column: 0 for column in dataframe.columns},
        missing_percentages={column: 0.0 for column in dataframe.columns},
        duplicate_row_count=0,
        duplicate_row_percentage=0.0,
        constant_columns=[],
        high_cardinality_columns=[],
        sample_rows=dataframe.head(5),
        column_summary=pd.DataFrame(),
        numeric_summary=pd.DataFrame(),
        categorical_summary=pd.DataFrame(),
        correlation_matrix=pd.DataFrame(),
        high_correlation_pairs=[],
        outlier_counts={"age": 0, "income": 0},
        outlier_percentages={"age": 0.0, "income": 0.0},
        target_column="target",
        target_distribution=None,
        class_imbalance_ratio=0.5,
        profile_version="1.0",
    )


def _task_detection_result() -> object:
    """Create a valid task detection result."""

    return detect_task_type(_workflow_data(), "target")


def _trained_workflow_results(include_explainability: bool = True):
    """Build the completed workflow results used by report tests."""

    dataframe = _workflow_data()
    profile_result = _dataset_profile()
    task_result = _task_detection_result()
    recommendation_result = recommend_pipelines(profile_result, task_result)
    built_pipeline = build_pipeline(recommendation_result.top_pipelines[0], profile_result, task_result)
    training_result = train_pipeline(built_pipeline, dataframe, "target", task_result.task_type, run_cross_validation=False)
    evaluation_result = evaluate_model(training_result, dataframe, "target", task_result.task_type)
    explainability_result = (
        explain_model(training_result, dataframe, "target", task_result.task_type)
        if include_explainability
        else None
    )
    return profile_result, task_result, recommendation_result, training_result, evaluation_result, explainability_result


def test_report_result_has_expected_fields() -> None:
    """The report result dataclass should expose the required fields."""

    assert [field.name for field in fields(ReportResult)] == [
        "success",
        "markdown_report",
        "pdf_bytes",
        "report_sections",
        "messages",
        "warnings",
        "errors",
    ]


def test_successful_markdown_report_generation() -> None:
    """Report generation should produce Markdown content."""

    profile_result, task_result, recommendation_result, training_result, evaluation_result, explainability_result = _trained_workflow_results()
    result = generate_report(
        profile_result,
        task_result,
        recommendation_result,
        training_result,
        evaluation_result,
        explainability_result=explainability_result,
        include_pdf=False,
    )

    assert result.success is True
    assert result.markdown_report is not None
    assert "## Dataset Summary" in result.markdown_report
    assert "## Conclusion" in result.markdown_report
    assert "Target column: target" in result.markdown_report


def test_successful_pdf_bytes_generation_or_graceful_fallback() -> None:
    """PDF output should be generated when possible or gracefully skipped."""

    profile_result, task_result, recommendation_result, training_result, evaluation_result, explainability_result = _trained_workflow_results()
    result = generate_report(
        profile_result,
        task_result,
        recommendation_result,
        training_result,
        evaluation_result,
        explainability_result=explainability_result,
        include_pdf=True,
    )

    reportlab_available = importlib.util.find_spec("reportlab") is not None
    if reportlab_available:
        assert result.pdf_bytes is not None
        assert result.pdf_bytes.startswith(b"%PDF")
    else:
        assert result.pdf_bytes is None
        assert any("PDF export is currently unavailable" in warning for warning in result.warnings)


def test_missing_optional_explainability() -> None:
    """Report generation should still succeed without explainability results."""

    profile_result, task_result, recommendation_result, training_result, evaluation_result, _ = _trained_workflow_results(include_explainability=False)
    result = generate_report(
        profile_result,
        task_result,
        recommendation_result,
        training_result,
        evaluation_result,
        explainability_result=None,
        include_pdf=False,
    )

    assert result.success is True
    assert result.markdown_report is not None
    assert "Explainability Summary" in result.report_sections
    assert any("Explainability result was not available" in warning for warning in result.warnings)


def test_target_column_prefers_training_result_then_task_result() -> None:
    """Report generation should prefer the training target column, then the task target column."""

    profile_result, task_result, recommendation_result, training_result, evaluation_result, explainability_result = _trained_workflow_results()
    training_result.target_column = "Attack Type"
    task_result.target_column = "Task Target"
    profile_result.target_column = "Profile Target"

    result = generate_report(
        profile_result,
        task_result,
        recommendation_result,
        training_result,
        evaluation_result,
        explainability_result=explainability_result,
        include_pdf=False,
    )

    assert result.success is True
    assert "Target column: Attack Type" in result.markdown_report


def test_target_column_falls_back_to_task_result() -> None:
    """Report generation should fall back to the task result target column."""

    profile_result, task_result, recommendation_result, training_result, evaluation_result, explainability_result = _trained_workflow_results()
    training_result.target_column = None
    task_result.target_column = "Attack Type"
    profile_result.target_column = "Profile Target"

    result = generate_report(
        profile_result,
        task_result,
        recommendation_result,
        training_result,
        evaluation_result,
        explainability_result=explainability_result,
        include_pdf=False,
    )

    assert result.success is True
    assert "Target column: Attack Type" in result.markdown_report


def test_report_never_contains_target_column_none() -> None:
    """Generated report should not contain Target column: None."""

    profile_result, task_result, recommendation_result, training_result, evaluation_result, explainability_result = _trained_workflow_results()
    training_result.target_column = None
    task_result.target_column = None
    profile_result.target_column = None

    result = generate_report(
        profile_result,
        task_result,
        recommendation_result,
        training_result,
        evaluation_result,
        explainability_result=explainability_result,
        include_pdf=False,
    )

    assert result.success is True
    assert "Target column: None" not in result.markdown_report
    assert "Target column: Not specified" in result.markdown_report


def test_failure_when_required_inputs_are_missing() -> None:
    """Report generation should fail when required inputs are missing."""

    profile_result, task_result, recommendation_result, training_result, evaluation_result, explainability_result = _trained_workflow_results()
    result = generate_report(
        None,
        task_result,
        recommendation_result,
        training_result,
        evaluation_result,
        explainability_result=explainability_result,
        include_pdf=False,
    )

    assert result.success is False
    assert result.markdown_report is None
    assert any("profile_result" in error for error in result.errors)