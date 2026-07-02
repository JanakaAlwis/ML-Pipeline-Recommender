"""Tests for the Explainable ML Pipeline Recommendation Engine."""

from __future__ import annotations

from dataclasses import fields

import pandas as pd
import pytest

from src.data_profiler import DatasetProfileResult
from src.knowledge_base import KnowledgeBase
from src.recommendation_engine import (
    ENGINE_VERSION,
    InvalidRecommendationInputError,
    PipelineRecommendation,
    PipelineRecommendationResult,
    RecommendationEngine,
    SupportingRecommendationItem,
    recommend_pipelines,
    validate_recommendation_input,
)
from src.task_detector import TaskDetectionResult


def _make_profile_result(
    *,
    row_count: int,
    column_count: int,
    numeric_columns: list[str],
    categorical_columns: list[str],
    missing_values: dict[str, int],
    class_imbalance_ratio: float | None,
    high_correlation_pairs: list[tuple[str, str, float]],
    high_cardinality_columns: list[str] | None = None,
) -> DatasetProfileResult:
    """Create a minimal valid profile result for tests."""

    dataframe = pd.DataFrame({"feature": list(range(row_count)), "target": list(range(row_count))})
    return DatasetProfileResult(
        success=True,
        dataframe=dataframe,
        messages=["ok"],
        warnings=[],
        errors=[],
        shape=(row_count, column_count),
        row_count=row_count,
        column_count=column_count,
        column_names=["feature", "target"],
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        boolean_columns=[],
        datetime_columns=[],
        missing_values=missing_values,
        missing_percentages={column: (count / row_count) * 100 for column, count in missing_values.items()},
        duplicate_row_count=0,
        duplicate_row_percentage=0.0,
        constant_columns=[],
        high_cardinality_columns=high_cardinality_columns or [],
        sample_rows=dataframe.head(5),
        column_summary=pd.DataFrame(),
        numeric_summary=pd.DataFrame(),
        categorical_summary=pd.DataFrame(),
        correlation_matrix=pd.DataFrame(),
        high_correlation_pairs=high_correlation_pairs,
        outlier_counts={column: 0 for column in numeric_columns},
        outlier_percentages={column: 0.0 for column in numeric_columns},
        target_column="target",
        target_distribution=pd.DataFrame({"target": [0, 1], "count": [row_count - 10, 10]}),
        class_imbalance_ratio=class_imbalance_ratio,
        profile_version="1.0",
    )


def _make_task_result(task_type: str, *, target_unique_count: int, confidence: float = 0.95) -> TaskDetectionResult:
    """Create a minimal valid task result for tests."""

    return TaskDetectionResult(
        success=True,
        task_type=task_type,
        confidence=confidence,
        messages=["ok"],
        warnings=[],
        errors=[],
        target_column="target",
        target_dtype="int64" if task_type == "classification" else "float64",
        target_unique_count=target_unique_count,
        target_unique_ratio=0.1,
        is_numeric_target=True,
        is_categorical_target=task_type == "classification",
        is_boolean_target=False,
        class_count=target_unique_count if task_type == "classification" else None,
        detection_reason="test",
        alternative_task_type=None,
        alternative_confidence=None,
        profile_version="1.0",
    )


def test_pipeline_recommendation_result_has_expected_fields() -> None:
    """The result dataclass should expose the agreed fields."""

    assert [field.name for field in fields(PipelineRecommendationResult)] == [
        "success",
        "task_type",
        "messages",
        "warnings",
        "errors",
        "top_pipelines",
        "supporting_recommendations",
        "profile_summary",
        "task_summary",
        "engine_version",
        "knowledge_base_version",
    ]


def test_pipeline_recommendation_has_expected_fields() -> None:
    """The pipeline dataclass should expose the agreed fields."""

    assert [field.name for field in fields(PipelineRecommendation)] == [
        "pipeline_rank",
        "pipeline_name",
        "task_type",
        "preprocessing_steps",
        "encoding_strategy",
        "scaling_strategy",
        "missing_value_strategy",
        "imbalance_handling_strategy",
        "dimensionality_reduction_strategy",
        "recommended_model_family",
        "evaluation_metrics",
        "cross_validation_recommendation",
        "confidence_score",
        "priority",
        "explanation",
        "evidence",
        "advantages",
        "limitations",
        "estimated_training_complexity",
        "supporting_recommendation_ids",
    ]


def test_supporting_recommendation_item_has_expected_fields() -> None:
    """The supporting item dataclass should expose the agreed fields."""

    assert [field.name for field in fields(SupportingRecommendationItem)] == [
        "recommendation_id",
        "category",
        "name",
        "recommendation",
        "reason",
        "confidence_score",
        "severity",
        "priority",
        "evidence",
        "applies_to",
        "advantages",
        "limitations",
    ]


def test_recommend_pipelines_generates_three_classification_pipelines() -> None:
    """Classification input should yield three ranked pipelines."""

    profile = _make_profile_result(
        row_count=120,
        column_count=6,
        numeric_columns=["age", "income"],
        categorical_columns=["city", "segment"],
        missing_values={"age": 2, "income": 0, "city": 4, "segment": 0},
        class_imbalance_ratio=0.35,
        high_correlation_pairs=[("age", "income", 0.82)],
    )
    task = _make_task_result("classification", target_unique_count=2)

    result = recommend_pipelines(profile, task)

    assert result.success is True
    assert result.task_type == "classification"
    assert len(result.top_pipelines) == 5
    assert [pipeline.pipeline_rank for pipeline in result.top_pipelines] == [1, 2, 3, 4, 5]
    assert any("classification" in pipeline.pipeline_name.lower() for pipeline in result.top_pipelines)


def test_recommend_pipelines_generates_three_regression_pipelines() -> None:
    """Regression input should yield three ranked pipelines."""

    profile = _make_profile_result(
        row_count=150,
        column_count=8,
        numeric_columns=["x1", "x2", "x3", "x4"],
        categorical_columns=["region"],
        missing_values={"x1": 1, "x2": 0, "region": 3},
        class_imbalance_ratio=None,
        high_correlation_pairs=[("x1", "x2", 0.9)],
    )
    task = _make_task_result("regression", target_unique_count=120)

    result = recommend_pipelines(profile, task)

    assert result.success is True
    assert result.task_type == "regression"
    assert len(result.top_pipelines) == 5
    assert [pipeline.pipeline_rank for pipeline in result.top_pipelines] == [1, 2, 3, 4, 5]
    assert any("regression" in pipeline.pipeline_name.lower() for pipeline in result.top_pipelines)


def test_recommend_pipelines_includes_imputation_support() -> None:
    """Missing values should trigger imputation support items."""

    profile = _make_profile_result(
        row_count=120,
        column_count=5,
        numeric_columns=["age", "income"],
        categorical_columns=["city"],
        missing_values={"age": 2, "income": 1, "city": 0},
        class_imbalance_ratio=0.5,
        high_correlation_pairs=[],
    )
    task = _make_task_result("classification", target_unique_count=2)

    result = recommend_pipelines(profile, task)

    assert any(item.category == "imputation" for item in result.supporting_recommendations)
    assert any("impute" in step.lower() for pipeline in result.top_pipelines for step in pipeline.preprocessing_steps)


def test_recommend_pipelines_includes_encoding_support() -> None:
    """Categorical features should trigger encoding support items."""

    profile = _make_profile_result(
        row_count=120,
        column_count=5,
        numeric_columns=["age"],
        categorical_columns=["city", "segment"],
        missing_values={"age": 0, "city": 0, "segment": 0},
        class_imbalance_ratio=0.5,
        high_correlation_pairs=[],
    )
    task = _make_task_result("classification", target_unique_count=2)

    result = recommend_pipelines(profile, task)

    assert any(item.category == "encoding" for item in result.supporting_recommendations)
    assert any("encoding" in pipeline.encoding_strategy.lower() for pipeline in result.top_pipelines)


def test_recommend_pipelines_includes_scaling_support_when_needed() -> None:
    """Scale-sensitive pipelines should include scaling support."""

    profile = _make_profile_result(
        row_count=120,
        column_count=5,
        numeric_columns=["age", "income"],
        categorical_columns=["city"],
        missing_values={"age": 0, "income": 0, "city": 0},
        class_imbalance_ratio=0.5,
        high_correlation_pairs=[],
    )
    task = _make_task_result("classification", target_unique_count=2)

    result = recommend_pipelines(profile, task)

    assert any(item.category == "scaling" for item in result.supporting_recommendations)
    assert any("scaling" in pipeline.scaling_strategy.lower() for pipeline in result.top_pipelines)


def test_recommend_pipelines_includes_imbalance_support_for_classification() -> None:
    """Strong imbalance should trigger imbalance handling support."""

    profile = _make_profile_result(
        row_count=200,
        column_count=6,
        numeric_columns=["age", "income"],
        categorical_columns=["city"],
        missing_values={"age": 0, "income": 0, "city": 0},
        class_imbalance_ratio=0.1,
        high_correlation_pairs=[],
    )
    task = _make_task_result("classification", target_unique_count=2)

    result = recommend_pipelines(profile, task)

    assert any(item.category == "imbalance_handling" for item in result.supporting_recommendations)
    assert any(
        pipeline.imbalance_handling_strategy is not None
        and pipeline.imbalance_handling_strategy != "No imbalance handling required"
        for pipeline in result.top_pipelines
    )


def test_recommend_pipelines_includes_dimensionality_reduction_for_correlated_features() -> None:
    """Strong correlations should trigger dimensionality reduction support."""

    profile = _make_profile_result(
        row_count=150,
        column_count=12,
        numeric_columns=[f"x{i}" for i in range(10)],
        categorical_columns=["region"],
        missing_values={"x1": 0, "x2": 0, "region": 0},
        class_imbalance_ratio=None,
        high_correlation_pairs=[("x1", "x2", 0.9), ("x3", "x4", 0.85)],
    )
    task = _make_task_result("regression", target_unique_count=150)

    result = recommend_pipelines(profile, task)

    assert any(
        item.category in {"dimensionality_reduction", "feature_selection"}
        for item in result.supporting_recommendations
    )
    assert any(
        pipeline.dimensionality_reduction_strategy is not None
        and pipeline.dimensionality_reduction_strategy != "No dimensionality reduction required"
        for pipeline in result.top_pipelines
    )


def test_recommend_pipelines_includes_classification_metrics() -> None:
    """Classification pipelines should include classification metrics."""

    profile = _make_profile_result(
        row_count=120,
        column_count=5,
        numeric_columns=["age"],
        categorical_columns=["city"],
        missing_values={"age": 0, "city": 0},
        class_imbalance_ratio=0.3,
        high_correlation_pairs=[],
    )
    task = _make_task_result("classification", target_unique_count=2)

    result = recommend_pipelines(profile, task)

    assert any("precision" in metric.lower() for metric in result.top_pipelines[0].evaluation_metrics)
    assert any(item.category == "evaluation_metrics_classification" for item in result.supporting_recommendations)


def test_recommend_pipelines_includes_regression_metrics() -> None:
    """Regression pipelines should include regression metrics."""

    profile = _make_profile_result(
        row_count=120,
        column_count=5,
        numeric_columns=["x1", "x2"],
        categorical_columns=["city"],
        missing_values={"x1": 0, "city": 0},
        class_imbalance_ratio=None,
        high_correlation_pairs=[("x1", "x2", 0.9)],
    )
    task = _make_task_result("regression", target_unique_count=120)

    result = recommend_pipelines(profile, task)

    assert any(metric in {"MAE", "MSE", "RMSE", "R2"} for metric in result.top_pipelines[0].evaluation_metrics)
    assert any(item.category == "evaluation_metrics_regression" for item in result.supporting_recommendations)


def test_recommend_pipelines_returns_ranked_pipelines() -> None:
    """Pipelines should be ranked by confidence and priority."""

    profile = _make_profile_result(
        row_count=200,
        column_count=6,
        numeric_columns=["age", "income"],
        categorical_columns=["city", "segment"],
        missing_values={"age": 0, "income": 0, "city": 0, "segment": 0},
        class_imbalance_ratio=0.1,
        high_correlation_pairs=[("age", "income", 0.85)],
    )
    task = _make_task_result("classification", target_unique_count=2)

    result = recommend_pipelines(profile, task)

    confidences = [pipeline.confidence_score for pipeline in result.top_pipelines]
    assert confidences == sorted(confidences, reverse=True)
    assert [pipeline.priority for pipeline in result.top_pipelines] == [1, 2, 3, 4, 5]
    assert [pipeline.pipeline_rank for pipeline in result.top_pipelines] == [1, 2, 3, 4, 5]


def test_recommend_pipelines_confidence_scores_are_between_zero_and_one() -> None:
    """All pipeline confidence scores should stay within the valid range."""

    profile = _make_profile_result(
        row_count=150,
        column_count=8,
        numeric_columns=["x1", "x2", "x3"],
        categorical_columns=["region"],
        missing_values={"x1": 1, "region": 0},
        class_imbalance_ratio=None,
        high_correlation_pairs=[("x1", "x2", 0.9)],
    )
    task = _make_task_result("regression", target_unique_count=150)

    result = recommend_pipelines(profile, task)

    assert all(0.0 <= pipeline.confidence_score <= 1.0 for pipeline in result.top_pipelines)
    assert all(0.0 <= item.confidence_score <= 1.0 for item in result.supporting_recommendations)


def test_recommend_pipelines_includes_explanation_and_evidence() -> None:
    """Pipelines should include explanations and evidence strings."""

    profile = _make_profile_result(
        row_count=120,
        column_count=6,
        numeric_columns=["age", "income"],
        categorical_columns=["city"],
        missing_values={"age": 1, "income": 0, "city": 0},
        class_imbalance_ratio=0.3,
        high_correlation_pairs=[],
    )
    task = _make_task_result("classification", target_unique_count=2)

    result = recommend_pipelines(profile, task)

    assert all(pipeline.explanation for pipeline in result.top_pipelines)
    assert all(pipeline.evidence for pipeline in result.top_pipelines)


def test_recommend_pipelines_includes_advantages_and_limitations() -> None:
    """Pipelines should include advantages and limitations."""

    profile = _make_profile_result(
        row_count=150,
        column_count=8,
        numeric_columns=["x1", "x2"],
        categorical_columns=["region"],
        missing_values={"x1": 0, "region": 0},
        class_imbalance_ratio=None,
        high_correlation_pairs=[("x1", "x2", 0.9)],
    )
    task = _make_task_result("regression", target_unique_count=150)

    result = recommend_pipelines(profile, task)

    assert all(pipeline.advantages for pipeline in result.top_pipelines)
    assert all(pipeline.limitations for pipeline in result.top_pipelines)


def test_recommend_pipelines_returns_failure_for_invalid_upstream_input() -> None:
    """Failed upstream results should return a failed recommendation result."""

    profile = _make_profile_result(
        row_count=120,
        column_count=5,
        numeric_columns=["age"],
        categorical_columns=["city"],
        missing_values={"age": 0, "city": 0},
        class_imbalance_ratio=0.5,
        high_correlation_pairs=[],
    )
    profile.success = False
    profile.errors = ["profile failed"]
    task = _make_task_result("classification", target_unique_count=2)

    result = recommend_pipelines(profile, task)

    assert result.success is False
    assert result.errors
    assert not result.top_pipelines


def test_validate_recommendation_input_accepts_valid_upstream_results() -> None:
    """Valid upstream results should pass validation."""

    profile = _make_profile_result(
        row_count=120,
        column_count=5,
        numeric_columns=["age"],
        categorical_columns=["city"],
        missing_values={"age": 0, "city": 0},
        class_imbalance_ratio=0.5,
        high_correlation_pairs=[],
    )
    task = _make_task_result("classification", target_unique_count=2)

    validate_recommendation_input(profile, task)


def test_validate_recommendation_input_rejects_failed_profile_result() -> None:
    """A failed profile result should be rejected."""

    profile = _make_profile_result(
        row_count=120,
        column_count=5,
        numeric_columns=["age"],
        categorical_columns=["city"],
        missing_values={"age": 0, "city": 0},
        class_imbalance_ratio=0.5,
        high_correlation_pairs=[],
    )
    profile.success = False
    profile.errors = ["profile failed"]
    task = _make_task_result("classification", target_unique_count=2)

    with pytest.raises(InvalidRecommendationInputError):
        validate_recommendation_input(profile, task)


def test_validate_recommendation_input_rejects_failed_task_result() -> None:
    """A failed task result should be rejected."""

    profile = _make_profile_result(
        row_count=120,
        column_count=5,
        numeric_columns=["age"],
        categorical_columns=["city"],
        missing_values={"age": 0, "city": 0},
        class_imbalance_ratio=0.5,
        high_correlation_pairs=[],
    )
    task = _make_task_result("classification", target_unique_count=2)
    task.success = False
    task.errors = ["task failed"]

    with pytest.raises(InvalidRecommendationInputError):
        validate_recommendation_input(profile, task)
