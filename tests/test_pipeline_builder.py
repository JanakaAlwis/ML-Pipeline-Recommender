"""Tests for the executable pipeline builder."""

from __future__ import annotations

from dataclasses import fields

import pandas as pd
import pytest

from src.data_profiler import DatasetProfileResult
from src.pipeline_builder import (
    BuiltPipelineResult,
    InvalidPipelineRecommendationError,
    PipelineBuilder,
    PipelineConstructionError,
    UnsupportedModelError,
    build_pipeline,
)
from src.recommendation_engine import PipelineRecommendation
from src.task_detector import TaskDetectionResult


def _make_profile_result(*, task_type: str, include_booleans: bool = False) -> DatasetProfileResult:
    """Create a small valid dataset profile result for builder tests."""

    dataframe = pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
            "income": [1000, 2000, 3000, 4000],
            "city": ["A", "B", "A", "B"],
            "segment": ["x", "y", "x", "y"],
            "flag": [True, False, True, False],
            "target": [0, 1, 0, 1] if task_type == "classification" else [10.0, 11.0, 12.0, 13.0],
        }
    )
    boolean_columns = ["flag"] if include_booleans else []
    return DatasetProfileResult(
        success=True,
        dataframe=dataframe,
        messages=["ok"],
        warnings=[],
        errors=[],
        shape=(4, 6),
        row_count=4,
        column_count=6,
        column_names=list(dataframe.columns),
        numeric_columns=["age", "income"],
        categorical_columns=["city", "segment"],
        boolean_columns=boolean_columns,
        datetime_columns=[],
        missing_values={"age": 0, "income": 0, "city": 0, "segment": 0, "flag": 0, "target": 0},
        missing_percentages={"age": 0.0},
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
        target_distribution=pd.DataFrame({"target": [0, 1], "count": [2, 2]}),
        class_imbalance_ratio=0.5 if task_type == "classification" else None,
        profile_version="1.0",
    )


def _make_task_result(task_type: str) -> TaskDetectionResult:
    """Create a small valid task detection result for builder tests."""

    return TaskDetectionResult(
        success=True,
        task_type=task_type,
        confidence=0.95,
        messages=["ok"],
        warnings=[],
        errors=[],
        target_column="target",
        target_dtype="int64" if task_type == "classification" else "float64",
        target_unique_count=2 if task_type == "classification" else 4,
        target_unique_ratio=0.5,
        is_numeric_target=True,
        is_categorical_target=task_type == "classification",
        is_boolean_target=False,
        class_count=2 if task_type == "classification" else None,
        detection_reason="test",
        alternative_task_type=None,
        alternative_confidence=None,
        profile_version="1.0",
    )


def _make_recommendation(
    *,
    task_type: str,
    model_name: str,
    scaling_strategy: str = "Standard scaling",
    dimensionality_reduction_strategy: str | None = None,
    imbalance_handling_strategy: str | None = None,
) -> PipelineRecommendation:
    """Create a pipeline recommendation for builder tests."""

    return PipelineRecommendation(
        pipeline_rank=1,
        pipeline_name="Test Pipeline",
        task_type=task_type,
        preprocessing_steps=["Impute missing values", "Encode categorical features"],
        encoding_strategy="One-hot encoding for categorical features",
        scaling_strategy=scaling_strategy,
        missing_value_strategy="Median imputation for numeric features",
        imbalance_handling_strategy=imbalance_handling_strategy,
        dimensionality_reduction_strategy=dimensionality_reduction_strategy,
        recommended_model_family=model_name,
        evaluation_metrics=["Accuracy", "F1-score"] if task_type == "classification" else ["MAE", "RMSE"],
        cross_validation_recommendation="Use 5-fold cross-validation and compare with a baseline pipeline.",
        confidence_score=0.9,
        priority=1,
        explanation="test",
        evidence=["test evidence"],
        advantages=["Simple"],
        limitations=["May underfit"],
        estimated_training_complexity="low",
        supporting_recommendation_ids=["RULE-1"],
    )


def test_built_pipeline_result_has_expected_fields() -> None:
    """The built result dataclass should expose the required fields."""

    assert [field.name for field in fields(BuiltPipelineResult)] == [
        "success",
        "pipeline",
        "pipeline_name",
        "task_type",
        "model_name",
        "preprocessing_steps",
        "messages",
        "warnings",
        "errors",
    ]


def test_classification_pipeline_creation() -> None:
    """A classification recommendation should produce an executable pipeline."""

    profile = _make_profile_result(task_type="classification")
    task = _make_task_result("classification")
    recommendation = _make_recommendation(task_type="classification", model_name="logistic_regression")

    result = build_pipeline(recommendation, profile, task)

    assert result.success is True
    assert result.pipeline is not None
    assert result.pipeline_name == "Test Pipeline"
    assert result.task_type == "classification"
    assert result.model_name == "LogisticRegression"


def test_regression_pipeline_creation() -> None:
    """A regression recommendation should produce an executable pipeline."""

    profile = _make_profile_result(task_type="regression")
    task = _make_task_result("regression")
    recommendation = _make_recommendation(task_type="regression", model_name="ridge")

    result = PipelineBuilder().build(recommendation, profile, task)

    assert result.success is True
    assert result.pipeline is not None
    assert result.task_type == "regression"
    assert result.model_name == "Ridge"


def test_numeric_preprocessing_uses_imputer_and_scaler() -> None:
    """Numeric preprocessing should include an imputer and the requested scaler."""

    profile = _make_profile_result(task_type="classification")
    task = _make_task_result("classification")
    recommendation = _make_recommendation(task_type="classification", model_name="logistic_regression", scaling_strategy="Robust scaling")

    result = build_pipeline(recommendation, profile, task)

    numeric_pipeline = result.pipeline.named_steps["preprocessor"].transformers[0][1]
    assert numeric_pipeline.named_steps["imputer"].strategy == "median"
    assert numeric_pipeline.named_steps["scaler"].__class__.__name__ == "RobustScaler"


def test_categorical_preprocessing_uses_imputer_and_one_hot_encoder() -> None:
    """Categorical preprocessing should include imputation and one-hot encoding."""

    profile = _make_profile_result(task_type="classification")
    task = _make_task_result("classification")
    recommendation = _make_recommendation(task_type="classification", model_name="logistic_regression")

    result = build_pipeline(recommendation, profile, task)

    categorical_pipeline = result.pipeline.named_steps["preprocessor"].transformers[1][1]
    assert categorical_pipeline.named_steps["imputer"].strategy == "most_frequent"
    assert categorical_pipeline.named_steps["onehot"].__class__.__name__ == "OneHotEncoder"


def test_scaler_selection_minmax() -> None:
    """MinMax scaling should map to MinMaxScaler."""

    profile = _make_profile_result(task_type="classification")
    task = _make_task_result("classification")
    recommendation = _make_recommendation(task_type="classification", model_name="logistic_regression", scaling_strategy="MinMax scaling")

    result = build_pipeline(recommendation, profile, task)

    numeric_pipeline = result.pipeline.named_steps["preprocessor"].transformers[0][1]
    assert numeric_pipeline.named_steps["scaler"].__class__.__name__ == "MinMaxScaler"


def test_scaler_selection_none() -> None:
    """Optional scaling should skip the scaler step."""

    profile = _make_profile_result(task_type="classification")
    task = _make_task_result("classification")
    recommendation = _make_recommendation(task_type="classification", model_name="decision_tree", scaling_strategy="No scaling required")

    result = build_pipeline(recommendation, profile, task)

    numeric_pipeline = result.pipeline.named_steps["preprocessor"].transformers[0][1]
    assert "scaler" not in numeric_pipeline.named_steps


def test_pca_inclusion() -> None:
    """A PCA recommendation should add a PCA step."""

    profile = _make_profile_result(task_type="classification")
    task = _make_task_result("classification")
    recommendation = _make_recommendation(
        task_type="classification",
        model_name="random_forest",
        dimensionality_reduction_strategy="Use PCA to reduce dimensionality",
    )

    result = build_pipeline(recommendation, profile, task)

    assert "pca" in result.pipeline.named_steps
    assert result.pipeline.named_steps["pca"].__class__.__name__ == "PCA"


def test_smote_inclusion_for_classification() -> None:
    """SMOTE should be inserted for imbalanced classification recommendations."""

    profile = _make_profile_result(task_type="classification")
    task = _make_task_result("classification")
    recommendation = _make_recommendation(
        task_type="classification",
        model_name="logistic_regression",
        imbalance_handling_strategy="Use SMOTE inside training folds",
    )

    result = build_pipeline(recommendation, profile, task)

    assert result.pipeline.__class__.__module__.startswith("imblearn")
    assert "smote" in result.pipeline.named_steps
    assert result.pipeline.named_steps["smote"].__class__.__name__ == "SMOTE"


def test_smote_rejection_for_regression() -> None:
    """Regression recommendations must reject SMOTE."""

    profile = _make_profile_result(task_type="regression")
    task = _make_task_result("regression")
    recommendation = _make_recommendation(
        task_type="regression",
        model_name="ridge",
        imbalance_handling_strategy="Use SMOTE inside training folds",
    )

    result = build_pipeline(recommendation, profile, task)

    assert result.success is False
    assert any("SMOTE" in error for error in result.errors)


def test_unsupported_model_handling() -> None:
    """Unsupported model families should return a failure result."""

    profile = _make_profile_result(task_type="classification")
    task = _make_task_result("classification")
    recommendation = _make_recommendation(task_type="classification", model_name="unsupported_model")

    result = build_pipeline(recommendation, profile, task)

    assert result.success is False
    assert any("Unsupported model family" in error for error in result.errors)


def test_invalid_recommendation_handling() -> None:
    """Invalid upstream inputs should fail cleanly."""

    profile = _make_profile_result(task_type="classification")
    task = _make_task_result("classification")
    bad_recommendation = _make_recommendation(task_type="classification", model_name="logistic_regression")
    bad_recommendation.task_type = "regression"

    result = build_pipeline(bad_recommendation, profile, task)

    assert result.success is False
    assert any("must match" in error for error in result.errors)