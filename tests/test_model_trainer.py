"""Tests for the model trainer and pipeline executor."""

from __future__ import annotations

from dataclasses import fields

import pandas as pd

from src.data_profiler import DatasetProfileResult
from src.model_trainer import (
    BuiltPipelineResult,
    InvalidTrainingInputError,
    ModelTrainer,
    TrainingResult,
    train_pipeline,
)
from src.pipeline_builder import build_pipeline
from src.recommendation_engine import PipelineRecommendation
from src.task_detector import TaskDetectionResult


def _classification_data() -> pd.DataFrame:
    """Create a small classification dataset."""

    return pd.DataFrame(
        {
            "age": [20, 30, 40, 50, 60, 70, 80, 90],
            "income": [100, 200, 300, 400, 500, 600, 700, 800],
            "city": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


def _regression_data() -> pd.DataFrame:
    """Create a small regression dataset."""

    return pd.DataFrame(
        {
            "age": [20, 30, 40, 50, 60, 70, 80, 90],
            "income": [100, 200, 300, 400, 500, 600, 700, 800],
            "city": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "target": [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0],
        }
    )


def _profile_result(task_type: str) -> DatasetProfileResult:
    """Create a dataset profile result that matches the datasets above."""

    dataframe = _classification_data() if task_type == "classification" else _regression_data()
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
        class_imbalance_ratio=0.5 if task_type == "classification" else None,
        profile_version="1.0",
    )


def _task_result(task_type: str) -> TaskDetectionResult:
    """Create a valid task detection result."""

    return TaskDetectionResult(
        success=True,
        task_type=task_type,
        confidence=0.95,
        messages=["ok"],
        warnings=[],
        errors=[],
        target_column="target",
        target_dtype="int64" if task_type == "classification" else "float64",
        target_unique_count=2 if task_type == "classification" else 8,
        target_unique_ratio=0.25,
        is_numeric_target=True,
        is_categorical_target=task_type == "classification",
        is_boolean_target=False,
        class_count=2 if task_type == "classification" else None,
        detection_reason="test",
        alternative_task_type=None,
        alternative_confidence=None,
        profile_version="1.0",
    )


def _recommendation(task_type: str, model_name: str, imbalance_strategy: str | None = None) -> PipelineRecommendation:
    """Create a pipeline recommendation for training tests."""

    return PipelineRecommendation(
        pipeline_rank=1,
        pipeline_name=f"{task_type.title()} Pipeline",
        task_type=task_type,
        preprocessing_steps=["Impute missing values", "One-hot encode categorical features"],
        encoding_strategy="One-hot encoding for categorical features",
        scaling_strategy="Standard scaling",
        missing_value_strategy="Median imputation for numeric features",
        imbalance_handling_strategy=imbalance_strategy,
        dimensionality_reduction_strategy=None,
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


def _built_result(task_type: str, model_name: str, imbalance_strategy: str | None = None) -> BuiltPipelineResult:
    """Build a concrete executable pipeline for training tests."""

    profile = _profile_result(task_type)
    task = _task_result(task_type)
    recommendation = _recommendation(task_type, model_name, imbalance_strategy)
    return build_pipeline(recommendation, profile, task)


def test_training_result_has_expected_fields() -> None:
    """The training result dataclass should expose the required fields."""

    assert [field.name for field in fields(TrainingResult)] == [
        "success",
        "trained_pipeline",
        "model_name",
        "task_type",
        "target_column",
        "train_row_count",
        "test_row_count",
        "train_features",
        "test_features",
        "train_target",
        "test_target",
        "training_time_seconds",
        "cross_validation_scores",
        "mean_cv_score",
        "std_cv_score",
        "scoring_metric",
        "messages",
        "warnings",
        "errors",
    ]


def test_successful_classification_training() -> None:
    """A classification pipeline should train successfully."""

    result = train_pipeline(_built_result("classification", "logistic_regression"), _classification_data(), "target", "classification")

    assert result.success is True
    assert result.task_type == "classification"
    assert result.trained_pipeline is not None
    assert result.model_name == "LogisticRegression"


def test_successful_regression_training() -> None:
    """A regression pipeline should train successfully."""

    result = ModelTrainer().train(_built_result("regression", "ridge"), _regression_data(), "target", "regression")

    assert result.success is True
    assert result.task_type == "regression"
    assert result.trained_pipeline is not None
    assert result.model_name == "Ridge"


def test_invalid_built_pipeline_result() -> None:
    """An unsuccessful built pipeline result should fail training."""

    built = _built_result("classification", "logistic_regression")
    built.success = False
    built.errors = ["builder failed"]

    result = train_pipeline(built, _classification_data(), "target", "classification")

    assert result.success is False
    assert any("success" in error.lower() for error in result.errors)


def test_missing_target_column() -> None:
    """Missing target columns should be rejected."""

    result = train_pipeline(_built_result("classification", "logistic_regression"), _classification_data(), "missing", "classification")

    assert result.success is False
    assert any("Target column not found" in error for error in result.errors)


def test_empty_dataframe() -> None:
    """Empty DataFrames should be rejected."""

    result = train_pipeline(_built_result("classification", "logistic_regression"), pd.DataFrame(), "target", "classification")

    assert result.success is False
    assert any("must not be empty" in error for error in result.errors)


def test_cross_validation_scores() -> None:
    """Cross-validation should return numeric scores when requested."""

    result = train_pipeline(_built_result("classification", "logistic_regression"), _classification_data(), "target", "classification", run_cross_validation=True)

    assert result.success is True
    assert result.cross_validation_scores
    assert result.mean_cv_score is not None
    assert result.std_cv_score is not None
    assert result.scoring_metric == "f1_weighted"


def test_training_time_is_recorded() -> None:
    """Training time should be measured and returned."""

    result = train_pipeline(_built_result("regression", "ridge"), _regression_data(), "target", "regression")

    assert result.success is True
    assert result.training_time_seconds is not None
    assert result.training_time_seconds >= 0.0


def test_no_cross_validation_mode() -> None:
    """Cross-validation can be disabled."""

    result = train_pipeline(_built_result("classification", "logistic_regression"), _classification_data(), "target", "classification", run_cross_validation=False)

    assert result.success is True
    assert result.cross_validation_scores == []
    assert result.mean_cv_score is None
    assert result.std_cv_score is None


def test_rows_with_missing_target_are_removed_before_split() -> None:
    """Rows with missing target values should be removed before train/test split."""

    dataframe = _classification_data().copy()
    dataframe.loc[0, "target"] = None

    result = train_pipeline(_built_result("classification", "logistic_regression"), dataframe, "target", "classification", run_cross_validation=False)

    assert result.success is True
    assert (result.train_row_count or 0) + (result.test_row_count or 0) == len(dataframe) - 1
    assert any("target column contains missing values" in warning.lower() for warning in result.warnings)
    assert result.train_target is not None
    assert result.test_target is not None
    assert not result.train_target.isna().any()
    assert not result.test_target.isna().any()


def test_smote_pipeline_trains_for_classification() -> None:
    """A pipeline with SMOTE should still train for classification."""

    result = train_pipeline(
        _built_result("classification", "logistic_regression", imbalance_strategy="Use SMOTE inside training folds"),
        _classification_data(),
        "target",
        "classification",
    )

    assert result.success is True
    assert result.trained_pipeline is not None


def test_regression_smote_rejection_is_propagated() -> None:
    """A regression pipeline should fail if the built pipeline itself is invalid for SMOTE."""

    built = _built_result("regression", "ridge", imbalance_strategy="Use SMOTE inside training folds")
    result = train_pipeline(built, _regression_data(), "target", "regression")

    assert result.success is False
    assert any("SMOTE" in error for error in result.errors)