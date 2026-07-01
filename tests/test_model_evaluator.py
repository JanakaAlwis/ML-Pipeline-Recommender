"""Tests for the model evaluator."""

from __future__ import annotations

from dataclasses import fields

import pandas as pd

from data_profiler import DatasetProfileResult
from model_evaluator import (
    EvaluationResult,
    InvalidEvaluationInputError,
    ModelEvaluator,
    evaluate_model,
)
from model_trainer import TrainingResult, train_pipeline
from pipeline_builder import build_pipeline
from recommendation_engine import PipelineRecommendation
from task_detector import TaskDetectionResult


def _classification_train_data() -> pd.DataFrame:
    """Create a small classification training dataset."""

    return pd.DataFrame(
        {
            "age": [20, 30, 40, 50, 60, 70, 80, 90],
            "income": [100, 200, 300, 400, 500, 600, 700, 800],
            "city": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


def _classification_eval_data() -> pd.DataFrame:
    """Create a small classification evaluation dataset."""

    return pd.DataFrame(
        {
            "age": [25, 35, 45, 55],
            "income": [150, 250, 350, 450],
            "city": ["A", "B", "A", "B"],
            "target": [0, 1, 0, 1],
        }
    )


def _regression_train_data() -> pd.DataFrame:
    """Create a small regression training dataset."""

    return pd.DataFrame(
        {
            "age": [20, 30, 40, 50, 60, 70, 80, 90],
            "income": [100, 200, 300, 400, 500, 600, 700, 800],
            "city": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "target": [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0],
        }
    )


def _regression_eval_data() -> pd.DataFrame:
    """Create a small regression evaluation dataset."""

    return pd.DataFrame(
        {
            "age": [25, 35, 45, 55],
            "income": [150, 250, 350, 450],
            "city": ["A", "B", "A", "B"],
            "target": [11.0, 13.0, 15.0, 17.0],
        }
    )


def _profile_result(task_type: str, *, eval_mode: bool = False) -> DatasetProfileResult:
    """Create a profile result that matches the datasets above."""

    dataframe = _classification_train_data() if task_type == "classification" else _regression_train_data()
    if eval_mode:
        dataframe = _classification_eval_data() if task_type == "classification" else _regression_eval_data()
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


def _recommendation(task_type: str, model_name: str) -> PipelineRecommendation:
    """Create a pipeline recommendation for evaluation tests."""

    return PipelineRecommendation(
        pipeline_rank=1,
        pipeline_name=f"{task_type.title()} Pipeline",
        task_type=task_type,
        preprocessing_steps=["Impute missing values", "One-hot encode categorical features"],
        encoding_strategy="One-hot encoding for categorical features",
        scaling_strategy="Standard scaling",
        missing_value_strategy="Median imputation for numeric features",
        imbalance_handling_strategy=None,
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


def _trained_result(task_type: str, model_name: str) -> TrainingResult:
    """Train a pipeline for evaluation tests."""

    profile = _profile_result(task_type)
    task = _task_result(task_type)
    recommendation = _recommendation(task_type, model_name)
    built = build_pipeline(recommendation, profile, task)
    data = _classification_train_data() if task_type == "classification" else _regression_train_data()
    return train_pipeline(built, data, "target", task_type, run_cross_validation=False)


def test_evaluation_result_has_expected_fields() -> None:
    """The evaluation result dataclass should expose the required fields."""

    assert [field.name for field in fields(EvaluationResult)] == [
        "success",
        "task_type",
        "model_name",
        "target_column",
        "row_count",
        "predictions",
        "classification_metrics",
        "regression_metrics",
        "confusion_matrix",
        "quality_rating",
        "quality_summary",
        "messages",
        "warnings",
        "errors",
    ]


def test_successful_classification_evaluation() -> None:
    """A trained classification pipeline should evaluate successfully."""

    training_result = _trained_result("classification", "logistic_regression")
    result = evaluate_model(training_result, _classification_eval_data(), "target", "classification")

    assert result.success is True
    assert result.task_type == "classification"
    assert result.predictions is not None
    assert result.classification_metrics is not None
    assert result.regression_metrics is None


def test_successful_regression_evaluation() -> None:
    """A trained regression pipeline should evaluate successfully."""

    training_result = _trained_result("regression", "ridge")
    result = ModelEvaluator().evaluate(training_result, _regression_eval_data(), "target", "regression")

    assert result.success is True
    assert result.task_type == "regression"
    assert result.predictions is not None
    assert result.regression_metrics is not None
    assert result.classification_metrics is None


def test_invalid_training_result() -> None:
    """An invalid training result should fail evaluation."""

    training_result = _trained_result("classification", "logistic_regression")
    training_result.success = False
    training_result.errors = ["training failed"]

    result = evaluate_model(training_result, _classification_eval_data(), "target", "classification")

    assert result.success is False
    assert any("success" in error.lower() for error in result.errors)


def test_missing_target_column() -> None:
    """Missing target columns should be rejected."""

    training_result = _trained_result("classification", "logistic_regression")
    result = evaluate_model(training_result, _classification_eval_data(), "missing", "classification")

    assert result.success is False
    assert any("Target column not found" in error for error in result.errors)


def test_empty_dataframe() -> None:
    """Empty DataFrames should be rejected."""

    training_result = _trained_result("classification", "logistic_regression")
    result = evaluate_model(training_result, pd.DataFrame(), "target", "classification")

    assert result.success is False
    assert any("must not be empty" in error for error in result.errors)


def test_classification_metrics_and_confusion_matrix() -> None:
    """Classification evaluation should return the expected metrics and matrix."""

    training_result = _trained_result("classification", "logistic_regression")
    result = evaluate_model(training_result, _classification_eval_data(), "target", "classification")

    assert result.classification_metrics is not None
    assert set(result.classification_metrics) == {"accuracy", "precision_weighted", "recall_weighted", "f1_weighted"}
    assert result.confusion_matrix is not None
    assert len(result.confusion_matrix) >= 2


def test_regression_metrics() -> None:
    """Regression evaluation should return the expected metrics."""

    training_result = _trained_result("regression", "ridge")
    result = evaluate_model(training_result, _regression_eval_data(), "target", "regression")

    assert result.regression_metrics is not None
    assert set(result.regression_metrics) == {"mae", "mse", "rmse", "r2"}


def test_quality_rating() -> None:
    """Evaluation should provide a quality rating and summary."""

    training_result = _trained_result("classification", "logistic_regression")
    result = evaluate_model(training_result, _classification_eval_data(), "target", "classification")

    assert result.quality_rating in {"excellent", "good", "acceptable", "weak"}
    assert result.quality_summary is not None


def test_prediction_list() -> None:
    """Evaluation should return predictions as a list."""

    training_result = _trained_result("regression", "ridge")
    result = evaluate_model(training_result, _regression_eval_data(), "target", "regression")

    assert isinstance(result.predictions, list)
    assert len(result.predictions) == len(_regression_eval_data())