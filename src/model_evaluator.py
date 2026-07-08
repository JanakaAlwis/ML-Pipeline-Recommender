"""Model evaluator for trained pipelines.

This module evaluates an already trained pipeline on provided tabular data.
It does not train models, recommend models, or build pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from src.model_trainer import TrainingResult

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EvaluationResult:
    """Structured result of a model evaluation run."""

    success: bool
    task_type: str | None
    model_name: str | None
    target_column: str | None
    row_count: int | None
    predictions: list | None
    classification_metrics: dict[str, float] | None
    regression_metrics: dict[str, float] | None
    confusion_matrix: list[list[int]] | None
    quality_rating: str | None
    quality_summary: str | None
    messages: list[str]
    warnings: list[str]
    errors: list[str]


class ModelEvaluatorError(Exception):
    """Base class for model evaluator errors."""


class InvalidEvaluationInputError(ModelEvaluatorError):
    """Raised when evaluation inputs are invalid."""


class EvaluationExecutionError(ModelEvaluatorError):
    """Raised when evaluation fails during prediction or metric calculation."""


class ModelEvaluator:
    """Evaluate a trained pipeline on a tabular dataset."""

    def evaluate(
        self,
        training_result: TrainingResult,
        dataframe: pd.DataFrame,
        target_column: str,
        task_type: str,
    ) -> EvaluationResult:
        """Evaluate a trained pipeline on the provided dataframe.

        Args:
            training_result: Result returned by the model trainer.
            dataframe: Evaluation dataset.
            target_column: Name of the target column.
            task_type: Task type, classification or regression.

        Returns:
            A structured EvaluationResult.
        """

        try:
            _validate_inputs(training_result, dataframe, target_column, task_type)

            pipeline = training_result.trained_pipeline
            if pipeline is None:
                raise InvalidEvaluationInputError("training_result.trained_pipeline must not be None.")

            train_features, test_features, train_target, test_target = _resolve_holdout_split(
                training_result,
                dataframe,
                target_column,
                task_type,
            )

            predictions = pipeline.predict(test_features)
            row_count = int(len(test_target))

            if task_type == "classification":
                return _build_classification_result(
                    training_result=training_result,
                    y_true=test_target,
                    predictions=predictions,
                    row_count=row_count,
                )
            return _build_regression_result(
                training_result=training_result,
                y_true=test_target,
                predictions=predictions,
                row_count=row_count,
            )
        except ModelEvaluatorError as exc:
            logger.error("Model evaluation failed: %s", exc)
            return _failure_result(training_result, target_column, task_type, exc)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.exception("Unexpected evaluation failure")
            return _failure_result(
                training_result,
                target_column,
                task_type,
                EvaluationExecutionError(str(exc)),
            )


def evaluate_model(
    training_result: TrainingResult,
    dataframe: pd.DataFrame,
    target_column: str,
    task_type: str,
) -> EvaluationResult:
    """Evaluate a trained pipeline using the default evaluator."""

    return ModelEvaluator().evaluate(training_result, dataframe, target_column, task_type)


def _validate_inputs(
    training_result: TrainingResult,
    dataframe: pd.DataFrame,
    target_column: str,
    task_type: str,
) -> None:
    """Validate evaluation inputs before prediction."""

    if not isinstance(training_result, TrainingResult):
        raise InvalidEvaluationInputError("training_result must be a TrainingResult instance.")
    if not training_result.success:
        error_details = "; ".join(training_result.errors) if training_result.errors else "unknown error"
        raise InvalidEvaluationInputError(f"training_result.success must be True: {error_details}")
    if training_result.trained_pipeline is None:
        raise InvalidEvaluationInputError("training_result.trained_pipeline must not be None.")
    if not isinstance(dataframe, pd.DataFrame):
        raise InvalidEvaluationInputError("dataframe must be a pandas DataFrame.")
    if dataframe.empty:
        raise InvalidEvaluationInputError("dataframe must not be empty.")
    if not isinstance(target_column, str) or not target_column.strip():
        raise InvalidEvaluationInputError("target_column must be a non-empty string.")
    if target_column not in dataframe.columns:
        raise InvalidEvaluationInputError(f"Target column not found: {target_column}")
    if task_type not in {"classification", "regression"}:
        raise InvalidEvaluationInputError("task_type must be classification or regression.")
    if training_result.task_type is not None and training_result.task_type != task_type:
        raise InvalidEvaluationInputError("training_result.task_type and task_type must match.")


def _resolve_holdout_split(
    training_result: TrainingResult,
    dataframe: pd.DataFrame,
    target_column: str,
    task_type: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return the cleaned hold-out split captured during training when available."""

    train_features = getattr(training_result, "train_features", None)
    test_features = getattr(training_result, "test_features", None)
    train_target = getattr(training_result, "train_target", None)
    test_target = getattr(training_result, "test_target", None)

    if all(value is not None for value in (train_features, test_features, train_target, test_target)):
        return train_features, test_features, train_target, test_target

    raise InvalidEvaluationInputError("Training result does not contain the cleaned hold-out split required for evaluation.")


def _build_classification_result(
    *,
    training_result: TrainingResult,
    y_true: pd.Series,
    predictions: Any,
    row_count: int,
) -> EvaluationResult:
    """Compute classification metrics and quality rating."""

    y_pred = np.asarray(predictions)
    y_true_series = pd.Series(y_true).reset_index(drop=True)
    y_pred_series = pd.Series(y_pred).reset_index(drop=True)
    valid_mask = y_true_series.notna() & y_pred_series.notna()
    if not valid_mask.all():
        y_true_series = y_true_series.loc[valid_mask].reset_index(drop=True)
        y_pred_series = y_pred_series.loc[valid_mask].reset_index(drop=True)

    unique_labels = pd.unique(pd.concat([y_true_series, y_pred_series], ignore_index=True).dropna())
    labels = sorted(unique_labels.tolist(), key=lambda value: str(value))
    y_pred_values = y_pred_series.to_numpy()
    classification_metrics = {
        "accuracy": float(accuracy_score(y_true_series, y_pred_values)),
        "precision_weighted": float(precision_score(y_true_series, y_pred_values, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true_series, y_pred_values, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true_series, y_pred_values, average="weighted", zero_division=0)),
    }
    matrix = confusion_matrix(y_true_series, y_pred_values, labels=labels)
    quality_rating = _rate_quality(classification_metrics["accuracy"])
    quality_summary = _summarize_quality(
        task_type="classification",
        quality_rating=quality_rating,
        primary_score=classification_metrics["accuracy"],
    )

    return EvaluationResult(
        success=True,
        task_type=training_result.task_type,
        model_name=training_result.model_name,
        target_column=training_result.target_column,
        row_count=row_count,
        predictions=y_pred.tolist(),
        classification_metrics=classification_metrics,
        regression_metrics=None,
        confusion_matrix=matrix.astype(int).tolist(),
        quality_rating=quality_rating,
        quality_summary=quality_summary,
        messages=["Evaluation completed successfully."],
        warnings=[],
        errors=[],
    )


def _build_regression_result(
    *,
    training_result: TrainingResult,
    y_true: pd.Series,
    predictions: Any,
    row_count: int,
) -> EvaluationResult:
    """Compute regression metrics and quality rating."""

    y_pred = np.asarray(predictions, dtype=float)
    y_true_values = pd.to_numeric(pd.Series(y_true), errors="coerce").to_numpy(dtype=float)
    valid_mask = ~np.isnan(y_true_values) & ~np.isnan(y_pred)
    if not valid_mask.all():
        y_true_values = y_true_values[valid_mask]
        y_pred = y_pred[valid_mask]
    if y_true_values.size == 0:
        raise EvaluationExecutionError("Regression evaluation could not be computed because no valid target values were available.")
    mse = float(mean_squared_error(y_true_values, y_pred))
    regression_metrics = {
        "mae": float(mean_absolute_error(y_true_values, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true_values, y_pred)),
    }
    quality_rating = _rate_quality(regression_metrics["r2"])
    quality_summary = _summarize_quality(
        task_type="regression",
        quality_rating=quality_rating,
        primary_score=regression_metrics["r2"],
    )

    return EvaluationResult(
        success=True,
        task_type=training_result.task_type,
        model_name=training_result.model_name,
        target_column=training_result.target_column,
        row_count=row_count,
        predictions=y_pred.tolist(),
        classification_metrics=None,
        regression_metrics=regression_metrics,
        confusion_matrix=None,
        quality_rating=quality_rating,
        quality_summary=quality_summary,
        messages=["Evaluation completed successfully."],
        warnings=[],
        errors=[],
    )


def _rate_quality(primary_score: float) -> str:
    """Map a primary metric score to a simple quality rating."""

    if primary_score >= 0.9:
        return "excellent"
    if primary_score >= 0.8:
        return "good"
    if primary_score >= 0.65:
        return "acceptable"
    return "weak"


def _summarize_quality(task_type: str, quality_rating: str, primary_score: float) -> str:
    """Create a plain-English quality summary."""

    if task_type == "classification":
        return f"The classification model achieved a {quality_rating} result with accuracy {primary_score:.3f}."
    return f"The regression model achieved a {quality_rating} result with R2 {primary_score:.3f}."


def _failure_result(
    training_result: TrainingResult,
    target_column: str,
    task_type: str,
    exc: Exception,
) -> EvaluationResult:
    """Build a failed evaluation result."""

    return EvaluationResult(
        success=False,
        task_type=task_type if isinstance(task_type, str) else None,
        model_name=training_result.model_name if isinstance(training_result, TrainingResult) else None,
        target_column=target_column if isinstance(target_column, str) else None,
        row_count=None,
        predictions=None,
        classification_metrics=None,
        regression_metrics=None,
        confusion_matrix=None,
        quality_rating=None,
        quality_summary=None,
        messages=[str(exc)],
        warnings=[],
        errors=[str(exc)],
    )


def _safe_labels(y_true: pd.Series, y_pred: np.ndarray) -> list[object]:
    """Return stable labels for classification metrics without sorting mixed types directly."""

    unique_labels = pd.unique(pd.concat([pd.Series(y_true), pd.Series(y_pred)], ignore_index=True).dropna())
    return sorted(unique_labels.tolist(), key=lambda value: str(value))