"""Model explainability utilities for the thesis project.

This module explains a trained pipeline using permutation importance on the
original input features. SHAP support is optional and not required for the
core workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.model_trainer import TrainingResult

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ExplainabilityResult:
    """Structured result of a model explainability run."""

    success: bool
    method_used: str | None
    feature_importance: dict[str, float] | None
    top_features: list[tuple[str, float]]
    explanation_summary: str | None
    messages: list[str]
    warnings: list[str]
    errors: list[str]


class ExplainabilityError(Exception):
    """Base class for explainability errors."""


class InvalidExplainabilityInputError(ExplainabilityError):
    """Raised when the explainability inputs are invalid."""


class ExplainabilityComputationError(ExplainabilityError):
    """Raised when explainability computation fails."""


def explain_model(
    training_result: TrainingResult,
    dataframe: pd.DataFrame,
    target_column: str,
    task_type: str,
) -> ExplainabilityResult:
    """Explain a trained model using permutation importance.

    Args:
        training_result: Output of the training phase.
        dataframe: Dataset used for explanation.
        target_column: Name of the target column.
        task_type: Supervised task type, classification or regression.

    Returns:
        A structured ExplainabilityResult.
    """

    try:
        _validate_inputs(training_result, dataframe, target_column, task_type)

        pipeline = training_result.trained_pipeline
        if pipeline is None:
            raise InvalidExplainabilityInputError("training_result.trained_pipeline must not be None.")

        X = dataframe.drop(columns=[target_column])
        y = dataframe[target_column]

        scoring = "f1_weighted" if task_type == "classification" else "r2"
        importance = permutation_importance(
            pipeline,
            X,
            y,
            n_repeats=10,
            random_state=42,
            scoring=scoring,
        )

        feature_names = list(X.columns)
        importances = {feature_names[index]: float(value) for index, value in enumerate(importance.importances_mean)}
        top_features = sorted(importances.items(), key=lambda item: item[1], reverse=True)[:10]

        top_feature_names = ", ".join(feature for feature, _ in top_features[:3]) if top_features else "no features"
        explanation_summary = (
            f"Permutation importance identified {len(top_features)} influential features for the {task_type} model. "
            f"The strongest signals were: {top_feature_names}."
        )

        messages = ["Explainability completed successfully using permutation importance."]

        try:
            import shap  # type: ignore  # pragma: no cover - optional dependency

            _ = shap.__name__
        except Exception:
            pass

        return ExplainabilityResult(
            success=True,
            method_used="permutation_importance",
            feature_importance=importances,
            top_features=top_features,
            explanation_summary=explanation_summary,
            messages=messages,
            warnings=[],
            errors=[],
        )
    except ExplainabilityError as exc:
        logger.error("Explainability failed: %s", exc)
        return _failure_result(exc)
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.exception("Unexpected explainability failure")
        return _failure_result(ExplainabilityComputationError(str(exc)))


def _validate_inputs(
    training_result: TrainingResult,
    dataframe: pd.DataFrame,
    target_column: str,
    task_type: str,
) -> None:
    """Validate explainability inputs before computing importance."""

    if not isinstance(training_result, TrainingResult):
        raise InvalidExplainabilityInputError("training_result must be a TrainingResult instance.")
    if not training_result.success:
        error_details = "; ".join(training_result.errors) if training_result.errors else "unknown error"
        raise InvalidExplainabilityInputError(f"training_result.success must be True: {error_details}")
    if training_result.trained_pipeline is None:
        raise InvalidExplainabilityInputError("training_result.trained_pipeline must not be None.")
    if not isinstance(dataframe, pd.DataFrame):
        raise InvalidExplainabilityInputError("dataframe must be a pandas DataFrame.")
    if dataframe.empty:
        raise InvalidExplainabilityInputError("dataframe must not be empty.")
    if not isinstance(target_column, str) or not target_column.strip():
        raise InvalidExplainabilityInputError("target_column must be a non-empty string.")
    if target_column not in dataframe.columns:
        raise InvalidExplainabilityInputError(f"Target column not found: {target_column}")
    if task_type not in {"classification", "regression"}:
        raise InvalidExplainabilityInputError("task_type must be classification or regression.")


def _failure_result(exc: Exception) -> ExplainabilityResult:
    """Build a failed explainability result."""

    return ExplainabilityResult(
        success=False,
        method_used=None,
        feature_importance=None,
        top_features=[],
        explanation_summary=None,
        messages=[str(exc)],
        warnings=[],
        errors=[str(exc)],
    )