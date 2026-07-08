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

MAX_PERMUTATION_ROWS = 5000
MAX_PERMUTATION_REPEATS = 5
TOP_FEATURE_COUNT = 10
SHAP_DEFAULT_SAMPLE_SIZE = 500
SHAP_MAX_SAMPLE_SIZE = 1000


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


@dataclass(slots=True)
class ShapExplanationResult:
    """Structured result of an optional SHAP explanation run."""

    success: bool
    method_used: str | None
    sample_size: int | None
    top_features: list[tuple[str, float]]
    explanation_summary: str | None
    interpretation: str | None
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
        X_sample, y_sample, sample_message = _sample_for_permutation_importance(X, y)

        scoring = "f1_weighted" if task_type == "classification" else "r2"
        repeats = MAX_PERMUTATION_REPEATS if len(X_sample) > 1000 or X_sample.shape[1] > 20 else 10
        importance = permutation_importance(
            pipeline,
            X_sample,
            y_sample,
            n_repeats=repeats,
            random_state=42,
            scoring=scoring,
            n_jobs=-1,
        )

        feature_names = list(X.columns)
        importances = {feature_names[index]: float(value) for index, value in enumerate(importance.importances_mean)}
        top_features = sorted(importances.items(), key=lambda item: item[1], reverse=True)[:TOP_FEATURE_COUNT]

        top_feature_names = ", ".join(feature for feature, _ in top_features[:3]) if top_features else "no features"
        explanation_summary = (
            f"Permutation importance identified the top {len(top_features)} features for the {task_type} model. "
            f"The strongest signals were: {top_feature_names}."
        )

        messages = ["Explainability completed successfully using sampled permutation importance."]
        if sample_message is not None:
            messages.append(sample_message)

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


def generate_shap_explanation(
    training_result: TrainingResult,
    dataframe: pd.DataFrame,
    target_column: str,
    task_type: str,
    sample_size: int = SHAP_DEFAULT_SAMPLE_SIZE,
) -> ShapExplanationResult:
    """Generate an optional SHAP explanation for supported tree-based models."""

    try:
        _validate_inputs(training_result, dataframe, target_column, task_type)

        pipeline = training_result.trained_pipeline
        if pipeline is None:
            raise InvalidExplainabilityInputError("training_result.trained_pipeline must not be None.")

        model, preprocessor = _extract_model_and_preprocessor(pipeline)
        if model is None or preprocessor is None:
            raise InvalidExplainabilityInputError("The trained pipeline does not expose a compatible model and preprocessor.")

        if not _is_tree_based_model(model):
            raise InvalidExplainabilityInputError(
                "SHAP explanation is not available for this model. Feature importance is shown instead."
            )

        import shap  # type: ignore  # pragma: no cover - optional dependency

        feature_frame = dataframe.drop(columns=[target_column])
        sample_size = max(1, min(int(sample_size), SHAP_MAX_SAMPLE_SIZE))
        sample_frame = _sample_frame(feature_frame, sample_size)
        sample_size_used = len(sample_frame)

        transformed_sample = preprocessor.transform(sample_frame)
        if hasattr(transformed_sample, "toarray"):
            transformed_sample = transformed_sample.toarray()

        feature_names = _get_feature_names(preprocessor, sample_frame.columns.tolist())
        if not feature_names:
            feature_names = [f"feature_{index + 1}" for index in range(transformed_sample.shape[1])]

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(transformed_sample)
        mean_abs_values = _mean_absolute_shap_values(shap_values)

        top_features = _top_feature_pairs(feature_names, mean_abs_values, TOP_FEATURE_COUNT)
        top_names = ", ".join(feature for feature, _ in top_features[:3]) if top_features else "no features"
        summary = f"SHAP values were calculated on a sample of {sample_size_used} rows using TreeExplainer."
        interpretation = (
            f"The most influential features in this sample were {top_names}. "
            "These SHAP values show how much each feature contributes to the model predictions in the sampled data."
        )

        return ShapExplanationResult(
            success=True,
            method_used="tree_explainer",
            sample_size=sample_size_used,
            top_features=top_features,
            explanation_summary=summary,
            interpretation=interpretation,
            messages=["SHAP explanation generated successfully."],
            warnings=[
                "SHAP explanations may take time for large datasets. A sample of the data was used."
            ],
            errors=[],
        )
    except InvalidExplainabilityInputError as exc:
        return ShapExplanationResult(
            success=False,
            method_used=None,
            sample_size=None,
            top_features=[],
            explanation_summary=None,
            interpretation=None,
            messages=[str(exc)],
            warnings=[],
            errors=[str(exc)],
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.exception("Unexpected SHAP explanation failure")
        return ShapExplanationResult(
            success=False,
            method_used=None,
            sample_size=None,
            top_features=[],
            explanation_summary=None,
            interpretation=None,
            messages=[str(exc)],
            warnings=[],
            errors=[str(exc)],
        )


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


def _sample_for_permutation_importance(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, str | None]:
    """Sample the input data for permutation importance when the dataset is large."""

    if len(features) <= MAX_PERMUTATION_ROWS:
        return features, target, None

    sample_frame = features.sample(n=MAX_PERMUTATION_ROWS, random_state=42)
    sampled_target = target.loc[sample_frame.index]
    return sample_frame, sampled_target, f"Permutation importance used a sample of {MAX_PERMUTATION_ROWS} rows for performance."


def _extract_model_and_preprocessor(pipeline: Any) -> tuple[Any | None, Any | None]:
    """Extract the final model and preprocessor from a fitted pipeline."""

    named_steps = getattr(pipeline, "named_steps", None)
    if not isinstance(named_steps, dict):
        return None, None
    return named_steps.get("model"), named_steps.get("preprocessor")


def _is_tree_based_model(model: Any) -> bool:
    """Check whether the final estimator is a tree-based model supported by TreeExplainer."""

    model_name = model.__class__.__name__.lower()
    return any(keyword in model_name for keyword in ["decisiontree", "randomforest", "xgb", "gradientboosting", "extraTrees".lower()])


def _sample_frame(dataframe: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    """Return a bounded sample of the dataframe for SHAP."""

    if len(dataframe) <= sample_size:
        return dataframe
    return dataframe.sample(n=sample_size, random_state=42)


def _get_feature_names(preprocessor: Any, fallback_columns: list[str]) -> list[str]:
    """Resolve transformed feature names when the preprocessor supports them."""

    try:
        feature_names = list(preprocessor.get_feature_names_out())
        if feature_names:
            return feature_names
    except Exception:
        pass
    return list(fallback_columns)


def _mean_absolute_shap_values(shap_values: Any) -> np.ndarray:
    """Convert SHAP output into a 1D mean absolute importance vector."""

    if isinstance(shap_values, list):
        stacked = np.stack([np.asarray(values) for values in shap_values], axis=0)
        return np.mean(np.abs(stacked), axis=(0, 1))

    values = np.asarray(shap_values)
    if values.ndim == 3:
        return np.mean(np.abs(values), axis=(0, 1))
    return np.mean(np.abs(values), axis=0)


def _top_feature_pairs(feature_names: list[str], scores: np.ndarray, limit: int) -> list[tuple[str, float]]:
    """Create a ranked feature list from importance scores."""

    paired_scores = [(feature_names[index], float(score)) for index, score in enumerate(np.asarray(scores).ravel())]
    return sorted(paired_scores, key=lambda item: item[1], reverse=True)[:limit]