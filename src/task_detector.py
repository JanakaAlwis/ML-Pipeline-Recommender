"""Task Detector module for Phase 3.

This module inspects a DataFrame target column and decides whether the
supervised learning task is classification or regression.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import pandas as pd
from pandas.api.types import CategoricalDtype

logger = logging.getLogger(__name__)

PROFILE_VERSION = "1.0"
BINARY_TARGET_UNIQUE_COUNT = 2
LOW_UNIQUE_CLASSIFICATION_THRESHOLD = 20
LOW_UNIQUE_RATIO_THRESHOLD = 0.05
HIGH_UNIQUE_RATIO_REGRESSION_THRESHOLD = 0.20
HIGH_CONFIDENCE = 0.95
MEDIUM_CONFIDENCE = 0.75
LOW_CONFIDENCE = 0.60


@dataclass(slots=True)
class TaskDetectionResult:
    """Container for the outcome of a task detection run."""

    success: bool
    task_type: str | None
    confidence: float | None
    messages: list[str]
    warnings: list[str]
    errors: list[str]
    target_column: str
    target_dtype: str | None
    target_unique_count: int | None
    target_unique_ratio: float | None
    is_numeric_target: bool | None
    is_categorical_target: bool | None
    is_boolean_target: bool | None
    class_count: int | None
    detection_reason: str | None
    alternative_task_type: str | None
    alternative_confidence: float | None
    profile_version: str


class TaskDetectorError(Exception):
    """Base class for task detector errors."""


class InvalidInputError(TaskDetectorError):
    """Raised when the input is not a valid DataFrame or target column."""


class MissingTargetColumnError(TaskDetectorError):
    """Raised when the requested target column is missing."""


class EmptyTargetError(TaskDetectorError):
    """Raised when the target column has no usable values."""


class TaskDetectionComputationError(TaskDetectorError):
    """Raised when task detection cannot complete."""


class TaskDetector:
    """Detect whether a supervised task is classification or regression."""

    def detect(self, df: pd.DataFrame, target_column: str) -> TaskDetectionResult:
        """Detect the task type for a given target column.

        Args:
            df: Input DataFrame.
            target_column: Name of the target column.

        Returns:
            A TaskDetectionResult containing the decision and confidence.
        """

        try:
            validate_task_detection_input(df, target_column)
            logger.info("Starting task detection for target column: %s", target_column)

            profile_df = df.copy(deep=True)
            target_series = profile_df[target_column]
            cleaned_target = target_series.dropna()
            target_dtype = str(target_series.dtype)
            target_unique_count = int(cleaned_target.nunique(dropna=True))
            total_rows = int(len(profile_df))
            target_unique_ratio = round(target_unique_count / total_rows, 4) if total_rows else None

            is_numeric_target = bool(pd.api.types.is_numeric_dtype(target_series))
            is_boolean_target = bool(pd.api.types.is_bool_dtype(target_series))
            is_categorical_target = bool(
                pd.api.types.is_object_dtype(target_series)
                or isinstance(target_series.dtype, CategoricalDtype)
                or pd.api.types.is_string_dtype(target_series)
            )

            task_type, confidence, detection_reason, alternative_task_type, alternative_confidence = _detect_task_from_target(
                target_series=cleaned_target,
                target_dtype=target_dtype,
                target_unique_count=target_unique_count,
                target_unique_ratio=target_unique_ratio,
                is_numeric_target=is_numeric_target,
                is_boolean_target=is_boolean_target,
                is_categorical_target=is_categorical_target,
            )

            class_count = target_unique_count if task_type == "classification" else None
            messages = [f"Task detection completed: {task_type}, confidence={confidence:.2f}"]
            warnings: list[str] = []
            if alternative_task_type is not None:
                warnings.append("Borderline target detected; task confidence is reduced.")
                messages.extend(warnings)

            logger.info("Target dtype detected: %s", target_dtype)
            logger.info("Target unique count: %s", target_unique_count)
            logger.info("Target unique ratio: %s", target_unique_ratio)
            logger.info("Task detection completed: %s, confidence=%.2f", task_type, confidence)

            return _build_task_detection_result(
                success=True,
                task_type=task_type,
                confidence=confidence,
                messages=messages,
                warnings=warnings,
                errors=[],
                target_column=target_column,
                target_dtype=target_dtype,
                target_unique_count=target_unique_count,
                target_unique_ratio=target_unique_ratio,
                is_numeric_target=is_numeric_target,
                is_categorical_target=is_categorical_target,
                is_boolean_target=is_boolean_target,
                class_count=class_count,
                detection_reason=detection_reason,
                alternative_task_type=alternative_task_type,
                alternative_confidence=alternative_confidence,
                profile_version=PROFILE_VERSION,
            )
        except TaskDetectorError as exc:
            logger.error("Task detection failed: %s", exc)
            return _build_task_detection_result(
                success=False,
                task_type=None,
                confidence=None,
                messages=[str(exc)],
                warnings=[],
                errors=[str(exc)],
                target_column=target_column,
                target_dtype=None,
                target_unique_count=None,
                target_unique_ratio=None,
                is_numeric_target=None,
                is_categorical_target=None,
                is_boolean_target=None,
                class_count=None,
                detection_reason=None,
                alternative_task_type=None,
                alternative_confidence=None,
                profile_version=PROFILE_VERSION,
            )


def detect_task_type(df: pd.DataFrame, target_column: str) -> TaskDetectionResult:
    """Detect the task type using the default TaskDetector instance."""

    return TaskDetector().detect(df, target_column)


def validate_task_detection_input(df: pd.DataFrame, target_column: str) -> None:
    """Validate task detection input.

    Args:
        df: Input DataFrame.
        target_column: Name of the target column.

    Raises:
        InvalidInputError: If the input is not a DataFrame or is empty.
        MissingTargetColumnError: If the target column is missing.
        EmptyTargetError: If the target column has no usable values.
    """

    if not isinstance(df, pd.DataFrame):
        raise InvalidInputError("Input must be a pandas DataFrame.")
    if df.empty or df.shape[0] == 0 or df.shape[1] == 0:
        raise InvalidInputError("DataFrame must contain at least one row and one column.")
    if not isinstance(target_column, str) or not target_column.strip():
        raise InvalidInputError("Target column name must be a non-empty string.")
    if target_column not in df.columns:
        raise MissingTargetColumnError(f"Target column not found: {target_column}")
    if df[target_column].dropna().empty:
        raise EmptyTargetError(f"Target column has no usable values: {target_column}")


def _detect_task_from_target(
    *,
    target_series: pd.Series,
    target_dtype: str,
    target_unique_count: int,
    target_unique_ratio: float | None,
    is_numeric_target: bool,
    is_boolean_target: bool,
    is_categorical_target: bool,
) -> tuple[str, float, str, str | None, float | None]:
    """Apply rule-based task detection to the target series."""

    if is_boolean_target:
        return (
            "classification",
            HIGH_CONFIDENCE,
            "Boolean target strongly indicates classification.",
            None,
            None,
        )

    if is_categorical_target:
        return (
            "classification",
            HIGH_CONFIDENCE,
            "Categorical target strongly indicates classification.",
            None,
            None,
        )

    if not is_numeric_target:
        return (
            "classification",
            HIGH_CONFIDENCE,
            "Non-numeric target indicates classification.",
            None,
            None,
        )

    if target_unique_count == BINARY_TARGET_UNIQUE_COUNT:
        return (
            "classification",
            HIGH_CONFIDENCE,
            "Binary numeric target indicates classification.",
            None,
            None,
        )

    if target_unique_count <= LOW_UNIQUE_CLASSIFICATION_THRESHOLD:
        if target_unique_ratio is not None and target_unique_ratio <= LOW_UNIQUE_RATIO_THRESHOLD:
            return (
                "classification",
                0.85,
                "Numeric target has a low number of unique values and a low unique ratio, indicating classification.",
                None,
                None,
            )
        if target_unique_count >= LOW_UNIQUE_CLASSIFICATION_THRESHOLD - 2:
            return (
                "classification",
                LOW_CONFIDENCE,
                "Numeric target is near the classification uniqueness threshold and is treated as borderline classification.",
                "regression",
                MEDIUM_CONFIDENCE,
            )
        return (
            "classification",
            0.80,
            "Numeric target has relatively few unique values, indicating classification.",
            None,
            None,
        )

    if target_unique_ratio is not None and target_unique_ratio >= HIGH_UNIQUE_RATIO_REGRESSION_THRESHOLD:
        return (
            "regression",
            HIGH_CONFIDENCE,
            "Numeric target has many unique values and a high unique ratio, indicating regression.",
            None,
            None,
        )

    return (
        "classification",
        LOW_CONFIDENCE,
        "Numeric target is borderline between discrete and continuous behavior.",
        "regression",
        MEDIUM_CONFIDENCE,
    )


def _build_task_detection_result(
    *,
    success: bool,
    task_type: str | None,
    confidence: float | None,
    messages: list[str],
    warnings: list[str],
    errors: list[str],
    target_column: str,
    target_dtype: str | None,
    target_unique_count: int | None,
    target_unique_ratio: float | None,
    is_numeric_target: bool | None,
    is_categorical_target: bool | None,
    is_boolean_target: bool | None,
    class_count: int | None,
    detection_reason: str | None,
    alternative_task_type: str | None,
    alternative_confidence: float | None,
    profile_version: str,
) -> TaskDetectionResult:
    """Build a standardized task detection result."""

    return TaskDetectionResult(
        success=success,
        task_type=task_type,
        confidence=confidence,
        messages=messages,
        warnings=warnings,
        errors=errors,
        target_column=target_column,
        target_dtype=target_dtype,
        target_unique_count=target_unique_count,
        target_unique_ratio=target_unique_ratio,
        is_numeric_target=is_numeric_target,
        is_categorical_target=is_categorical_target,
        is_boolean_target=is_boolean_target,
        class_count=class_count,
        detection_reason=detection_reason,
        alternative_task_type=alternative_task_type,
        alternative_confidence=alternative_confidence,
        profile_version=profile_version,
    )