"""Tests for the Task Detector module."""

from __future__ import annotations

from dataclasses import fields

import pandas as pd

from src.task_detector import (
    TaskDetectionResult,
    InvalidInputError,
    MissingTargetColumnError,
    EmptyTargetError,
    BINARY_TARGET_UNIQUE_COUNT,
    LOW_UNIQUE_CLASSIFICATION_THRESHOLD,
    HIGH_CONFIDENCE,
    LOW_CONFIDENCE,
    TaskDetector,
    detect_task_type,
    validate_task_detection_input,
)


def test_task_detection_result_has_expected_fields() -> None:
    """The result dataclass should expose the agreed Phase 3 fields."""

    field_names = [field.name for field in fields(TaskDetectionResult)]
    assert field_names == [
        "success",
        "task_type",
        "confidence",
        "messages",
        "warnings",
        "errors",
        "target_column",
        "target_dtype",
        "target_unique_count",
        "target_unique_ratio",
        "is_numeric_target",
        "is_categorical_target",
        "is_boolean_target",
        "class_count",
        "detection_reason",
        "alternative_task_type",
        "alternative_confidence",
        "profile_version",
    ]


def test_detect_task_type_classification_for_object_target() -> None:
    """Object targets should be classified as classification."""

    df = pd.DataFrame({"feature": [1, 2, 3], "target": ["a", "b", "a"]})

    result = detect_task_type(df, "target")

    assert result.success is True
    assert result.task_type == "classification"


def test_detect_task_type_classification_for_boolean_target() -> None:
    """Boolean targets should be classified as classification."""

    df = pd.DataFrame({"feature": [1, 2, 3], "target": [True, False, True]})

    result = detect_task_type(df, "target")

    assert result.task_type == "classification"
    assert result.confidence == HIGH_CONFIDENCE


def test_detect_task_type_classification_for_binary_numeric_target() -> None:
    """Binary numeric targets should be classified as classification."""

    df = pd.DataFrame({"feature": [1, 2, 3, 4], "target": [0, 1, 0, 1]})

    result = detect_task_type(df, "target")

    assert result.task_type == "classification"
    assert result.target_unique_count == BINARY_TARGET_UNIQUE_COUNT


def test_detect_task_type_classification_for_low_unique_numeric_target() -> None:
    """Low-unique numeric targets should be classified as classification."""

    df = pd.DataFrame({"feature": list(range(10)), "target": [0, 0, 1, 1, 2, 2, 2, 1, 0, 2]})

    result = detect_task_type(df, "target")

    assert result.task_type == "classification"
    assert result.target_unique_count == 3


def test_detect_task_type_regression_for_continuous_numeric_target() -> None:
    """Continuous numeric targets should be classified as regression."""

    df = pd.DataFrame({"feature": list(range(50)), "target": [x * 1.5 + 0.25 for x in range(50)]})

    result = detect_task_type(df, "target")

    assert result.task_type == "regression"
    assert result.confidence == HIGH_CONFIDENCE


def test_detect_task_type_rejects_non_dataframe_input() -> None:
    """Non-DataFrame input should be rejected."""

    result = TaskDetector().detect("not a dataframe", "target")  # type: ignore[arg-type]

    assert result.success is False
    assert any("pandas DataFrame" in error for error in result.errors)


def test_detect_task_type_rejects_missing_target_column() -> None:
    """Missing target columns should be rejected."""

    df = pd.DataFrame({"feature": [1, 2, 3]})

    result = detect_task_type(df, "target")

    assert result.success is False
    assert any("Target column not found" in error for error in result.errors)


def test_detect_task_type_rejects_empty_dataframe() -> None:
    """Empty DataFrames should be rejected."""

    result = detect_task_type(pd.DataFrame(), "target")

    assert result.success is False
    assert any("at least one row and one column" in error for error in result.errors)


def test_detect_task_type_returns_high_confidence_for_clear_classification() -> None:
    """Clear classification targets should yield high confidence."""

    df = pd.DataFrame({"feature": [1, 2, 3], "target": ["x", "y", "x"]})

    result = detect_task_type(df, "target")

    assert result.task_type == "classification"
    assert result.confidence == HIGH_CONFIDENCE


def test_detect_task_type_returns_high_confidence_for_clear_regression() -> None:
    """Clear regression targets should yield high confidence."""

    df = pd.DataFrame({"feature": list(range(40)), "target": [float(x) + 0.1 for x in range(40)]})

    result = detect_task_type(df, "target")

    assert result.task_type == "regression"
    assert result.confidence == HIGH_CONFIDENCE


def test_detect_task_type_handles_borderline_numeric_target() -> None:
    """Borderline numeric targets should be detected with lower confidence."""

    values = [i % 18 for i in range(100)]
    df = pd.DataFrame({"feature": list(range(100)), "target": values})

    result = detect_task_type(df, "target")

    assert result.task_type == "classification"
    assert result.confidence == LOW_CONFIDENCE
    assert result.alternative_task_type == "regression"


def test_detect_task_type_includes_detection_reason() -> None:
    """The result should include a human-readable detection reason."""

    df = pd.DataFrame({"feature": [1, 2, 3], "target": [0, 0, 1]})

    result = detect_task_type(df, "target")

    assert result.detection_reason is not None
    assert result.detection_reason != ""


def test_detect_task_type_includes_alternative_task_for_ambiguous_case() -> None:
    """Ambiguous numeric targets should include an alternative task type."""

    df = pd.DataFrame({"feature": list(range(100)), "target": [i % 18 for i in range(100)]})

    result = detect_task_type(df, "target")

    assert result.alternative_task_type == "regression"
    assert result.alternative_confidence is not None


def test_validate_task_detection_input_accepts_valid_dataframe() -> None:
    """Validation should accept a normal DataFrame and target."""

    df = pd.DataFrame({"feature": [1, 2, 3], "target": [0, 1, 0]})

    validate_task_detection_input(df, "target")


def test_validate_task_detection_input_rejects_empty_target() -> None:
    """Validation should reject target columns with no usable values."""

    df = pd.DataFrame({"feature": [1, 2, 3], "target": [None, None, None]})

    try:
        validate_task_detection_input(df, "target")
    except EmptyTargetError:
        assert True
    else:
        assert False
