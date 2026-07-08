"""Tests for the Dataset Profiler module."""

from __future__ import annotations

from dataclasses import fields

import pandas as pd

from src.data_profiler import (
    DatasetProfileResult,
    EmptyDatasetError,
    InvalidDatasetError,
    MissingTargetColumnError,
    HIGH_CORRELATION_THRESHOLD,
    HIGH_CARDINALITY_THRESHOLD,
    MISSING_VALUE_WARNING_THRESHOLD,
    OUTLIER_IQR_MULTIPLIER,
    PROFILE_VERSION,
    DatasetProfiler,
    profile_dataset,
    validate_dataset_for_profiling,
)


def test_profile_result_has_expected_fields() -> None:
    """The profile result dataclass should expose the agreed fields."""

    field_names = [field.name for field in fields(DatasetProfileResult)]
    assert field_names == [
        "success",
        "dataframe",
        "messages",
        "warnings",
        "errors",
        "shape",
        "row_count",
        "column_count",
        "column_names",
        "numeric_columns",
        "categorical_columns",
        "boolean_columns",
        "datetime_columns",
        "missing_values",
        "missing_percentages",
        "duplicate_row_count",
        "duplicate_row_percentage",
        "constant_columns",
        "high_cardinality_columns",
        "sample_rows",
        "column_summary",
        "numeric_summary",
        "categorical_summary",
        "correlation_matrix",
        "high_correlation_pairs",
        "outlier_counts",
        "outlier_percentages",
        "target_column",
        "target_distribution",
        "class_imbalance_ratio",
        "profile_version",
    ]


def test_profile_dataset_success() -> None:
    """A valid DataFrame should profile successfully."""

    df = pd.DataFrame(
        {
            "age": [20, 21, 22, 23, 24],
            "income": [100, 110, 120, 130, 140],
            "city": ["A", "B", "A", "B", "A"],
            "target": [0, 0, 0, 1, 1],
        }
    )

    result = profile_dataset(df, target_column="target")

    assert result.success is True
    assert result.shape == (5, 4)
    assert result.row_count == 5
    assert result.column_count == 4
    assert result.profile_version == PROFILE_VERSION
    assert result.dataframe is not None
    assert result.target_distribution is not None


def test_profile_dataset_rejects_non_dataframe_input() -> None:
    """Non-DataFrame input should be rejected."""

    result = DatasetProfiler().profile("not a dataframe")  # type: ignore[arg-type]

    assert result.success is False
    assert any("pandas DataFrame" in error for error in result.errors)


def test_profile_dataset_rejects_empty_dataframe() -> None:
    """Empty DataFrames should be rejected."""

    df = pd.DataFrame()

    result = profile_dataset(df)

    assert result.success is False
    assert any("at least one row and one column" in error for error in result.errors)


def test_profile_dataset_rejects_missing_target_column() -> None:
    """Missing target columns should be rejected."""

    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    result = profile_dataset(df, target_column="missing")

    assert result.success is False
    assert any("Target column not found" in error for error in result.errors)


def test_profile_dataset_detects_numeric_and_categorical_columns() -> None:
    """Numeric and categorical column detection should work."""

    df = pd.DataFrame(
        {
            "num": [1, 2, 3],
            "cat": ["x", "y", "x"],
            "flag": [True, False, True],
        }
    )

    result = profile_dataset(df)

    assert result.numeric_columns == ["num"]
    assert result.categorical_columns == ["cat"]
    assert result.boolean_columns == ["flag"]


def test_profile_dataset_counts_missing_values_correctly() -> None:
    """Missing values should be counted correctly."""

    df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, 3]})

    result = profile_dataset(df)

    assert result.missing_values == {"a": 1, "b": 2}
    assert result.missing_percentages["a"] == 33.3333
    assert result.missing_percentages["b"] == 66.6667


def test_profile_dataset_missing_value_warning_threshold() -> None:
    """Columns above the missing-value warning threshold should produce a warning."""

    df = pd.DataFrame({"a": [1, None, None, None, 5], "b": [1, 2, 3, 4, 5]})

    result = profile_dataset(df)

    assert any("High missingness" in warning for warning in result.warnings)
    assert result.missing_percentages["a"] >= MISSING_VALUE_WARNING_THRESHOLD * 100.0


def test_profile_dataset_counts_duplicate_rows_correctly() -> None:
    """Duplicate rows should be counted correctly."""

    df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})

    result = profile_dataset(df)

    assert result.duplicate_row_count == 1
    assert result.duplicate_row_percentage == 33.3333


def test_profile_dataset_builds_column_summary() -> None:
    """The column summary should be populated."""

    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"]})

    result = profile_dataset(df)

    assert result.column_summary is not None
    assert set(result.column_summary["column"]) == {"a", "b"}


def test_profile_dataset_builds_numeric_summary() -> None:
    """The numeric summary should be populated for numeric columns."""

    df = pd.DataFrame({"a": [1, 2, 3], "b": [2, 4, 6]})

    result = profile_dataset(df)

    assert result.numeric_summary is not None
    assert set(result.numeric_summary["column"]) == {"a", "b"}


def test_profile_dataset_builds_categorical_summary() -> None:
    """The categorical summary should be populated for categorical columns."""

    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"]})

    result = profile_dataset(df)

    assert result.categorical_summary is not None
    assert set(result.categorical_summary["column"]) == {"b"}


def test_profile_dataset_detects_high_correlation_pairs() -> None:
    """Highly correlated numeric columns should be detected."""

    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [2, 4, 6, 8, 10],
            "z": [5, 4, 3, 2, 1],
        }
    )

    result = profile_dataset(df)

    assert any({pair[0], pair[1]} == {"x", "y"} for pair in result.high_correlation_pairs)
    assert any(abs(pair[2]) >= HIGH_CORRELATION_THRESHOLD for pair in result.high_correlation_pairs)


def test_profile_dataset_detects_high_correlation_pairs_with_missing_and_duplicates() -> None:
    """High correlation detection should still work with missing values and duplicate rows."""

    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 5],
            "y": [2, 4, None, 8, 10, 10],
            "z": [5, 4, 3, 2, 1, 1],
        }
    )

    result = profile_dataset(df)

    assert any({pair[0], pair[1]} == {"x", "y"} for pair in result.high_correlation_pairs)
    assert result.duplicate_row_count == 1


def test_profile_dataset_detects_iqr_outliers() -> None:
    """IQR outliers should be detected for obvious outlier values."""

    df = pd.DataFrame({"value": [10, 11, 12, 13, 100]})

    result = profile_dataset(df)

    assert result.outlier_counts["value"] >= 1
    assert result.outlier_percentages["value"] > 0


def test_profile_dataset_profiles_target_distribution() -> None:
    """The target distribution should be generated when a target is supplied."""

    df = pd.DataFrame({"feature": [1, 2, 3, 4], "target": [0, 0, 0, 1]})

    result = profile_dataset(df, target_column="target")

    assert result.target_distribution is not None
    assert set(result.target_distribution["target"]) == {0, 1}


def test_profile_dataset_computes_class_imbalance_ratio() -> None:
    """The class imbalance ratio should be computed for imbalanced targets."""

    df = pd.DataFrame({"feature": [1, 2, 3, 4, 5], "target": [0, 0, 0, 0, 1]})

    result = profile_dataset(df, target_column="target")

    assert result.class_imbalance_ratio == 0.25


def test_profile_dataset_returns_warnings_for_high_missingness() -> None:
    """High missingness should generate a warning message."""

    df = pd.DataFrame({"a": [1, None, None, None, 5], "b": [1, 2, 3, 4, 5]})

    result = profile_dataset(df)

    assert any("High missingness" in warning for warning in result.warnings)


def test_validate_dataset_for_profiling_accepts_valid_dataframe() -> None:
    """Validation should accept a normal DataFrame."""

    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    validate_dataset_for_profiling(df)


def test_validate_dataset_for_profiling_rejects_empty_dataframe() -> None:
    """Validation should reject empty DataFrames."""

    try:
        validate_dataset_for_profiling(pd.DataFrame())
    except EmptyDatasetError:
        assert True
    else:
        assert False


def test_profile_dataset_constant_and_high_cardinality_columns() -> None:
    """Constant and high-cardinality columns should be detected."""

    df = pd.DataFrame(
        {
            "constant": [1] * HIGH_CARDINALITY_THRESHOLD,
            "category": [f"c{i}" for i in range(HIGH_CARDINALITY_THRESHOLD)],
        }
    )

    result = profile_dataset(df)

    assert "constant" in result.constant_columns
    assert "category" in result.high_cardinality_columns
