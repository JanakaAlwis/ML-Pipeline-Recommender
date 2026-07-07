"""Tests for the shared missing-value analysis helpers."""

from __future__ import annotations

import pandas as pd

from src.data_profiler import DatasetProfileResult
from src.missing_value_analysis import (
    build_high_missingness_warnings,
    build_missing_value_analysis_frame,
    build_missing_value_overview,
)


def _profile_with_missing_values() -> DatasetProfileResult:
    """Create a profile result with missing values for analysis tests."""

    dataframe = pd.DataFrame(
        {
            "age": [20, None, 40, 50],
            "city": ["A", "B", None, "B"],
            "target": [0, 1, 0, 1],
        }
    )
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
        numeric_columns=["age"],
        categorical_columns=["city", "target"],
        boolean_columns=[],
        datetime_columns=[],
        missing_values={"age": 1, "city": 1, "target": 0},
        missing_percentages={"age": 25.0, "city": 25.0, "target": 0.0},
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
        outlier_counts={"age": 0},
        outlier_percentages={"age": 0.0},
        target_column="target",
        target_distribution=None,
        class_imbalance_ratio=0.5,
        profile_version="1.0",
    )


def test_build_missing_value_analysis_frame_includes_recommended_handling() -> None:
    """The analysis table should describe the dtype and handling for each missing column."""

    profile_result = _profile_with_missing_values()

    frame = build_missing_value_analysis_frame(profile_result)

    assert list(frame["Column"]) == ["age", "city"]
    assert list(frame["Recommended handling"]) == ["Median imputation", "Most frequent imputation"]
    assert list(frame["Review note"]) == ["Review recommended due to high missingness.", "Review recommended due to high missingness."]


def test_build_high_missingness_warnings_for_columns_above_fifty_percent() -> None:
    """Columns with >50% missingness should include a recommendation warning."""

    profile_result = _profile_with_missing_values()
    profile_result.missing_values["city"] = 3
    profile_result.missing_percentages["city"] = 75.0

    warnings = build_high_missingness_warnings(profile_result)

    assert len(warnings) == 1
    assert "city contains 75.00% missing values" in warnings[0].lower()
    assert "most frequent imputation" in warnings[0].lower()
    assert "review whether retaining this feature is appropriate" in warnings[0].lower()


def test_build_missing_value_overview_reports_strategies() -> None:
    """The overview should summarize the detected missing-value handling strategy."""

    profile_result = _profile_with_missing_values()

    overview = build_missing_value_overview(profile_result)

    assert overview["has_missing_values"] is True
    assert overview["total_missing_values"] == 2
    assert overview["affected_column_count"] == 2
    assert overview["numerical_strategy"] == "Median imputation"
    assert overview["categorical_strategy"] == "Most frequent imputation"
    assert "median imputation" in overview["strategy_summary"].lower()
    assert "most frequent imputation" in overview["explanation"].lower()


def test_build_missing_value_overview_reports_no_handling_when_clean() -> None:
    """A clean profile should report that no missing value handling is required."""

    dataframe = pd.DataFrame({"age": [20, 30], "target": [0, 1]})
    profile_result = DatasetProfileResult(
        success=True,
        dataframe=dataframe,
        messages=["ok"],
        warnings=[],
        errors=[],
        shape=dataframe.shape,
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        column_names=list(dataframe.columns),
        numeric_columns=["age"],
        categorical_columns=["target"],
        boolean_columns=[],
        datetime_columns=[],
        missing_values={"age": 0, "target": 0},
        missing_percentages={"age": 0.0, "target": 0.0},
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
        outlier_counts={"age": 0},
        outlier_percentages={"age": 0.0},
        target_column="target",
        target_distribution=None,
        class_imbalance_ratio=0.5,
        profile_version="1.0",
    )

    overview = build_missing_value_overview(profile_result)

    assert overview["has_missing_values"] is False
    assert overview["strategy_summary"] == "No missing values were detected. No missing value handling was required."