"""Dataset Profiler module for Phase 2.

This module analyzes a pandas DataFrame returned by the dataset loader and
produces a structured profiling result for later task detection and pipeline
recommendation phases.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

PROFILE_VERSION = "1.0"
MISSING_VALUE_WARNING_THRESHOLD = 0.20
HIGH_CARDINALITY_THRESHOLD = 50
HIGH_CORRELATION_THRESHOLD = 0.80
OUTLIER_IQR_MULTIPLIER = 1.5
LOW_UNIQUE_CLASSIFICATION_THRESHOLD = 20
IMBALANCE_RATIO_WARNING_THRESHOLD = 0.20
SAMPLE_ROW_COUNT = 5


@dataclass(slots=True)
class DatasetProfileResult:
    """Container for the outcome of a dataset profiling run."""

    success: bool
    dataframe: pd.DataFrame | None
    messages: list[str]
    warnings: list[str]
    errors: list[str]
    shape: tuple[int, int] | None
    row_count: int | None
    column_count: int | None
    column_names: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    boolean_columns: list[str]
    datetime_columns: list[str]
    missing_values: dict[str, int]
    missing_percentages: dict[str, float]
    duplicate_row_count: int | None
    duplicate_row_percentage: float | None
    constant_columns: list[str]
    high_cardinality_columns: list[str]
    sample_rows: pd.DataFrame | None
    column_summary: pd.DataFrame | None
    numeric_summary: pd.DataFrame | None
    categorical_summary: pd.DataFrame | None
    correlation_matrix: pd.DataFrame | None
    high_correlation_pairs: list[tuple[str, str, float]]
    outlier_counts: dict[str, int]
    outlier_percentages: dict[str, float]
    target_column: str | None
    target_distribution: pd.DataFrame | None
    class_imbalance_ratio: float | None
    profile_version: str


class DatasetProfilerError(Exception):
    """Base class for dataset profiler errors."""


class InvalidDatasetError(DatasetProfilerError):
    """Raised when the input is not a valid DataFrame."""


class EmptyDatasetError(DatasetProfilerError):
    """Raised when the DataFrame has no rows or columns."""


class MissingTargetColumnError(DatasetProfilerError):
    """Raised when a requested target column is not present."""


class ProfilingComputationError(DatasetProfilerError):
    """Raised when profiling statistics cannot be computed."""


class DatasetProfiler:
    """Profile a tabular dataset represented as a pandas DataFrame."""

    def profile(
        self,
        df: pd.DataFrame,
        target_column: str | None = None,
    ) -> DatasetProfileResult:
        """Profile a DataFrame and return a structured result.

        Args:
            df: Input DataFrame to analyze.
            target_column: Optional target column name.

        Returns:
            A DatasetProfileResult containing dataset statistics.
        """

        try:
            validate_dataset_for_profiling(df, target_column)
            logger.info("Starting dataset profiling")

            profile_df = df.copy(deep=True)
            shape = profile_df.shape
            row_count, column_count = shape

            numeric_columns, categorical_columns, boolean_columns, datetime_columns = _split_column_types(
                profile_df
            )
            missing_values, missing_percentages = _calculate_missing_values(profile_df)
            duplicate_row_count, duplicate_row_percentage = _calculate_duplicate_rows(profile_df)
            constant_columns = _find_constant_columns(profile_df)
            high_cardinality_columns = _find_high_cardinality_columns(profile_df, categorical_columns)
            column_summary = _build_column_summary(profile_df)
            numeric_summary = _build_numeric_summary(profile_df, numeric_columns)
            categorical_summary = _build_categorical_summary(profile_df, categorical_columns)
            correlation_matrix = _calculate_correlation_matrix(profile_df, numeric_columns)
            high_correlation_pairs = _find_high_correlation_pairs(
                correlation_matrix, HIGH_CORRELATION_THRESHOLD
            )
            outlier_counts, outlier_percentages = _calculate_outliers_iqr(
                profile_df, numeric_columns
            )
            target_distribution, class_imbalance_ratio = _profile_target_column(
                profile_df, target_column
            )
            sample_rows = profile_df.head(SAMPLE_ROW_COUNT).copy(deep=True)

            messages = ["Dataset profiling completed successfully"]
            warnings = _build_warnings(
                missing_percentages=missing_percentages,
                duplicate_row_percentage=duplicate_row_percentage,
                high_cardinality_columns=high_cardinality_columns,
                high_correlation_pairs=high_correlation_pairs,
                class_imbalance_ratio=class_imbalance_ratio,
            )
            messages.extend(warnings)

            logger.info(
                "Dataset shape: rows=%s, columns=%s",
                row_count,
                column_count,
            )
            logger.info(
                "Detected column types: numeric=%s, categorical=%s, boolean=%s, datetime=%s",
                len(numeric_columns),
                len(categorical_columns),
                len(boolean_columns),
                len(datetime_columns),
            )
            logger.info("Missing values found in %s columns", len([v for v in missing_values.values() if v > 0]))
            logger.info("Duplicate rows detected: %s", duplicate_row_count)
            logger.info("High correlation pairs detected: %s", len(high_correlation_pairs))
            logger.info("Dataset profiling completed successfully")

            return _build_profile_result(
                success=True,
                dataframe=profile_df,
                messages=messages,
                warnings=warnings,
                errors=[],
                shape=shape,
                row_count=row_count,
                column_count=column_count,
                column_names=list(profile_df.columns),
                numeric_columns=numeric_columns,
                categorical_columns=categorical_columns,
                boolean_columns=boolean_columns,
                datetime_columns=datetime_columns,
                missing_values=missing_values,
                missing_percentages=missing_percentages,
                duplicate_row_count=duplicate_row_count,
                duplicate_row_percentage=duplicate_row_percentage,
                constant_columns=constant_columns,
                high_cardinality_columns=high_cardinality_columns,
                sample_rows=sample_rows,
                column_summary=column_summary,
                numeric_summary=numeric_summary,
                categorical_summary=categorical_summary,
                correlation_matrix=correlation_matrix,
                high_correlation_pairs=high_correlation_pairs,
                outlier_counts=outlier_counts,
                outlier_percentages=outlier_percentages,
                target_column=target_column,
                target_distribution=target_distribution,
                class_imbalance_ratio=class_imbalance_ratio,
                profile_version=PROFILE_VERSION,
            )
        except DatasetProfilerError as exc:
            logger.error("Dataset profiling failed: %s", exc)
            return _build_profile_result(
                success=False,
                dataframe=None,
                messages=[str(exc)],
                warnings=[],
                errors=[str(exc)],
                shape=None,
                row_count=None,
                column_count=None,
                column_names=[],
                numeric_columns=[],
                categorical_columns=[],
                boolean_columns=[],
                datetime_columns=[],
                missing_values={},
                missing_percentages={},
                duplicate_row_count=None,
                duplicate_row_percentage=None,
                constant_columns=[],
                high_cardinality_columns=[],
                sample_rows=None,
                column_summary=None,
                numeric_summary=None,
                categorical_summary=None,
                correlation_matrix=None,
                high_correlation_pairs=[],
                outlier_counts={},
                outlier_percentages={},
                target_column=target_column,
                target_distribution=None,
                class_imbalance_ratio=None,
                profile_version=PROFILE_VERSION,
            )


def profile_dataset(df: pd.DataFrame, target_column: str | None = None) -> DatasetProfileResult:
    """Profile a DataFrame using the default DatasetProfiler instance."""

    return DatasetProfiler().profile(df, target_column=target_column)


def validate_dataset_for_profiling(
    df: pd.DataFrame,
    target_column: str | None = None,
) -> None:
    """Validate that the DataFrame can be profiled.

    Args:
        df: Input DataFrame.
        target_column: Optional target column name.

    Raises:
        InvalidDatasetError: If the input is not a DataFrame.
        EmptyDatasetError: If the DataFrame is empty.
        MissingTargetColumnError: If a target column is not present.
    """

    if not isinstance(df, pd.DataFrame):
        raise InvalidDatasetError("Input must be a pandas DataFrame.")
    if df.empty or df.shape[0] == 0 or df.shape[1] == 0:
        raise EmptyDatasetError("DataFrame must contain at least one row and one column.")
    if target_column is not None and target_column not in df.columns:
        raise MissingTargetColumnError(f"Target column not found: {target_column}")


def _split_column_types(
    df: pd.DataFrame,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Split DataFrame columns into numeric, categorical, boolean, and datetime groups."""

    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    boolean_columns: list[str] = []
    datetime_columns: list[str] = []

    for column in df.columns:
        series = df[column]
        if pd.api.types.is_bool_dtype(series):
            boolean_columns.append(column)
        elif pd.api.types.is_datetime64_any_dtype(series):
            datetime_columns.append(column)
        elif pd.api.types.is_numeric_dtype(series):
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)

    return numeric_columns, categorical_columns, boolean_columns, datetime_columns


def _calculate_missing_values(df: pd.DataFrame) -> tuple[dict[str, int], dict[str, float]]:
    """Calculate missing value counts and percentages per column."""

    missing_counts = df.isna().sum().to_dict()
    total_rows = len(df)
    if total_rows == 0:
        return {column: 0 for column in df.columns}, {column: 0.0 for column in df.columns}

    missing_percentages = {
        column: round((missing_counts[column] / total_rows) * 100.0, 4)
        for column in df.columns
    }
    return missing_counts, missing_percentages


def _calculate_duplicate_rows(df: pd.DataFrame) -> tuple[int, float]:
    """Calculate duplicate row count and percentage."""

    duplicate_row_count = int(df.duplicated().sum())
    if len(df) == 0:
        return duplicate_row_count, 0.0
    duplicate_row_percentage = round((duplicate_row_count / len(df)) * 100.0, 4)
    return duplicate_row_count, duplicate_row_percentage


def _find_constant_columns(df: pd.DataFrame) -> list[str]:
    """Return columns that contain a single unique non-null value."""

    constant_columns: list[str] = []
    for column in df.columns:
        non_null_unique_count = df[column].dropna().nunique()
        if non_null_unique_count == 1:
            constant_columns.append(column)
    return constant_columns


def _find_high_cardinality_columns(
    df: pd.DataFrame,
    categorical_columns: list[str],
) -> list[str]:
    """Return categorical columns with many unique values."""

    high_cardinality_columns: list[str] = []
    for column in categorical_columns:
        unique_count = int(df[column].nunique(dropna=True))
        if unique_count >= HIGH_CARDINALITY_THRESHOLD:
            high_cardinality_columns.append(column)
    return high_cardinality_columns


def _build_column_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build a per-column summary DataFrame."""

    rows: list[dict[str, Any]] = []
    total_rows = len(df)
    for column in df.columns:
        series = df[column]
        missing_count = int(series.isna().sum())
        row: dict[str, Any] = {
            "column": column,
            "dtype": str(series.dtype),
            "non_null_count": int(series.notna().sum()),
            "null_count": missing_count,
            "missing_percentage": round((missing_count / total_rows) * 100.0, 4) if total_rows else 0.0,
            "unique_count": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series):
            row.update(
                {
                    "min": float(series.min()) if series.notna().any() else None,
                    "max": float(series.max()) if series.notna().any() else None,
                    "mean": float(series.mean()) if series.notna().any() else None,
                    "median": float(series.median()) if series.notna().any() else None,
                    "std": float(series.std()) if series.notna().sum() > 1 else None,
                }
            )
        else:
            mode_series = series.mode(dropna=True)
            row.update(
                {
                    "top_value": mode_series.iloc[0] if not mode_series.empty else None,
                    "top_frequency": int(series.value_counts(dropna=True).iloc[0]) if not series.dropna().empty else None,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _build_numeric_summary(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    """Build a numeric summary DataFrame."""

    if not numeric_columns:
        return pd.DataFrame()

    summary = df[numeric_columns].describe().transpose().reset_index().rename(columns={"index": "column"})
    if "50%" in summary.columns:
        summary = summary.rename(columns={"50%": "median"})
    return summary


def _build_categorical_summary(df: pd.DataFrame, categorical_columns: list[str]) -> pd.DataFrame:
    """Build a categorical summary DataFrame."""

    rows: list[dict[str, Any]] = []
    for column in categorical_columns:
        series = df[column]
        value_counts = series.value_counts(dropna=True)
        rows.append(
            {
                "column": column,
                "unique_count": int(series.nunique(dropna=True)),
                "top_value": value_counts.index[0] if not value_counts.empty else None,
                "top_frequency": int(value_counts.iloc[0]) if not value_counts.empty else None,
            }
        )
    return pd.DataFrame(rows)


def _calculate_correlation_matrix(
    df: pd.DataFrame,
    numeric_columns: list[str],
) -> pd.DataFrame | None:
    """Calculate the numeric correlation matrix."""

    if len(numeric_columns) < 2:
        return None
    return df[numeric_columns].corr(numeric_only=True)


def _find_high_correlation_pairs(
    correlation_matrix: pd.DataFrame | None,
    threshold: float,
) -> list[tuple[str, str, float]]:
    """Find high absolute correlation pairs above a threshold."""

    if correlation_matrix is None or correlation_matrix.empty:
        return []

    pairs: list[tuple[str, str, float]] = []
    columns = list(correlation_matrix.columns)
    for index, column_a in enumerate(columns):
        for column_b in columns[index + 1 :]:
            correlation_value = correlation_matrix.loc[column_a, column_b]
            if pd.notna(correlation_value) and abs(float(correlation_value)) >= threshold:
                pairs.append((column_a, column_b, float(correlation_value)))
    return pairs


def _calculate_outliers_iqr(
    df: pd.DataFrame,
    numeric_columns: list[str],
) -> tuple[dict[str, int], dict[str, float]]:
    """Calculate IQR-based outlier counts and percentages for numeric columns."""

    outlier_counts: dict[str, int] = {}
    outlier_percentages: dict[str, float] = {}
    total_rows = len(df)

    for column in numeric_columns:
        series = df[column].dropna()
        if series.empty:
            outlier_counts[column] = 0
            outlier_percentages[column] = 0.0
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - OUTLIER_IQR_MULTIPLIER * iqr
        upper_bound = q3 + OUTLIER_IQR_MULTIPLIER * iqr
        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outlier_count = int(outlier_mask.sum())
        outlier_counts[column] = outlier_count
        outlier_percentages[column] = round((outlier_count / total_rows) * 100.0, 4) if total_rows else 0.0

    return outlier_counts, outlier_percentages


def _profile_target_column(
    df: pd.DataFrame,
    target_column: str | None,
) -> tuple[pd.DataFrame | None, float | None]:
    """Profile the target column when one is provided."""

    if target_column is None:
        return None, None

    series = df[target_column]
    distribution = series.value_counts(dropna=False).reset_index()
    distribution.columns = [target_column, "count"]
    total = int(distribution["count"].sum())
    if total == 0:
        return distribution, None

    if distribution.shape[0] <= 1:
        return distribution, None

    max_count = int(distribution["count"].max())
    min_non_zero = int(distribution["count"].replace(0, pd.NA).dropna().min())
    if min_non_zero == 0:
        return distribution, None

    imbalance_ratio = round(min_non_zero / max_count, 4)
    return distribution, imbalance_ratio


def _build_warnings(
    *,
    missing_percentages: dict[str, float],
    duplicate_row_percentage: float | None,
    high_cardinality_columns: list[str],
    high_correlation_pairs: list[tuple[str, str, float]],
    class_imbalance_ratio: float | None,
) -> list[str]:
    """Build user-facing warning messages from profiling statistics."""

    warnings: list[str] = []
    high_missing_columns = [
        column
        for column, percentage in missing_percentages.items()
        if percentage / 100.0 >= MISSING_VALUE_WARNING_THRESHOLD
    ]
    if high_missing_columns:
        warnings.append(
            f"High missingness detected in {len(high_missing_columns)} column(s)."
        )
    if duplicate_row_percentage is not None and duplicate_row_percentage > 0:
        warnings.append("Duplicate rows detected in the dataset.")
    if high_cardinality_columns:
        warnings.append("High-cardinality categorical columns detected.")
    if high_correlation_pairs:
        warnings.append("Strongly correlated numeric features detected.")
    if class_imbalance_ratio is not None and class_imbalance_ratio < IMBALANCE_RATIO_WARNING_THRESHOLD:
        warnings.append("Target classes appear imbalanced.")
    return warnings


def _build_profile_result(
    *,
    success: bool,
    dataframe: pd.DataFrame | None,
    messages: list[str],
    warnings: list[str],
    errors: list[str],
    shape: tuple[int, int] | None,
    row_count: int | None,
    column_count: int | None,
    column_names: list[str],
    numeric_columns: list[str],
    categorical_columns: list[str],
    boolean_columns: list[str],
    datetime_columns: list[str],
    missing_values: dict[str, int],
    missing_percentages: dict[str, float],
    duplicate_row_count: int | None,
    duplicate_row_percentage: float | None,
    constant_columns: list[str],
    high_cardinality_columns: list[str],
    sample_rows: pd.DataFrame | None,
    column_summary: pd.DataFrame | None,
    numeric_summary: pd.DataFrame | None,
    categorical_summary: pd.DataFrame | None,
    correlation_matrix: pd.DataFrame | None,
    high_correlation_pairs: list[tuple[str, str, float]],
    outlier_counts: dict[str, int],
    outlier_percentages: dict[str, float],
    target_column: str | None,
    target_distribution: pd.DataFrame | None,
    class_imbalance_ratio: float | None,
    profile_version: str,
) -> DatasetProfileResult:
    """Build a standardized dataset profile result."""

    return DatasetProfileResult(
        success=success,
        dataframe=dataframe,
        messages=messages,
        warnings=warnings,
        errors=errors,
        shape=shape,
        row_count=row_count,
        column_count=column_count,
        column_names=column_names,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        boolean_columns=boolean_columns,
        datetime_columns=datetime_columns,
        missing_values=missing_values,
        missing_percentages=missing_percentages,
        duplicate_row_count=duplicate_row_count,
        duplicate_row_percentage=duplicate_row_percentage,
        constant_columns=constant_columns,
        high_cardinality_columns=high_cardinality_columns,
        sample_rows=sample_rows,
        column_summary=column_summary,
        numeric_summary=numeric_summary,
        categorical_summary=categorical_summary,
        correlation_matrix=correlation_matrix,
        high_correlation_pairs=high_correlation_pairs,
        outlier_counts=outlier_counts,
        outlier_percentages=outlier_percentages,
        target_column=target_column,
        target_distribution=target_distribution,
        class_imbalance_ratio=class_imbalance_ratio,
        profile_version=profile_version,
    )