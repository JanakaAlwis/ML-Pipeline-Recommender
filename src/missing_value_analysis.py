"""Shared missing-value analysis helpers for the thesis application."""

from __future__ import annotations

from typing import Any

import pandas as pd


REVIEW_MISSINGNESS_THRESHOLD = 20.0
HIGH_MISSINGNESS_REVIEW_THRESHOLD = 50.0


def build_missing_value_analysis_frame(profile_result: Any, *, target_column: str | None = None) -> pd.DataFrame:
    """Build a clean table describing missing-value status and actions."""

    expected_columns = [
        "Column",
        "Data type",
        "Missing values",
        "Missing percentage",
        "Recommended action",
        "Reason",
        "Recommended handling",
        "Applied action during preprocessing",
        "Review note",
    ]

    dataframe = getattr(profile_result, "dataframe", None)
    missing_values = getattr(profile_result, "missing_values", {}) or {}
    missing_percentages = getattr(profile_result, "missing_percentages", {}) or {}
    if dataframe is None or not missing_values:
        return pd.DataFrame(columns=expected_columns)

    numeric_columns = set(getattr(profile_result, "numeric_columns", []) or [])
    categorical_columns = set(getattr(profile_result, "categorical_columns", []) or [])
    boolean_columns = set(getattr(profile_result, "boolean_columns", []) or [])

    rows: list[dict[str, Any]] = []
    for column, count in missing_values.items():
        missing_count = int(count or 0)
        if missing_count <= 0:
            continue

        percentage = float(missing_percentages.get(column, 0.0) or 0.0)
        recommended_action, reason = _recommended_action(column, percentage, target_column)
        handling = _recommended_handling(column, numeric_columns, categorical_columns, boolean_columns)
        rows.append(
            {
                "Column": column,
                "Data type": _column_data_type(dataframe, column),
                "Missing values": missing_count,
                "Missing percentage": f"{percentage:.2f}%",
                "Recommended action": recommended_action,
                "Reason": reason,
                "Recommended handling": handling,
                "Applied action during preprocessing": _applied_action(recommended_action, handling),
                "Review note": _review_note(percentage),
            }
        )

    if not rows:
        return pd.DataFrame(columns=expected_columns)
    return pd.DataFrame(rows, columns=expected_columns)


def build_missing_value_overview(profile_result: Any) -> dict[str, Any]:
    """Build compact missing-value summary fields for pages and reports."""

    target_column = getattr(profile_result, "target_column", None)
    analysis_frame = build_missing_value_analysis_frame(profile_result, target_column=target_column)
    total_missing_values = int(sum(int(value or 0) for value in getattr(profile_result, "missing_values", {}).values()))
    target_missing_values = int((getattr(profile_result, "missing_values", {}) or {}).get(target_column, 0)) if target_column else 0

    feature_analysis = analysis_frame.copy()
    if target_column and not feature_analysis.empty:
        feature_analysis = feature_analysis[feature_analysis["Column"] != target_column]

    drop_feature_columns = (
        feature_analysis.loc[feature_analysis["Recommended action"] == "Drop Feature", "Column"].tolist()
        if not feature_analysis.empty
        else []
    )
    review_feature_columns = (
        feature_analysis.loc[feature_analysis["Recommended action"] == "Review", "Column"].tolist()
        if not feature_analysis.empty
        else []
    )
    imputed_feature_columns = (
        feature_analysis.loc[feature_analysis["Recommended action"].isin(["Impute", "Review"]), "Column"].tolist()
        if not feature_analysis.empty
        else []
    )

    if analysis_frame.empty:
        return {
            "has_missing_values": False,
            "total_missing_values": 0,
            "affected_column_count": 0,
            "analysis_frame": analysis_frame,
            "feature_analysis_frame": feature_analysis,
            "columns": [],
            "drop_feature_columns": [],
            "review_feature_columns": [],
            "imputed_feature_columns": [],
            "target_missing_values": target_missing_values,
            "numerical_strategy": "Not required",
            "categorical_strategy": "Not required",
            "boolean_strategy": "Not required",
            "strategy_summary": "No missing values were detected. No missing value handling was required.",
            "explanation": "No missing values were detected. No missing value handling was required.",
        }

    columns = analysis_frame["Column"].tolist()
    has_numeric = any(column in set(getattr(profile_result, "numeric_columns", []) or []) for column in columns)
    has_categorical = any(column in set(getattr(profile_result, "categorical_columns", []) or []) for column in columns)
    has_boolean = any(column in set(getattr(profile_result, "boolean_columns", []) or []) for column in columns)

    numerical_strategy = "Median imputation" if has_numeric else "Not required"
    categorical_strategy = "Most frequent imputation" if has_categorical else "Not required"
    boolean_strategy = "Most frequent imputation" if has_boolean else "Not required"

    explanation_parts = [
        "Median imputation is used for numeric columns because it is robust to outliers.",
        "Most frequent imputation is used for categorical and boolean columns because it preserves the most common observed value.",
    ]
    if review_feature_columns:
        explanation_parts.append("Columns with high missing percentages should be reviewed before modeling.")
    if drop_feature_columns:
        explanation_parts.append("Columns above 50% missingness are recommended to be dropped before preprocessing.")

    strategy_summary = _strategy_summary(numerical_strategy, categorical_strategy, boolean_strategy)
    if drop_feature_columns:
        strategy_summary = f"{strategy_summary} Features recommended for dropping: {', '.join(drop_feature_columns)}."

    return {
        "has_missing_values": True,
        "total_missing_values": total_missing_values,
        "affected_column_count": int(len(analysis_frame)),
        "analysis_frame": analysis_frame,
        "feature_analysis_frame": feature_analysis,
        "columns": columns,
        "drop_feature_columns": drop_feature_columns,
        "review_feature_columns": review_feature_columns,
        "imputed_feature_columns": imputed_feature_columns,
        "target_missing_values": target_missing_values,
        "numerical_strategy": numerical_strategy,
        "categorical_strategy": categorical_strategy,
        "boolean_strategy": boolean_strategy,
        "strategy_summary": strategy_summary,
        "explanation": " ".join(explanation_parts),
    }


def build_high_missingness_warnings(profile_result: Any) -> list[str]:
    """Build explicit warnings for columns with very high missingness (>50%)."""

    analysis_frame = build_missing_value_analysis_frame(profile_result, target_column=getattr(profile_result, "target_column", None))
    if analysis_frame.empty:
        return []

    warnings: list[str] = []
    for _, row in analysis_frame.iterrows():
        percentage_text = str(row.get("Missing percentage", "0")).replace("%", "")
        try:
            percentage = float(percentage_text)
        except ValueError:
            continue

        if percentage <= HIGH_MISSINGNESS_REVIEW_THRESHOLD:
            continue

        if row.get("Recommended action") == "Target review required":
            continue

        column = str(row.get("Column", "Unknown column"))
        handling = str(row.get("Recommended handling", "Imputation"))
        warnings.append(
            f"{column} contains {percentage:.2f}% missing values. {handling} was applied according to the selected preprocessing pipeline. "
            "However, due to the high level of missingness, users should review whether retaining this feature is appropriate."
        )
    return warnings


def _column_data_type(dataframe: pd.DataFrame, column: str) -> str:
    """Return a readable column dtype label."""

    if column in dataframe.columns:
        return str(dataframe[column].dtype)
    return "Unknown"


def _recommended_handling(
    column: str,
    numeric_columns: set[str],
    categorical_columns: set[str],
    boolean_columns: set[str],
) -> str:
    """Return the recommended handling method for a column."""

    if column in numeric_columns:
        return "Median imputation"
    if column in categorical_columns or column in boolean_columns:
        return "Most frequent imputation"
    return "Review manually"


def _recommended_action(column: str, percentage: float, target_column: str | None) -> tuple[str, str]:
    """Return the recommended action and reason for a column."""

    if target_column is not None and column == target_column:
        return (
            "Target review required",
            "Supervised learning requires valid target values. Rows with missing target values should be removed before splitting.",
        )
    if percentage > HIGH_MISSINGNESS_REVIEW_THRESHOLD:
        return ("Drop Feature", "Missing values exceed 50%.")
    if percentage >= REVIEW_MISSINGNESS_THRESHOLD:
        return ("Review", "Review recommended due to high missingness.")
    return ("Impute", "Missingness is within a normal range for imputation.")


def _applied_action(recommended_action: str, handling: str) -> str:
    """Return the action applied during preprocessing."""

    if recommended_action == "Drop Feature":
        return "Dropped before preprocessing"
    if recommended_action == "Target review required":
        return "Rows with missing target removed before train/test split"
    if recommended_action in {"Impute", "Review"}:
        return f"Imputed using {handling}"
    return "Not specified"


def _review_note(percentage: float) -> str:
    """Return a review note for high-missingness columns."""

    if REVIEW_MISSINGNESS_THRESHOLD <= percentage <= HIGH_MISSINGNESS_REVIEW_THRESHOLD:
        return "Review recommended due to high missingness."
    if percentage > HIGH_MISSINGNESS_REVIEW_THRESHOLD:
        return "Reason: Missing values exceed 50%."
    return ""


def _strategy_summary(numerical_strategy: str, categorical_strategy: str, boolean_strategy: str) -> str:
    """Combine the selected strategies into one compact sentence."""

    parts: list[str] = []
    if numerical_strategy != "Not required":
        parts.append("median imputation for numeric features")
    if categorical_strategy != "Not required":
        parts.append("most frequent imputation for categorical features")
    if boolean_strategy != "Not required":
        parts.append("most frequent imputation for boolean features")

    if not parts:
        return "No missing value handling was required."
    if len(parts) == 1:
        return f"Missing value handling uses {parts[0]}."
    if len(parts) == 2:
        return f"Missing value handling uses {parts[0]} and {parts[1]}."
    return f"Missing value handling uses {parts[0]}, {parts[1]}, and {parts[2]}."
