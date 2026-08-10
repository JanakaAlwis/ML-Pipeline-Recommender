"""Lightweight pre-modeling feature assessment helpers."""

from __future__ import annotations

from collections import defaultdict
import hashlib
from typing import Any

import numpy as np
import pandas as pd

from src.data_profiler import HIGH_CORRELATION_THRESHOLD

MISSING_REVIEW_THRESHOLD = 20.0
MISSING_HIGH_THRESHOLD = 50.0
IDENTIFIER_REVIEW_THRESHOLD = 0.45
IDENTIFIER_POSSIBLE_THRESHOLD = 0.75
RELATIONSHIP_STRONG_THRESHOLD = 0.50
RELATIONSHIP_MODERATE_THRESHOLD = 0.20
REDUNDANCY_REVIEW_THRESHOLD = 0.70

FEATURE_ASSESSMENT_COLUMNS = [
    "Feature",
    "Data Type",
    "ID Indicator",
    "Missing %",
    "Target Relationship",
    "Redundancy",
    "Assessment",
]


def build_feature_assessment_frame(profile_result: Any, target_column: str | None = None) -> pd.DataFrame:
    """Build a compact profiling-only feature assessment table."""

    dataframe = getattr(profile_result, "dataframe", None)
    if dataframe is None or not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
        return pd.DataFrame(columns=FEATURE_ASSESSMENT_COLUMNS)

    feature_columns = [column for column in dataframe.columns if column != target_column]
    if not feature_columns:
        return pd.DataFrame(columns=FEATURE_ASSESSMENT_COLUMNS)

    numeric_columns = set(getattr(profile_result, "numeric_columns", []) or [])
    categorical_columns = set(getattr(profile_result, "categorical_columns", []) or [])
    boolean_columns = set(getattr(profile_result, "boolean_columns", []) or [])
    missing_percentages = getattr(profile_result, "missing_percentages", {}) or {}
    correlation_pairs = getattr(profile_result, "high_correlation_pairs", []) or []

    target_series = None
    if target_column is not None and target_column in dataframe.columns:
        target_series = dataframe[target_column]

    feature_metadata: dict[str, dict[str, Any]] = {}
    for column in feature_columns:
        series = dataframe[column]
        feature_metadata[column] = {
            "series": series,
            "data_type": str(series.dtype),
            "id_indicator": _identifier_indicator(column, series, len(dataframe)),
            "missing_percentage": float(missing_percentages.get(column, 0.0) or 0.0),
        }

    redundancy_candidates = [
        column
        for column in feature_columns
        if feature_metadata[column]["id_indicator"] == "Low identifier risk"
    ]
    redundancy_map = _build_redundancy_map(dataframe, redundancy_candidates, correlation_pairs)

    rows: list[dict[str, Any]] = []
    if target_series is not None and target_column is not None:
        rows.append(
            {
                "Feature": target_column,
                "Data Type": str(target_series.dtype),
                "ID Indicator": "—",
                "Missing %": float(missing_percentages.get(target_column, 0.0) or 0.0),
                "Target Relationship": "—",
                "Redundancy": "—",
                "Assessment": "Target variable",
            }
        )

    for column in feature_columns:
        series = feature_metadata[column]["series"]
        data_type = feature_metadata[column]["data_type"]
        identifier_label = feature_metadata[column]["id_indicator"]
        missing_percentage = feature_metadata[column]["missing_percentage"]
        relationship_label = _target_relationship_label(
            series,
            target_series,
            column in numeric_columns,
            column in categorical_columns or column in boolean_columns,
        )
        if identifier_label != "Low identifier risk":
            redundancy_label = "Not assessed"
        else:
            redundancy_label = redundancy_map.get(column, "Low redundancy risk")
        assessment = _final_assessment(
            missing_percentage=missing_percentage,
            identifier_label=identifier_label,
            relationship_label=relationship_label,
            redundancy_label=redundancy_label,
        )
        rows.append(
            {
                "Feature": column,
                "Data Type": data_type,
                "ID Indicator": identifier_label,
                "Missing %": missing_percentage,
                "Target Relationship": relationship_label,
                "Redundancy": redundancy_label,
                "Assessment": assessment,
            }
        )

    return pd.DataFrame(rows, columns=FEATURE_ASSESSMENT_COLUMNS)


def _identifier_indicator(column: str, series: pd.Series, row_count: int) -> str:
    """Return a conservative identifier-risk label for a feature."""

    if row_count <= 0:
        return "Review recommended"

    score = 0.0
    normalized_name = _normalize_name(column)
    name_tokens = {
        "id",
        "identifier",
        "ident",
        "record",
        "rec",
        "key",
        "index",
        "uuid",
        "guid",
        "code",
    }
    if any(token == normalized_name or token in _split_name_tokens(normalized_name) for token in name_tokens):
        score += 0.45

    uniqueness_ratio = float(series.nunique(dropna=True)) / float(row_count)
    if uniqueness_ratio >= 0.99:
        score += 0.35
    elif uniqueness_ratio >= 0.95:
        score += 0.25
    elif uniqueness_ratio >= 0.80:
        score += 0.15

    if _looks_sequential(series):
        score += 0.30

    if _looks_identifier_like_string(series):
        score += 0.20

    if score >= IDENTIFIER_POSSIBLE_THRESHOLD:
        return "Possible identifier"
    if score >= IDENTIFIER_REVIEW_THRESHOLD:
        return "Review recommended"
    return "Low identifier risk"


def _target_relationship_label(
    feature_series: pd.Series,
    target_series: pd.Series | None,
    feature_is_numeric: bool,
    feature_is_categorical: bool,
) -> str:
    """Return a lightweight association label with the target column."""

    if target_series is None:
        return "Not available"

    if target_series.dropna().empty or feature_series.dropna().empty:
        return "Not available"

    target_is_numeric = bool(pd.api.types.is_numeric_dtype(target_series))
    target_is_categorical = bool(
        pd.api.types.is_object_dtype(target_series)
        or pd.api.types.is_string_dtype(target_series)
        or isinstance(target_series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(target_series)
    )

    if feature_is_numeric and target_is_numeric:
        score = _spearman_score(feature_series, target_series)
    elif feature_is_numeric and target_is_categorical:
        score = _correlation_ratio(target_series, feature_series)
    elif feature_is_categorical and target_is_numeric:
        score = _correlation_ratio(feature_series, target_series)
    elif feature_is_categorical and target_is_categorical:
        score = _cramers_v(feature_series, target_series)
    else:
        score = None

    return _strength_label(score)


def _final_assessment(
    *,
    missing_percentage: float,
    identifier_label: str,
    relationship_label: str,
    redundancy_label: str,
) -> str:
    """Combine profiling signals into one conservative assessment label."""

    if missing_percentage >= MISSING_HIGH_THRESHOLD:
        return "High missingness - review recommended"
    if identifier_label == "Possible identifier":
        return "Potential identifier"
    if redundancy_label.startswith("Potentially redundant"):
        return "Potentially redundant"
    if missing_percentage >= MISSING_REVIEW_THRESHOLD:
        return "Review recommended"
    if identifier_label == "Review recommended" or redundancy_label == "Review recommended":
        return "Review recommended"
    if relationship_label == "Strong":
        return "Potentially useful"
    return "Looks suitable"


def _build_redundancy_map(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    correlation_pairs: list[tuple[str, str, float]],
) -> dict[str, str]:
    """Build a per-feature redundancy label from profiling-safe signals."""

    redundancy_map: dict[str, str] = {column: "Low redundancy risk" for column in feature_columns}

    for column_a, column_b, value in correlation_pairs:
        if column_a not in redundancy_map or column_b not in redundancy_map:
            continue
        if abs(float(value)) >= HIGH_CORRELATION_THRESHOLD:
            _set_redundancy_label(redundancy_map, column_a, column_b, "Potentially redundant")
        elif abs(float(value)) >= REDUNDANCY_REVIEW_THRESHOLD:
            _set_redundancy_label(redundancy_map, column_a, column_b, "Review recommended")

    duplicate_groups = _find_exact_duplicate_groups(dataframe, feature_columns)
    for group in duplicate_groups:
        if len(group) < 2:
            continue
        for column in group:
            if redundancy_map.get(column, "Low redundancy risk") != "Potentially redundant":
                other_columns = [item for item in group if item != column]
                reference = other_columns[0] if other_columns else None
                if reference is not None:
                    redundancy_map[column] = f"Potentially redundant with {reference}"

    return redundancy_map


def _set_redundancy_label(redundancy_map: dict[str, str], column_a: str, column_b: str, label: str) -> None:
    """Update both columns with a shared redundancy label."""

    if redundancy_map.get(column_a, "Low redundancy risk") == "Low redundancy risk":
        redundancy_map[column_a] = f"{label} with {column_b}"
    if redundancy_map.get(column_b, "Low redundancy risk") == "Low redundancy risk":
        redundancy_map[column_b] = f"{label} with {column_a}"


def _find_exact_duplicate_groups(dataframe: pd.DataFrame, feature_columns: list[str]) -> list[list[str]]:
    """Find columns that are exactly duplicated across the dataset."""

    digests: dict[str, list[str]] = defaultdict(list)
    for column in feature_columns:
        series = dataframe[column].astype("object").where(dataframe[column].notna(), "<NA>")
        hashed = pd.util.hash_pandas_object(series, index=False)
        digest = hashlib.blake2b(hashed.values.tobytes(), digest_size=8).hexdigest()
        digests[digest].append(column)

    duplicate_groups: list[list[str]] = []
    for group in digests.values():
        if len(group) < 2:
            continue
        confirmed_group: list[str] = [group[0]]
        for candidate in group[1:]:
            if dataframe[confirmed_group[0]].equals(dataframe[candidate]):
                confirmed_group.append(candidate)
        if len(confirmed_group) > 1:
            duplicate_groups.append(confirmed_group)
    return duplicate_groups


def _strength_label(score: float | None) -> str:
    """Map a numeric association score to a simple label."""

    if score is None or np.isnan(score):
        return "Not available"
    if abs(float(score)) >= RELATIONSHIP_STRONG_THRESHOLD:
        return "Strong"
    if abs(float(score)) >= RELATIONSHIP_MODERATE_THRESHOLD:
        return "Moderate"
    return "Weak"


def _spearman_score(feature_series: pd.Series, target_series: pd.Series) -> float | None:
    """Calculate a rank-based association score for two numeric series."""

    numeric_frame = pd.DataFrame(
        {
            "feature": pd.to_numeric(feature_series, errors="coerce"),
            "target": pd.to_numeric(target_series, errors="coerce"),
        }
    ).dropna()
    if len(numeric_frame) < 2:
        return None
    if numeric_frame["feature"].nunique(dropna=True) < 2 or numeric_frame["target"].nunique(dropna=True) < 2:
        return None
    return float(numeric_frame["feature"].corr(numeric_frame["target"], method="spearman"))


def _correlation_ratio(categories: pd.Series, values: pd.Series) -> float | None:
    """Calculate a one-way association ratio for categorical versus numeric data."""

    frame = pd.DataFrame({"category": categories, "value": pd.to_numeric(values, errors="coerce")}).dropna()
    if frame.empty or frame["category"].nunique(dropna=True) < 2 or frame["value"].nunique(dropna=True) < 2:
        return None

    overall_mean = float(frame["value"].mean())
    grouped = frame.groupby("category", dropna=True)["value"]
    between_sum = float(sum(len(group) * (float(group.mean()) - overall_mean) ** 2 for _, group in grouped))
    total_sum = float(((frame["value"] - overall_mean) ** 2).sum())
    if total_sum <= 0:
        return None
    return float(np.sqrt(between_sum / total_sum))


def _cramers_v(feature_series: pd.Series, target_series: pd.Series) -> float | None:
    """Calculate Cramer's V for two categorical series."""

    frame = pd.DataFrame({"feature": feature_series, "target": target_series}).dropna()
    if frame.empty or frame["feature"].nunique(dropna=True) < 2 or frame["target"].nunique(dropna=True) < 2:
        return None

    contingency = pd.crosstab(frame["feature"], frame["target"])
    if contingency.empty:
        return None

    observed = contingency.to_numpy(dtype=float)
    total = float(observed.sum())
    if total <= 0:
        return None

    row_totals = observed.sum(axis=1, keepdims=True)
    column_totals = observed.sum(axis=0, keepdims=True)
    expected = row_totals @ column_totals / total
    with np.errstate(divide="ignore", invalid="ignore"):
        chi_square = float(np.nansum((observed - expected) ** 2 / expected))

    rows, columns = observed.shape
    denominator = total * max(min(rows - 1, columns - 1), 1)
    if denominator <= 0:
        return None
    return float(np.sqrt(chi_square / denominator))


def _looks_sequential(series: pd.Series) -> bool:
    """Return True when a numeric column looks like a sequential identifier."""

    if not pd.api.types.is_numeric_dtype(series):
        return False

    cleaned = pd.to_numeric(series, errors="coerce").dropna()
    if cleaned.empty or cleaned.nunique(dropna=True) != len(cleaned):
        return False

    sorted_values = cleaned.sort_values().to_numpy()
    if len(sorted_values) < 3:
        return False

    diffs = np.diff(sorted_values)
    if not np.allclose(diffs, 1.0):
        return False

    return True


def _looks_identifier_like_string(series: pd.Series) -> bool:
    """Return True for string columns that resemble codes or UUIDs."""

    if pd.api.types.is_numeric_dtype(series):
        return False

    cleaned = series.dropna().astype(str)
    if cleaned.empty:
        return False

    sample = cleaned.head(min(len(cleaned), 25))
    uuid_like_count = sum(_looks_uuid_like(value) for value in sample)
    code_like_count = sum(_looks_code_like(value) for value in sample)
    return uuid_like_count >= max(1, len(sample) // 2) or code_like_count >= max(1, len(sample) // 2)


def _looks_uuid_like(value: str) -> bool:
    """Return True when a value resembles a UUID or GUID."""

    text = value.strip()
    if len(text) not in {32, 36}:
        return False
    hexadecimal = set("0123456789abcdefABCDEF-")
    return all(character in hexadecimal for character in text)


def _looks_code_like(value: str) -> bool:
    """Return True when a value looks like a short alphanumeric code."""

    text = value.strip()
    if len(text) < 6:
        return False
    if text.isdigit():
        return False
    alphanumeric = sum(character.isalnum() for character in text)
    return alphanumeric == len(text) and any(character.isdigit() for character in text) and any(character.isalpha() for character in text)


def _normalize_name(column: str) -> str:
    """Normalize a column name for pattern checks."""

    normalized = []
    for character in column:
        if character.isupper() and normalized:
            normalized.append(" ")
        normalized.append(character.lower() if character.isalnum() else " ")
    return " ".join("".join(normalized).split())


def _split_name_tokens(normalized_name: str) -> set[str]:
    """Return a token set for a normalized column name."""

    return {token for token in normalized_name.replace("_", " ").split() if token}