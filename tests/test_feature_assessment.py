"""Tests for the lightweight feature assessment helpers."""

from __future__ import annotations

import time

import pandas as pd
import numpy as np

from src.feature_assessment import FEATURE_ASSESSMENT_COLUMNS, build_feature_assessment_frame
from src.data_profiler import profile_dataset


def _assessment_frame(dataframe: pd.DataFrame, target_column: str | None = None) -> pd.DataFrame:
    """Build a feature assessment frame for a test dataframe."""

    profile_result = profile_dataset(dataframe, target_column=target_column)
    assert profile_result.success is True
    return build_feature_assessment_frame(profile_result, target_column=target_column)


def test_feature_assessment_frame_has_expected_columns() -> None:
    """The assessment table should expose the compact agreed columns."""

    dataframe = pd.DataFrame(
        {
            "Rec": [1, 2, 3, 4, 5, 6],
            "Dummy": [0, 1, 0, 1, 0, 1],
            "Dummy1": [0, 1, 0, 1, 0, 1],
            "sepal_length": [5.1, 4.9, 4.7, 4.6, 5.0, 5.4],
            "sepal_width": [3.5, 3.0, 3.2, 3.1, 3.6, 3.9],
            "petal_length": [1.4, 1.4, 1.3, 1.5, 1.4, 1.7],
            "petal_width": [0.2, 0.2, 0.2, 0.2, 0.2, 0.4],
            "species": ["setosa", "setosa", "setosa", "setosa", "versicolor", "versicolor"],
        }
    )

    frame = _assessment_frame(dataframe, target_column="species")

    assert list(frame.columns) == FEATURE_ASSESSMENT_COLUMNS
    assert set(frame["Feature"]) == {"species", "Rec", "Dummy", "Dummy1", "sepal_length", "sepal_width", "petal_length", "petal_width"}
    target_row = frame.loc[frame["Feature"] == "species"].iloc[0]
    assert target_row["Assessment"] == "Target variable"
    assert target_row["ID Indicator"] == "—"
    assert target_row["Redundancy"] == "—"
    assert target_row["Target Relationship"] == "—"


def test_feature_assessment_identifies_possible_identifier() -> None:
    """Sequential identifier-like columns should be flagged conservatively."""

    dataframe = pd.DataFrame(
        {
            "PassengerId": [1001, 1002, 1003, 1004, 1005, 1006],
            "age": [22, 35, 28, 41, 30, 26],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )

    frame = _assessment_frame(dataframe, target_column="target")

    passenger_row = frame.loc[frame["Feature"] == "PassengerId"].iloc[0]
    assert passenger_row["ID Indicator"] == "Possible identifier"
    assert passenger_row["Assessment"] == "Potential identifier"


def test_feature_assessment_marks_missingness_and_redundancy() -> None:
    """Missing values and high correlation should be reflected in the assessment."""

    dataframe = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4, 5, 6],
            "feature_b": [2, 4, 6, 8, 10, 12],
            "feature_c": [1, None, 3, None, 5, None],
            "target": [0, 0, 1, 1, 0, 1],
        }
    )

    frame = _assessment_frame(dataframe, target_column="target")

    redundancy_row = frame.loc[frame["Feature"] == "feature_b"].iloc[0]
    missing_row = frame.loc[frame["Feature"] == "feature_c"].iloc[0]

    assert redundancy_row["Redundancy"].startswith("Potentially redundant")
    assert redundancy_row["Assessment"] == "Potentially redundant"
    assert missing_row["Missing %"] == 50.0
    assert missing_row["Assessment"] == "High missingness - review recommended"


def test_identifier_like_columns_do_not_drive_redundancy() -> None:
    """Identifier-like columns should stay visible but not influence redundancy labels."""

    dataframe = pd.DataFrame(
        {
            "Rec": [1, 2, 3, 4, 5, 6],
            "petal_length": [1.4, 1.4, 1.3, 1.5, 1.4, 1.7],
            "petal_width": [0.2, 0.2, 0.2, 0.2, 0.2, 0.4],
            "target": [0, 0, 1, 1, 1, 1],
        }
    )

    frame = _assessment_frame(dataframe, target_column="target")

    identifier_row = frame.loc[frame["Feature"] == "Rec"].iloc[0]
    petal_length_row = frame.loc[frame["Feature"] == "petal_length"].iloc[0]

    assert identifier_row["Assessment"] == "Potential identifier"
    assert identifier_row["Redundancy"] == "Not assessed"
    assert "Rec" not in str(petal_length_row["Redundancy"])


def test_feature_assessment_reports_target_relationship_and_not_available() -> None:
    """Target associations should be shown when possible and omitted otherwise."""

    dataframe = pd.DataFrame(
        {
            "x1": [1, 2, 3, 4, 5, 6],
            "x2": [6, 5, 4, 3, 2, 1],
            "target": [0, 0, 1, 1, 1, 1],
        }
    )

    with_target = _assessment_frame(dataframe, target_column="target")
    without_target = _assessment_frame(dataframe, target_column=None)

    assert set(with_target["Target Relationship"]) <= {"Strong", "Moderate", "Weak", "Not available", "—"}
    target_row = with_target.loc[with_target["Feature"] == "target"].iloc[0]
    assert target_row["Assessment"] == "Target variable"
    assert without_target["Target Relationship"].eq("Not available").all()


def test_changing_target_updates_feature_assessment() -> None:
    """Changing the selected target should update the assessment output."""

    dataframe = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4, 5, 6],
            "feature_b": [10, 20, 10, 20, 10, 20],
            "target_classification": [0, 0, 1, 1, 1, 1],
            "target_regression": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        }
    )

    classification_frame = _assessment_frame(dataframe, target_column="target_classification")
    regression_frame = _assessment_frame(dataframe, target_column="target_regression")

    class_target_row = classification_frame.loc[classification_frame["Feature"] == "target_classification"].iloc[0]
    regression_target_row = regression_frame.loc[regression_frame["Feature"] == "target_regression"].iloc[0]

    assert class_target_row["Assessment"] == "Target variable"
    assert regression_target_row["Assessment"] == "Target variable"


def test_feature_assessment_preserves_original_dataframe() -> None:
    """The assessment should not mutate the source dataframe."""

    dataframe = pd.DataFrame(
        {
            "Rec": [1, 2, 3, 4],
            "feature": [10, 20, 30, 40],
            "target": [0, 1, 0, 1],
        }
    )
    original = dataframe.copy(deep=True)

    _assessment_frame(dataframe, target_column="target")

    pd.testing.assert_frame_equal(dataframe, original)


def test_feature_assessment_is_reasonable_on_large_dataset() -> None:
    """The profiling-only assessment should remain lightweight on larger datasets."""

    rng = np.random.default_rng(42)
    rows = 50_000
    dataframe = pd.DataFrame({f"num_{index}": rng.normal(size=rows) for index in range(45)})
    dataframe["dup_a"] = dataframe["num_0"] * 2
    dataframe["dup_b"] = dataframe["dup_a"]
    dataframe["id_col"] = np.arange(1, rows + 1)
    dataframe["cat_1"] = rng.choice(["A", "B", "C", "D"], size=rows)
    dataframe["cat_2"] = rng.choice(["X", "Y"], size=rows)
    dataframe["target"] = rng.choice([0, 1], size=rows)

    start = time.perf_counter()
    profile_result = profile_dataset(dataframe, target_column="target")
    frame = build_feature_assessment_frame(profile_result, target_column="target")
    elapsed = time.perf_counter() - start

    assert profile_result.success is True
    assert len(frame) == 51
    assert elapsed < 15.0