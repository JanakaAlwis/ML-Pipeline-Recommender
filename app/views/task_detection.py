"""Task Detection page for the Streamlit thesis application."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app.streamlit_app import render_header, render_wrapped_key_value_table, render_wrapped_table
from src.application_state import has_dataset, has_profile, initialize_session_state
from src.display_formatting import format_confidence, format_percentage
from src.feature_assessment import build_feature_assessment_frame
from src.task_detector import detect_task_type


def _is_categorical_target(series: pd.Series) -> bool:
    """Return True when the target should be displayed as categorical."""

    return bool(
        pd.api.types.is_bool_dtype(series)
        or pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or pd.api.types.is_categorical_dtype(series)
    )


def _render_categorical_target_distribution(series: pd.Series) -> None:
    """Render a frequency table and bar chart for categorical targets."""

    target_counts = series.value_counts(dropna=False).head(20)
    target_table = target_counts.rename_axis("value").reset_index(name="count")
    st.subheader("Target distribution")
    render_wrapped_table(target_table)
    if not target_counts.empty:
        st.bar_chart(target_counts)


def _render_numeric_target_distribution(series: pd.Series) -> None:
    """Render regression-friendly target distribution visuals."""

    st.subheader("Target distribution")

    numeric_series = pd.to_numeric(series, errors="coerce").dropna()
    if numeric_series.empty:
        st.info("No numeric target values are available for distribution analysis.")
        return

    summary_table = pd.DataFrame(
        [
            {
                "count": numeric_series.count(),
                "mean": numeric_series.mean(),
                "std": numeric_series.std(),
                "min": numeric_series.min(),
                "median": numeric_series.median(),
                "max": numeric_series.max(),
            }
        ]
    )
    st.subheader("Summary statistics")
    st.dataframe(summary_table, use_container_width=True, hide_index=True)

    histogram = px.histogram(numeric_series.to_frame(name=series.name or "target"), x=series.name or "target", nbins=30)
    histogram.update_layout(margin={"l": 10, "r": 10, "t": 30, "b": 10})
    st.plotly_chart(histogram, use_container_width=True)

    box_plot = px.box(numeric_series.to_frame(name=series.name or "target"), y=series.name or "target")
    box_plot.update_layout(margin={"l": 10, "r": 10, "t": 30, "b": 10})
    st.plotly_chart(box_plot, use_container_width=True)


def _render_task_detection_result(task_result: object) -> None:
    """Render the detected task details."""

    if not getattr(task_result, "success", False):
        for error in getattr(task_result, "errors", []) or []:
            st.error(error)
        return

    st.success("Task type detected successfully.")
    detail_rows: list[tuple[str, object]] = [
        ("Task type", task_result.task_type),
        ("Confidence", format_confidence(task_result.confidence)),
        ("Target column", task_result.target_column),
        ("Target dtype", task_result.target_dtype),
        ("Detection reason", task_result.detection_reason),
    ]
    if task_result.alternative_task_type is not None:
        detail_rows.append(("Alternative task type", task_result.alternative_task_type))
    render_wrapped_key_value_table(detail_rows)


def _render_feature_assessment(profile_result: object, target_column: str) -> None:
    """Render the lightweight pre-modeling feature assessment."""

    st.subheader("Feature Assessment")
    st.caption(
        "Feature Assessment provides a preliminary review of the selected dataset features after the target variable has been identified. "
        "It highlights features that may require review before model training. The assessment does not automatically remove or modify features."
    )
    with st.expander("How the assessment works", expanded=False):
        st.write(
            "The assessment uses profiling-only signals such as data type, identifier-like patterns, missing values, target associations, and redundancy checks."
        )
        st.write(
            "The selected target variable is shown separately and is not treated as a predictive feature. Identifier-like columns are kept visible but are excluded from redundancy comparisons."
        )

    feature_assessment_table = build_feature_assessment_frame(profile_result, target_column=target_column)
    if feature_assessment_table.empty:
        st.info("Feature Assessment is not available for this dataset.")
        return

    display_table = feature_assessment_table.copy()
    if "Missing %" in display_table.columns:
        display_table["Missing %"] = display_table["Missing %"].map(format_percentage)
    st.caption(f"Selected target variable: {target_column}")
    st.dataframe(display_table, use_container_width=True, hide_index=True)


def render() -> None:
    """Render the task detection page."""

    initialize_session_state(st.session_state)

    render_header(
        "Task Detection",
        "Select the target column and determine the task type.",
        "After detection, generate pipeline recommendations.",
        "The task type controls the later recommendation, training, and evaluation steps.",
    )

    if not has_dataset(st.session_state):
        st.warning("Please upload a dataset first.")
        return

    if not has_profile(st.session_state):
        st.warning("Please generate the dataset profile first.")
        return

    dataframe = st.session_state["dataframe"]
    if not isinstance(dataframe, pd.DataFrame):
        st.warning("Please upload a dataset first.")
        return

    st.write("Select the target column and detect the task type.")
    placeholder_label = "Select a target column"
    selected_target_column = st.selectbox(
        "Target column",
        options=[placeholder_label, *list(dataframe.columns)],
        index=0,
        key="task_detection_target_column",
    )

    if selected_target_column == placeholder_label:
        st.session_state["target_column"] = None
        st.info("Select a target column to view target distribution.")
    else:
        st.session_state["target_column"] = selected_target_column
        target_series = dataframe[selected_target_column]
        stored_task_result = st.session_state.get("task_result")
        if getattr(stored_task_result, "target_column", None) == selected_target_column and getattr(stored_task_result, "success", False):
            task_type = getattr(stored_task_result, "task_type", None)
            if task_type == "regression":
                _render_numeric_target_distribution(target_series)
            else:
                _render_categorical_target_distribution(target_series)
            _render_task_detection_result(stored_task_result)
        else:
            if _is_categorical_target(target_series):
                _render_categorical_target_distribution(target_series)
            else:
                _render_numeric_target_distribution(target_series)

    detect_disabled = selected_target_column == placeholder_label
    detect_clicked = st.button("Detect Task Type", disabled=detect_disabled)
    if detect_clicked:
        task_result = detect_task_type(dataframe, selected_target_column)
        st.session_state["task_result"] = task_result

        _render_task_detection_result(task_result)

    stored_task_result = st.session_state.get("task_result")
    if selected_target_column != placeholder_label and getattr(stored_task_result, "target_column", None) == selected_target_column:
        st.caption("Detected task details remain available below the button.")
        if getattr(stored_task_result, "success", False):
            _render_feature_assessment(st.session_state["profile_result"], selected_target_column)

    st.info("Next: open Pipeline Recommendations.")