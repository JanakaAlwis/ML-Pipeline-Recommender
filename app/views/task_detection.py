"""Task Detection page for the Streamlit thesis application."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.streamlit_app import render_header, render_wrapped_key_value_table, render_wrapped_table
from src.application_state import has_dataset, has_profile, initialize_session_state
from src.task_detector import detect_task_type


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
    selected_target_column = st.selectbox(
        "Target column",
        options=list(dataframe.columns),
        index=0,
        key="task_detection_target_column",
    )
    st.session_state["target_column"] = selected_target_column

    target_counts = dataframe[selected_target_column].value_counts(dropna=False).head(20)
    target_table = target_counts.rename_axis("value").reset_index(name="count")
    st.subheader("Target distribution")
    render_wrapped_table(target_table)
    if not target_counts.empty:
        st.bar_chart(target_counts)

    if st.button("Detect Task Type"):
        task_result = detect_task_type(dataframe, selected_target_column)
        st.session_state["task_result"] = task_result

        if not task_result.success:
            for error in task_result.errors:
                st.error(error)
            return

        st.success("Task type detected successfully.")
        detail_rows: list[tuple[str, object]] = [
            ("Task type", task_result.task_type),
            ("Confidence", task_result.confidence),
            ("Target column", task_result.target_column),
            ("Target dtype", task_result.target_dtype),
            ("Detection reason", task_result.detection_reason),
        ]
        if task_result.alternative_task_type is not None:
            detail_rows.append(("Alternative task type", task_result.alternative_task_type))
        render_wrapped_key_value_table(detail_rows)

    st.info("Next: open Pipeline Recommendations.")