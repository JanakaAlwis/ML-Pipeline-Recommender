"""Task Detection page for the Streamlit thesis application."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.application_state import has_dataset, has_profile, initialize_session_state
from src.task_detector import detect_task_type


def render() -> None:
    """Render the task detection page."""

    initialize_session_state(st.session_state)

    st.title("Task Detection")

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

    st.write("Select the target column to detect whether the task is classification or regression.")
    selected_target_column = st.selectbox(
        "Target column",
        options=list(dataframe.columns),
        index=0,
        key="task_detection_target_column",
    )
    st.session_state["target_column"] = selected_target_column

    if st.button("Detect Task Type"):
        task_result = detect_task_type(dataframe, selected_target_column)
        st.session_state["task_result"] = task_result

        if not task_result.success:
            for error in task_result.errors:
                st.error(error)
            return

        st.success("Task type detected successfully.")
        st.write(f"Task type: {task_result.task_type}")
        st.write(f"Confidence: {task_result.confidence}")
        st.write(f"Target column: {task_result.target_column}")
        st.write(f"Target dtype: {task_result.target_dtype}")
        st.write(f"Unique count: {task_result.target_unique_count}")
        st.write(f"Unique ratio: {task_result.target_unique_ratio}")
        st.write(f"Detection reason: {task_result.detection_reason}")
        if task_result.alternative_task_type is not None:
            st.write(f"Alternative task type: {task_result.alternative_task_type}")

    st.markdown(
        """
        ### Placeholder content
        The detected task will be stored in session state for the recommendation
        step that comes next.
        """
    )