"""Evaluation page for the Streamlit thesis application."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.application_state import has_dataset, has_task_detection, has_training_result, initialize_session_state
from src.model_evaluator import evaluate_model


def render() -> None:
    """Render the evaluation page."""

    initialize_session_state(st.session_state)

    st.title("Evaluation")

    if not has_dataset(st.session_state):
        st.warning("Please upload a dataset first.")
        return

    if not has_training_result(st.session_state):
        st.warning("Please train a model first.")
        return

    if not has_task_detection(st.session_state):
        st.warning("Please detect the task type first.")
        return

    if st.session_state.get("target_column") is None:
        st.warning("Please select a target column first.")
        return

    st.write("Evaluate the trained pipeline on the available dataset.")

    if st.button("Evaluate Model"):
        evaluation_result = evaluate_model(
            st.session_state["training_result"],
            st.session_state["dataframe"],
            st.session_state["target_column"],
            st.session_state["task_result"].task_type,
        )
        st.session_state["evaluation_result"] = evaluation_result

        if not evaluation_result.success:
            for error in evaluation_result.errors:
                st.error(error)
            return

        st.success("Evaluation completed successfully.")
        st.write(f"Quality rating: {evaluation_result.quality_rating}")
        st.write(f"Quality summary: {evaluation_result.quality_summary}")

        if evaluation_result.classification_metrics is not None:
            st.subheader("Classification Metrics")
            st.json(evaluation_result.classification_metrics)
            if evaluation_result.confusion_matrix is not None:
                st.subheader("Confusion Matrix")
                confusion_df = pd.DataFrame(evaluation_result.confusion_matrix)
                st.dataframe(confusion_df, use_container_width=True)

        if evaluation_result.regression_metrics is not None:
            st.subheader("Regression Metrics")
            st.json(evaluation_result.regression_metrics)

    st.markdown(
        """
        ### Placeholder content
        The evaluation result will be stored in session state for future reporting.
        """
    )