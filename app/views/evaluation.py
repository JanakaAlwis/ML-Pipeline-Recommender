"""Evaluation page for the Streamlit thesis application."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.streamlit_app import render_header
from src.application_state import has_dataset, has_task_detection, has_training_result, initialize_session_state
from src.model_evaluator import evaluate_model


def render() -> None:
    """Render the evaluation page."""

    initialize_session_state(st.session_state)

    render_header(
        "Evaluation",
        "Review the fitted model on the held-out data.",
        "After evaluation, generate explainability results.",
        "These metrics show how well the model performs on unseen data.",
    )

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

    st.write("Evaluate the trained pipeline.")

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
        quality_left, quality_right = st.columns(2)
        quality_left.metric("Quality rating", evaluation_result.quality_rating or "N/A")
        quality_right.metric("Rows evaluated", evaluation_result.row_count if evaluation_result.row_count is not None else "N/A")

        st.write(evaluation_result.quality_summary)

        if evaluation_result.classification_metrics is not None:
            st.subheader("Classification metrics")
            classification_metrics_table = pd.DataFrame(list(evaluation_result.classification_metrics.items()), columns=["metric", "value"])
            st.dataframe(classification_metrics_table, use_container_width=True)
            if evaluation_result.confusion_matrix is not None:
                st.subheader("Confusion matrix")
                confusion_df = pd.DataFrame(evaluation_result.confusion_matrix)
                st.dataframe(confusion_df, use_container_width=True)
                st.bar_chart(confusion_df)

        if evaluation_result.regression_metrics is not None:
            st.subheader("Regression metrics")
            regression_metrics_table = pd.DataFrame(list(evaluation_result.regression_metrics.items()), columns=["metric", "value"])
            st.dataframe(regression_metrics_table, use_container_width=True)

    st.info("Next: open Explainability.")