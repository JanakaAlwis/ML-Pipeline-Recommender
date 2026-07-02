"""Explainability page for the Streamlit thesis application."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.streamlit_app import render_header
from src.application_state import has_dataset, has_task_detection, has_training_result, initialize_session_state
from src.explainability import explain_model


def render() -> None:
    """Render the explainability page."""

    initialize_session_state(st.session_state)

    render_header(
        "Explainability",
        "Interpret the trained model using feature importance.",
        "Review the most influential features and their relative importance.",
        "This page shows which original features influenced the model most.",
    )

    if not has_dataset(st.session_state):
        st.warning("Please upload a dataset first.")
        return

    if not has_task_detection(st.session_state):
        st.warning("Please detect the task type first.")
        return

    if not has_training_result(st.session_state):
        st.warning("Please train a model first.")
        return

    if st.session_state.get("target_column") is None:
        st.warning("Please select a target column first.")
        return

    st.write("Generate the explanation for the trained model.")

    if st.button("Generate Explainability"):
        explainability_result = explain_model(
            st.session_state["training_result"],
            st.session_state["dataframe"],
            st.session_state["target_column"],
            st.session_state["task_result"].task_type,
        )
        st.session_state["explainability_result"] = explainability_result

        if not explainability_result.success:
            for error in explainability_result.errors:
                st.error(error)
            return

    explainability_result = st.session_state.get("explainability_result")
    if explainability_result is None:
        st.markdown(
            """
            Generate explainability to see feature importance.
            """
        )
        return

    if not explainability_result.success:
        for error in explainability_result.errors:
            st.error(error)
        return

    summary_left, summary_right = st.columns(2)
    summary_left.metric("Method", explainability_result.method_used or "N/A")
    summary_right.metric("Features ranked", len(explainability_result.top_features))

    st.subheader("Summary")
    st.write(explainability_result.explanation_summary)

    if explainability_result.top_features:
        top_features_df = pd.DataFrame(
            explainability_result.top_features,
            columns=["Feature", "Importance"],
        )
        st.subheader("Top features")
        st.dataframe(top_features_df, use_container_width=True)
        st.subheader("Feature importance")
        st.bar_chart(top_features_df.set_index("Feature"))

    st.info("Next: open Report.")