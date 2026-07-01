"""Training page for the Streamlit thesis application."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.application_state import has_dataset, has_profile, has_task_detection, initialize_session_state
from src.model_trainer import train_pipeline
from src.pipeline_builder import build_pipeline


def render() -> None:
    """Render the training page."""

    initialize_session_state(st.session_state)

    st.title("Training")

    if not has_dataset(st.session_state):
        st.warning("Please upload a dataset first.")
        return

    if not has_profile(st.session_state):
        st.warning("Please generate the dataset profile first.")
        return

    if not has_task_detection(st.session_state):
        st.warning("Please detect the task type first.")
        return

    if st.session_state.get("selected_pipeline") is None:
        st.warning("Please select a pipeline from the recommendations first.")
        return

    if st.session_state.get("target_column") is None:
        st.warning("Please select a target column first.")
        return

    st.write("Train the selected pipeline on the uploaded dataset.")

    if st.button("Train Selected Pipeline"):
        built_pipeline_result = build_pipeline(
            st.session_state["selected_pipeline"],
            st.session_state["profile_result"],
            st.session_state["task_result"],
        )
        st.session_state["built_pipeline_result"] = built_pipeline_result

        if not built_pipeline_result.success:
            for error in built_pipeline_result.errors:
                st.error(error)
            return

        training_result = train_pipeline(
            built_pipeline_result,
            st.session_state["dataframe"],
            st.session_state["target_column"],
            st.session_state["task_result"].task_type,
        )
        st.session_state["training_result"] = training_result

        if not training_result.success:
            for error in training_result.errors:
                st.error(error)
            return

        st.success("Training completed successfully.")
        st.write(f"Training time (seconds): {training_result.training_time_seconds}")
        st.write(f"Cross-validation scores: {training_result.cross_validation_scores}")
        st.write(f"Mean CV score: {training_result.mean_cv_score}")
        st.write(f"Std CV score: {training_result.std_cv_score}")

        if training_result.cross_validation_scores:
            cv_table = pd.DataFrame({"cv_score": training_result.cross_validation_scores})
            st.dataframe(cv_table, use_container_width=True)

    st.markdown(
        """
        ### Placeholder content
        The fitted pipeline will be stored in session state for evaluation.
        """
    )