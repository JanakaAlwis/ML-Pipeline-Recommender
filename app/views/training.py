"""Training page for the Streamlit thesis application."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.streamlit_app import render_header
from src.application_state import has_dataset, has_profile, has_task_detection, initialize_session_state
from src.model_trainer import train_pipeline
from src.pipeline_builder import build_pipeline


def render() -> None:
    """Render the training page."""

    initialize_session_state(st.session_state)

    render_header(
        "Training",
        "Build and fit the selected pipeline.",
        "After training, evaluate the model and inspect the results.",
        "This step turns the recommendation into a fitted model.",
    )

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

    st.write("Train the selected pipeline.")

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
        metrics_left, metrics_right, metrics_third = st.columns(3)
        metrics_left.metric("Training time (s)", f"{training_result.training_time_seconds:.3f}" if training_result.training_time_seconds is not None else "N/A")
        metrics_right.metric("Mean CV score", f"{training_result.mean_cv_score:.3f}" if training_result.mean_cv_score is not None else "N/A")
        metrics_third.metric("Std CV score", f"{training_result.std_cv_score:.3f}" if training_result.std_cv_score is not None else "N/A")

        if training_result.cross_validation_scores:
            cv_table = pd.DataFrame({"fold": list(range(1, len(training_result.cross_validation_scores) + 1)), "cv_score": training_result.cross_validation_scores})
            st.subheader("Cross-validation scores")
            st.dataframe(cv_table, use_container_width=True)
            st.bar_chart(cv_table.set_index("fold"))

            st.subheader("Training summary")
        st.write(f"Model: {training_result.model_name}")
        st.write(f"Scoring metric: {training_result.scoring_metric}")
        st.write(f"Train rows: {training_result.train_row_count}, Test rows: {training_result.test_row_count}")

        st.info("Next: open Evaluation.")