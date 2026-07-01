"""Pipeline Recommendations page for the Streamlit thesis application."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.application_state import has_profile, has_task_detection, initialize_session_state
from src.recommendation_engine import recommend_pipelines


def render() -> None:
    """Render the pipeline recommendation page."""

    initialize_session_state(st.session_state)

    st.title("Pipeline Recommendations")

    if not has_profile(st.session_state):
        st.warning("Please generate the dataset profile first.")
        return

    if not has_task_detection(st.session_state):
        st.warning("Please detect the task type first.")
        return

    st.write("Review the recommended pipelines and choose one to continue.")

    if st.button("Generate Recommendations"):
        recommendation_result = recommend_pipelines(
            st.session_state["profile_result"],
            st.session_state["task_result"],
        )
        st.session_state["recommendation_result"] = recommendation_result

        if not recommendation_result.success:
            for error in recommendation_result.errors:
                st.error(error)
            return

    recommendation_result = st.session_state.get("recommendation_result")
    if recommendation_result is None:
        st.markdown(
            """
            ### Placeholder content
            Generate recommendations to see the top pipeline candidates here.
            """
        )
        return

    if not recommendation_result.success:
        for error in recommendation_result.errors:
            st.error(error)
        return

    top_pipelines = recommendation_result.top_pipelines[:3]
    if not top_pipelines:
        st.warning("No pipeline recommendations were generated.")
        return

    pipeline_options = {pipeline.pipeline_name: pipeline for pipeline in top_pipelines}
    selected_name = st.selectbox(
        "Select a pipeline",
        options=list(pipeline_options.keys()),
        key="recommendations_selected_pipeline",
    )
    selected_pipeline = pipeline_options[selected_name]
    st.session_state["selected_pipeline"] = selected_pipeline

    summary_table = pd.DataFrame(
        [
            {
                "Rank": pipeline.pipeline_rank,
                "Pipeline": pipeline.pipeline_name,
                "Task": pipeline.task_type,
                "Confidence": pipeline.confidence_score,
                "Complexity": pipeline.estimated_training_complexity,
            }
            for pipeline in top_pipelines
        ]
    )

    st.subheader("Top 3 Pipelines")
    st.dataframe(summary_table, use_container_width=True)

    st.subheader("Selected Pipeline Details")
    st.write(f"Confidence: {selected_pipeline.confidence_score}")
    st.write(f"Evidence: {selected_pipeline.evidence}")
    st.write(f"Advantages: {selected_pipeline.advantages}")
    st.write(f"Limitations: {selected_pipeline.limitations}")
    st.write(f"Preprocessing steps: {selected_pipeline.preprocessing_steps}")
    st.write(f"Model family: {selected_pipeline.recommended_model_family}")

    st.markdown(
        """
        ### Placeholder content
        The selected pipeline will be passed to the training step in the next page.
        """
    )