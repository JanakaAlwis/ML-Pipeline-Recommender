"""Report page for the Streamlit thesis application."""

from __future__ import annotations

import streamlit as st

from app.streamlit_app import render_header
from src.application_state import (
    has_evaluation_result,
    has_profile,
    has_recommendations,
    has_task_detection,
    has_training_result,
    initialize_session_state,
)
from src.report_generator import generate_report


def _workflow_results_missing() -> list[str]:
    """Return required workflow warnings for report generation."""

    missing_messages: list[str] = []
    if not has_profile(st.session_state):
        missing_messages.append("Generate the dataset profile first.")
    if not has_task_detection(st.session_state):
        missing_messages.append("Detect the task type first.")
    if not has_recommendations(st.session_state):
        missing_messages.append("Generate pipeline recommendations first.")
    if not has_training_result(st.session_state):
        missing_messages.append("Train a model first.")
    if st.session_state.get("built_pipeline_result") is None:
        missing_messages.append("Build and train the selected pipeline first.")
    if not has_evaluation_result(st.session_state):
        missing_messages.append("Evaluate the model first.")
    if st.session_state.get("selected_pipeline") is None:
        missing_messages.append("Select a pipeline before generating the report.")
    built_pipeline_result = st.session_state.get("built_pipeline_result")
    selected_pipeline = st.session_state.get("selected_pipeline")
    if (
        selected_pipeline is not None
        and built_pipeline_result is not None
        and getattr(built_pipeline_result, "pipeline_name", None) != getattr(selected_pipeline, "pipeline_name", None)
    ):
        missing_messages.append("Retrain the selected pipeline before generating the report.")
    training_result = st.session_state.get("training_result")
    if (
        built_pipeline_result is not None
        and training_result is not None
        and getattr(built_pipeline_result, "model_name", None) != getattr(training_result, "model_name", None)
    ):
        missing_messages.append("Retrain the selected pipeline before generating the report.")
    return missing_messages


def render() -> None:
    """Render the report page."""

    initialize_session_state(st.session_state)

    render_header(
        "Report",
        "Generate a final thesis-style summary of the workflow.",
        "Use this page after training and evaluation are complete.",
        "The report combines the workflow results into a Markdown summary.",
    )

    missing_messages = _workflow_results_missing()
    if missing_messages:
        for message in missing_messages:
            st.warning(message)
        st.info("Complete the earlier workflow steps before generating the report.")
        return

    st.write("Generate a final report from the completed workflow.")

    markdown_result = st.session_state.get("report_markdown_result")

    if st.button("Generate Markdown Report", use_container_width=True):
        markdown_result = generate_report(
            st.session_state.get("profile_result"),
            st.session_state.get("task_result"),
            st.session_state.get("recommendation_result"),
            st.session_state.get("training_result"),
            st.session_state.get("evaluation_result"),
            st.session_state.get("explainability_result"),
            include_pdf=False,
            dataset_name=st.session_state.get("source_name"),
            selected_pipeline=st.session_state.get("selected_pipeline"),
        )
        st.session_state["report_markdown_result"] = markdown_result
        st.session_state["report_result"] = markdown_result
        if not markdown_result.success:
            for error in markdown_result.errors:
                st.error(error)
            return

    report_result = markdown_result or st.session_state.get("report_result")
    if report_result is None:
        st.info("Generate the report to view the Markdown summary and download it.")
        return

    if not report_result.success:
        for error in report_result.errors:
            st.error(error)
        return

    st.success(report_result.messages[0] if report_result.messages else "Report generated successfully.")

    st.subheader("Markdown report")
    st.markdown(report_result.markdown_report or "No report content available.")

    st.download_button(
        label="Download Markdown",
        data=(report_result.markdown_report or "").encode("utf-8"),
        file_name="workflow_report.md",
        mime="text/markdown",
        use_container_width=True,
    )

