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
    if not has_evaluation_result(st.session_state):
        missing_messages.append("Evaluate the model first.")
    return missing_messages


def render() -> None:
    """Render the report page."""

    initialize_session_state(st.session_state)

    render_header(
        "Report",
        "Generate a final thesis-style summary of the workflow.",
        "Use this page after training and evaluation are complete.",
        "The report combines the workflow results into one Markdown summary.",
    )

    missing_messages = _workflow_results_missing()
    if missing_messages:
        for message in missing_messages:
            st.warning(message)
        st.info("Complete the earlier workflow steps before generating the report.")
        return

    st.write("Generate a final report from the completed workflow.")

    if st.button("Generate Report"):
        report_result = generate_report(
            st.session_state.get("profile_result"),
            st.session_state.get("task_result"),
            st.session_state.get("recommendation_result"),
            st.session_state.get("training_result"),
            st.session_state.get("evaluation_result"),
            st.session_state.get("explainability_result"),
            include_pdf=True,
        )
        if report_result.success:
            st.session_state["report_result"] = report_result
        else:
            for error in report_result.errors:
                st.error(error)
            return

    report_result = st.session_state.get("report_result")
    if report_result is None:
        st.info("Generate the report to view the Markdown summary and download files.")
        return

    if not report_result.success:
        for error in report_result.errors:
            st.error(error)
        return

    for warning in report_result.warnings:
        st.warning(warning)

    st.subheader("Markdown report")
    st.markdown(report_result.markdown_report or "No report content available.")

    st.download_button(
        label="Download Markdown",
        data=(report_result.markdown_report or "").encode("utf-8"),
        file_name="workflow_report.md",
        mime="text/markdown",
        use_container_width=True,
    )

    if report_result.pdf_bytes is not None:
        st.download_button(
            label="Download PDF",
            data=report_result.pdf_bytes,
            file_name="workflow_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )