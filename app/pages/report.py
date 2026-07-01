"""Report page for the Streamlit thesis application."""

from __future__ import annotations

import streamlit as st

from app.streamlit_app import render_header


def render() -> None:
    """Render the report page."""

    render_header(
        "Report",
        "A thesis-style summary of the completed workflow.",
        "Use this page after evaluation and explainability are complete.",
        "This page is a placeholder for a future report generator and does not alter the workflow.",
    )

    st.info("Report generation is not implemented yet.")
    st.write("Once the workflow is complete, this page will gather the upload, profile, task, training, evaluation, and explainability results into a final summary.")