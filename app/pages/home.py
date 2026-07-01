"""Home page for the Streamlit thesis application."""

from __future__ import annotations

import streamlit as st

from app.streamlit_app import APP_ICON, APP_SUBTITLE, APP_TITLE, render_header, render_workflow_progress


def render() -> None:
    """Render the home page."""

    render_header(APP_TITLE, APP_SUBTITLE, "Upload a CSV dataset to begin the workflow.", "This app guides you through a full thesis demo from data upload to explanation.")

    st.markdown(f"### {APP_ICON} What this prototype does")
    st.write("It helps you upload a tabular dataset, detect the machine learning task, recommend a pipeline, train it, evaluate it, and explain the result.")

    render_workflow_progress()

    left_column, right_column = st.columns(2)
    with left_column:
        st.subheader("Workflow")
        st.markdown(
            """
            1. Upload CSV
            2. Profile data
            3. Detect task
            4. Recommend pipeline
            5. Train model
            6. Evaluate model
            7. Explain model
            """
        )
    with right_column:
        st.subheader("Thesis focus")
        st.info("The interface is designed to support a clear defense narrative: data understanding, automatic recommendation, and explainable results.")

    st.markdown("### Start here")
    st.write("Use the sidebar to move to Upload Dataset and begin the workflow.")