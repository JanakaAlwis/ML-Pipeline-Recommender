"""Home page for the Streamlit thesis application."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    """Render the home page."""

    st.title("ML Pipeline Recommender")
    st.write("This is the thesis project dashboard for the machine learning pipeline workflow.")
    st.info("Use the sidebar to move through dataset upload, profiling, recommendation, training, and evaluation.")
    st.markdown(
        """
        ### Placeholder content
        The backend workflow is connected in later phases. This page will eventually
        introduce the project goals, workflow steps, and thesis context.
        """
    )