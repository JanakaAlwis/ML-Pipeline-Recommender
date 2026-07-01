"""Evaluation page for the Streamlit thesis application."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    """Render the evaluation page."""

    st.title("Evaluation")
    st.write("This page will show evaluation metrics for a trained pipeline.")
    st.info("Evaluation logic will be connected after the backend workflow is ready.")
    st.markdown(
        """
        ### Placeholder content
        The final page will present test metrics, confusion matrices, and regression
        error summaries in a thesis-friendly format.
        """
    )