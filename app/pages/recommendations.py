"""Pipeline Recommendations page for the Streamlit thesis application."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    """Render the pipeline recommendation page."""

    st.title("Pipeline Recommendations")
    st.write("This page will present the recommended preprocessing and model choices.")
    st.info("The recommendation engine is not connected yet in this UI phase.")
    st.markdown(
        """
        ### Placeholder content
        Future content will display the recommended pipelines, their confidence,
        and the reasoning behind each suggestion.
        """
    )