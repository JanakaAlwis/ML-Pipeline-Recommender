"""Dataset Profile page for the Streamlit thesis application."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    """Render the dataset profiling page."""

    st.title("Dataset Profile")
    st.write("This page will summarize dataset structure, missing values, and feature types.")
    st.info("Profiling logic is intentionally left as a placeholder for now.")
    st.markdown(
        """
        ### Placeholder content
        Later phases will display profiling statistics, warnings, and a quick visual
        summary of the uploaded dataset.
        """
    )