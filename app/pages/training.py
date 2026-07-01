"""Training page for the Streamlit thesis application."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    """Render the training page."""

    st.title("Training")
    st.write("This page will trigger pipeline training in a later phase.")
    st.info("Model fitting is intentionally not wired in this skeleton.")
    st.markdown(
        """
        ### Placeholder content
        The finished application will show training progress, fitted pipeline details,
        and cross-validation summaries.
        """
    )