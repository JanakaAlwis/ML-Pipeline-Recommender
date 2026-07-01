"""Explainability page for the Streamlit thesis application."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    """Render the explainability page."""

    st.title("Explainability")
    st.write("This page will later explain why a pipeline and model were selected.")
    st.info("Explainability outputs are intentionally not connected yet.")
    st.markdown(
        """
        ### Placeholder content
        The final version will summarize feature influence, preprocessing decisions,
        and other human-readable explanations.
        """
    )