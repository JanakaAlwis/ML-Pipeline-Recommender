"""Task Detection page for the Streamlit thesis application."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    """Render the task detection page."""

    st.title("Task Detection")
    st.write("This page will determine whether the problem is classification or regression.")
    st.info("The detection engine will be connected later; this page is only a shell.")
    st.markdown(
        """
        ### Placeholder content
        The final version will show the detected task, confidence, and the reasoning
        used to choose between classification and regression.
        """
    )