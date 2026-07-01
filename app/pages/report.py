"""Report page for the Streamlit thesis application."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    """Render the report page."""

    st.title("Report")
    st.write("This page will later collect the workflow into a thesis-style report.")
    st.info("Report generation is not implemented in this skeleton.")
    st.markdown(
        """
        ### Placeholder content
        The final application will assemble an end-to-end report for the selected
        dataset, task, recommendation, training, and evaluation results.
        """
    )