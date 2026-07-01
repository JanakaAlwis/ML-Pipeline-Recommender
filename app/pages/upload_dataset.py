"""Upload Dataset page for the Streamlit thesis application."""

from __future__ import annotations

import streamlit as st

from src.application_state import initialize_session_state, reset_workflow_state
from src.data_loader import load_csv


def render() -> None:
    """Render the dataset upload page."""

    initialize_session_state(st.session_state)

    st.title("Upload Dataset")
    st.write("Upload a CSV file to start the thesis workflow.")
    st.info("Only CSV files are supported at this stage.")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        accept_multiple_files=False,
        key="uploaded_csv_file",
    )

    if st.button("Clear uploaded dataset"):
        st.session_state.pop("uploaded_csv_file", None)
        reset_workflow_state(st.session_state)
        st.success("Uploaded dataset cleared.")
        st.rerun()

    if uploaded_file is None:
        st.markdown(
            """
            ### Placeholder content
            After you upload a CSV, the dataframe will be stored in session state
            and the downstream workflow will be reset.
            """
        )
        return

    load_result = load_csv(uploaded_file)

    if not load_result.success or load_result.dataframe is None:
        for message in load_result.errors or load_result.messages:
            st.error(message)
        return

    reset_workflow_state(st.session_state)
    st.session_state["dataframe"] = load_result.dataframe

    st.success(f"Loaded '{load_result.source_name}' successfully.")
    st.write(f"Rows: {load_result.rows}, Columns: {load_result.columns}")
    st.dataframe(load_result.dataframe.head(10), use_container_width=True)