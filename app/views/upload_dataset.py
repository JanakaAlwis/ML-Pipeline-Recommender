"""Upload Dataset page for the Streamlit thesis application."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.streamlit_app import render_header
from src.application_state import initialize_session_state, reset_workflow_state
from src.data_loader import load_csv


def render() -> None:
    """Render the dataset upload page."""

    initialize_session_state(st.session_state)

    render_header(
        "Upload Dataset",
        "Load a CSV file to start the workflow.",
        "After upload, generate a dataset profile.",
        "The uploaded dataframe is used by all later steps.",
    )

    st.write("Upload a CSV file to start.")

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
        st.info("Choose a CSV file, then continue to Dataset Profile.")
        return

    load_result = load_csv(uploaded_file)

    if not load_result.success or load_result.dataframe is None:
        for message in load_result.errors or load_result.messages:
            st.error(message)
        return

    reset_workflow_state(st.session_state)
    st.session_state["dataframe"] = load_result.dataframe

    st.success(f"Loaded '{load_result.source_name}' successfully.")
    summary_left, summary_right, summary_third = st.columns(3)
    summary_left.metric("Rows", load_result.rows)
    summary_right.metric("Columns", load_result.columns)
    summary_third.metric("Missing cells", int(load_result.dataframe.isna().sum().sum()))

    st.subheader("Dataset preview")
    st.dataframe(load_result.dataframe.head(10), use_container_width=True)

    st.subheader("Column types")
    column_types = pd.DataFrame({
        "column": load_result.dataframe.columns,
        "dtype": [str(dtype) for dtype in load_result.dataframe.dtypes],
    })
    st.dataframe(column_types, use_container_width=True)

    st.info("Next: open Dataset Profile.")