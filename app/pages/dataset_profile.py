"""Dataset Profile page for the Streamlit thesis application."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.application_state import has_dataset, initialize_session_state
from src.data_profiler import profile_dataset


def render() -> None:
    """Render the dataset profiling page."""

    initialize_session_state(st.session_state)

    st.title("Dataset Profile")
    st.write("This page will summarize dataset structure, missing values, and feature types.")

    if not has_dataset(st.session_state):
        st.warning("Please upload a dataset first.")
        return

    dataframe = st.session_state["dataframe"]
    if dataframe is None:
        st.warning("Please upload a dataset first.")
        return

    st.subheader("Dataset Overview")
    st.write(f"Shape: {dataframe.shape[0]} rows x {dataframe.shape[1]} columns")
    st.write("Columns:")
    st.write(list(dataframe.columns))

    if st.button("Generate Dataset Profile"):
        profile_result = profile_dataset(dataframe)
        st.session_state["profile_result"] = profile_result

        if not profile_result.success:
            for error in profile_result.errors:
                st.error(error)
            return

        st.success("Dataset profile generated successfully.")
        st.write(f"Row count: {profile_result.row_count}")
        st.write(f"Column count: {profile_result.column_count}")
        st.write(f"Numeric columns: {profile_result.numeric_columns}")
        st.write(f"Categorical columns: {profile_result.categorical_columns}")
        st.write(f"Boolean columns: {profile_result.boolean_columns}")
        st.write(f"Datetime columns: {profile_result.datetime_columns}")

        missing_values_table = pd.DataFrame(
            list(profile_result.missing_values.items()),
            columns=["column", "missing_count"],
        )
        st.subheader("Missing Values")
        st.dataframe(missing_values_table, use_container_width=True)

        st.subheader("Duplicate Rows")
        st.write(profile_result.duplicate_row_count)

        high_correlation_pairs_table = pd.DataFrame(
            profile_result.high_correlation_pairs,
            columns=["column_1", "column_2", "correlation"],
        )
        st.subheader("High Correlation Pairs")
        st.dataframe(high_correlation_pairs_table, use_container_width=True)

        outlier_counts_table = pd.DataFrame(
            list(profile_result.outlier_counts.items()),
            columns=["column", "outlier_count"],
        )
        st.subheader("Outlier Counts")
        st.dataframe(outlier_counts_table, use_container_width=True)

        st.subheader("Sample Rows")
        if profile_result.sample_rows is not None:
            st.dataframe(profile_result.sample_rows, use_container_width=True)
        else:
            st.write("No sample rows available.")

    st.markdown(
        """
        ### Placeholder content
        The profile is generated on demand and stored in session state for the
        next workflow steps.
        """
    )