"""Dataset Profile page for the Streamlit thesis application."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.streamlit_app import render_header
from src.application_state import has_dataset, initialize_session_state
from src.data_profiler import profile_dataset
from src.missing_value_analysis import build_high_missingness_warnings, build_missing_value_analysis_frame, build_missing_value_overview


def render() -> None:
    """Render the dataset profiling page."""

    initialize_session_state(st.session_state)

    render_header(
        "Dataset Profile",
        "Inspect data quality before selecting the target.",
        "Use the result to choose the target column and detect the task type.",
        "This summary helps explain the dataset before modeling.",
    )

    if not has_dataset(st.session_state):
        st.warning("Please upload a dataset first.")
        return

    dataframe = st.session_state["dataframe"]
    if dataframe is None:
        st.warning("Please upload a dataset first.")
        return

    st.subheader("Dataset overview")
    overview_left, overview_right, overview_third = st.columns(3)
    overview_left.metric("Rows", dataframe.shape[0])
    overview_right.metric("Columns", dataframe.shape[1])
    overview_third.metric("Unique columns", len(dataframe.columns))

    st.write("Columns in the uploaded dataset:")
    st.dataframe(pd.DataFrame({"column": list(dataframe.columns)}), use_container_width=True)

    if st.button("Generate Dataset Profile"):
        profile_result = profile_dataset(dataframe)
        st.session_state["profile_result"] = profile_result

        if not profile_result.success:
            for error in profile_result.errors:
                st.error(error)
            return

        st.success("Dataset profile generated successfully.")
        profile_left, profile_right, profile_third = st.columns(3)
        profile_left.metric("Rows", profile_result.row_count)
        profile_right.metric("Columns", profile_result.column_count)
        profile_third.metric("Duplicate rows", profile_result.duplicate_row_count)

        feature_types_table = pd.DataFrame(
            [
                {"feature_type": "Numeric", "columns": ", ".join(profile_result.numeric_columns) or "None"},
                {"feature_type": "Categorical", "columns": ", ".join(profile_result.categorical_columns) or "None"},
                {"feature_type": "Boolean", "columns": ", ".join(profile_result.boolean_columns) or "None"},
                {"feature_type": "Datetime", "columns": ", ".join(profile_result.datetime_columns) or "None"},
            ]
        )
        st.subheader("Column types")
        st.dataframe(feature_types_table, use_container_width=True)

        missing_value_overview = build_missing_value_overview(profile_result)
        st.subheader("Missing Value Analysis")
        if missing_value_overview["has_missing_values"]:
            missing_summary_left, missing_summary_right = st.columns(2)
            missing_summary_left.metric("Total missing values", missing_value_overview["total_missing_values"])
            missing_summary_right.metric("Affected columns", missing_value_overview["affected_column_count"])
            st.write(missing_value_overview["strategy_summary"])
            st.caption(missing_value_overview["explanation"])
            missing_values_table = build_missing_value_analysis_frame(profile_result)
            visible_columns = [
                "Column",
                "Data type",
                "Missing percentage",
                "Recommended action",
                "Applied action during preprocessing",
            ]
            st.dataframe(missing_values_table[visible_columns], use_container_width=True, hide_index=True)
            for warning_message in build_high_missingness_warnings(profile_result):
                st.warning(warning_message)
        else:
            st.info("No missing values were detected. No missing value handling was required.")

        st.subheader("Duplicates")
        st.write(profile_result.duplicate_row_count)

        high_correlation_pairs_table = pd.DataFrame(
            profile_result.high_correlation_pairs,
            columns=["column_1", "column_2", "correlation"],
        )
        st.subheader("High correlations")
        st.dataframe(high_correlation_pairs_table, use_container_width=True)

        outlier_counts_table = pd.DataFrame(
            list(profile_result.outlier_counts.items()),
            columns=["column", "outlier_count"],
        )
        st.subheader("Outliers")
        st.dataframe(outlier_counts_table, use_container_width=True)

        if profile_result.target_distribution is not None and not profile_result.target_distribution.empty:
            st.subheader("Target distribution")
            st.bar_chart(profile_result.target_distribution)

        st.subheader("Sample rows")
        if profile_result.sample_rows is not None:
            st.dataframe(profile_result.sample_rows, use_container_width=True)
        else:
            st.write("No sample rows available.")

    st.info("Next: open Task Detection.")