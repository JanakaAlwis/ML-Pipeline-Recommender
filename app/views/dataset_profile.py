"""Dataset Profile page for the Streamlit thesis application."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app.streamlit_app import render_header
from src.application_state import has_dataset, initialize_session_state
from src.data_profiler import profile_dataset
from src.display_formatting import format_count, format_percentage
from src.missing_value_analysis import build_high_missingness_warnings, build_missing_value_analysis_frame, build_missing_value_overview


def _render_advanced_visualizations(profile_result: object) -> None:
    """Render optional advanced visualizations for numerical features."""

    with st.expander("Advanced Dataset Visualisations", expanded=False):
        numeric_columns = list(getattr(profile_result, "numeric_columns", []) or [])
        dataframe = getattr(profile_result, "dataframe", None)

        if dataframe is None or not isinstance(dataframe, pd.DataFrame):
            st.info("Advanced visualisations are not available because the profiled dataset is missing.")
            return

        st.subheader("Correlation Matrix")
        st.caption("Correlation values close to +1 or -1 indicate strong relationships between numerical features.")
        if len(numeric_columns) < 2:
            st.info("Correlation matrix requires at least two numerical columns.")
        else:
            correlation_matrix = dataframe[numeric_columns].corr(numeric_only=True)
            heatmap = px.imshow(
                correlation_matrix,
                text_auto=".2f",
                color_continuous_scale="RdBu",
                zmin=-1,
                zmax=1,
                aspect="auto",
                labels={"color": "Correlation"},
            )
            heatmap.update_layout(margin={"l": 10, "r": 10, "t": 30, "b": 10})
            st.plotly_chart(heatmap, use_container_width=True)

        st.subheader("Feature Distribution Histogram")
        st.caption("This chart shows how values are distributed for one numerical feature.")
        if not numeric_columns:
            st.info("Histogram requires at least one numerical column.")
        else:
            selected_histogram_column = st.selectbox(
                "Select a numerical feature for histogram",
                options=numeric_columns,
                key="profile_histogram_column",
            )
            histogram = px.histogram(
                dataframe,
                x=selected_histogram_column,
                nbins=30,
                title=f"Distribution of {selected_histogram_column}",
            )
            histogram.update_layout(margin={"l": 10, "r": 10, "t": 40, "b": 10})
            st.plotly_chart(histogram, use_container_width=True)

        st.subheader("Scatter Plot")
        st.caption("Scatter plots help inspect relationships between two numerical features.")
        if len(numeric_columns) < 2:
            st.info("Scatter plot requires at least two numerical columns.")
        else:
            scatter_left, scatter_right = st.columns(2)
            scatter_x = scatter_left.selectbox(
                "Select X-axis feature",
                options=numeric_columns,
                key="profile_scatter_x",
            )
            scatter_y_options = [column for column in numeric_columns if column != scatter_x] or numeric_columns
            scatter_y = scatter_right.selectbox(
                "Select Y-axis feature",
                options=scatter_y_options,
                key="profile_scatter_y",
            )
            scatter_plot = px.scatter(
                dataframe,
                x=scatter_x,
                y=scatter_y,
                title=f"{scatter_y} vs {scatter_x}",
                opacity=0.7,
            )
            scatter_plot.update_layout(margin={"l": 10, "r": 10, "t": 40, "b": 10})
            st.plotly_chart(scatter_plot, use_container_width=True)

        st.subheader("Pairwise Relationship View")
        st.caption("Optional overview of pairwise relationships for a small set of numerical features.")
        if len(numeric_columns) < 2:
            st.info("Pairwise view requires at least two numerical columns.")
        else:
            selected_pairwise_columns = st.multiselect(
                "Select up to 4 numerical features",
                options=numeric_columns,
                default=numeric_columns[: min(3, len(numeric_columns))],
                max_selections=4,
                key="profile_pairwise_columns",
            )
            if len(selected_pairwise_columns) < 2:
                st.info("Select at least two numerical features to display the pairwise view.")
            else:
                pairwise_plot = px.scatter_matrix(
                    dataframe,
                    dimensions=selected_pairwise_columns,
                    title="Pairwise relationships",
                )
                pairwise_plot.update_traces(diagonal_visible=False, showupperhalf=False)
                pairwise_plot.update_layout(height=700, margin={"l": 10, "r": 10, "t": 40, "b": 10})
                st.plotly_chart(pairwise_plot, use_container_width=True)


def _render_profile_results(profile_result: object) -> None:
    """Render the generated dataset profile summary."""

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
        missing_values_table = build_missing_value_analysis_frame(profile_result).copy()
        if "Missing percentage" in missing_values_table.columns:
            missing_values_table["Missing percentage"] = missing_values_table["Missing percentage"].map(format_percentage)
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
    if getattr(profile_result, "duplicate_row_count", 0) and getattr(profile_result, "duplicate_row_count", 0) > 0:
        st.caption(
            "Duplicate rows were detected. The system reports duplicates for user review. Duplicate removal is not applied automatically unless explicitly included in preprocessing."
        )
    else:
        st.caption("No duplicate rows were detected.")

    high_correlation_pairs_table = pd.DataFrame(
        profile_result.high_correlation_pairs,
        columns=["column_1", "column_2", "correlation"],
    )
    if len(profile_result.numeric_columns or []) < 2:
        st.info("Correlation analysis requires at least two numerical columns.")
    elif high_correlation_pairs_table.empty:
        st.info("No high correlations above the selected threshold were found.")
    else:
        high_correlation_pairs_table["correlation"] = high_correlation_pairs_table["correlation"].map(lambda value: f"{value:.4f}" if isinstance(value, (int, float)) else value)
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

    _render_advanced_visualizations(profile_result)


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
    overview_left.metric("Rows", format_count(dataframe.shape[0]))
    overview_right.metric("Columns", format_count(dataframe.shape[1]))
    overview_third.metric("Unique columns", format_count(len(dataframe.columns)))

    st.write("Columns in the uploaded dataset:")
    st.dataframe(pd.DataFrame({"column": list(dataframe.columns)}), use_container_width=True)

    if st.button("Generate Dataset Profile"):
        profile_result = profile_dataset(dataframe)
        st.session_state["profile_result"] = profile_result

        if not profile_result.success:
            for error in profile_result.errors:
                st.error(error)
            return

    profile_result = st.session_state.get("profile_result")
    if profile_result is not None:
        if not getattr(profile_result, "success", False):
            for error in getattr(profile_result, "errors", []) or []:
                st.error(error)
        else:
            _render_profile_results(profile_result)

    st.info("Next: open Task Detection.")