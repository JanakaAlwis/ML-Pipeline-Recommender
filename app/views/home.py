"""Home page for the Streamlit thesis application."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render() -> None:
    """Render the home page."""

    st.title("Explainable ML Pipeline Recommendation System")
    st.caption("Bachelor's Thesis Prototype")

    st.write(
        "This application automatically analyzes a structured tabular CSV dataset, recommends suitable machine learning pipelines, trains the selected model, evaluates performance, and explains the model results."
    )

    st.subheader("Supported Dataset")
    st.write(
        "The system supports CSV datasets that have a single header row, contain one observation per row, have one target column selected by the user, include numerical and/or categorical features, and are in structured tabular format."
    )

    st.subheader("Data Quality Support")
    st.write(
        "The system can analyze datasets with missing values, numerical features, categorical features, boolean features, class imbalance, correlated features, and outliers."
    )

    st.subheader("Workflow")
    st.markdown(
        "Upload Dataset → Dataset Profile → Task Detection → Pipeline Recommendation → Model Training → Model Evaluation → Explainability → Report Generation"
    )

    st.subheader("Limitations")
    st.write(
        "The system is not designed for multiple tables in one CSV, free-text documents, image datasets, audio datasets, time-series specific workflows, or graph or network datasets."
    )

    st.subheader("Technologies Used")
    technologies = pd.DataFrame(
        {
            "Technology": [
                "Python",
                "Streamlit",
                "Pandas",
                "Scikit-learn",
                "SHAP",
                "Plotly/Matplotlib",
                "Pytest",
            ]
        }
    )
    st.dataframe(technologies, use_container_width=True, hide_index=True)

    st.subheader("Getting Started")
    st.write("Select Upload Dataset from the left sidebar to begin.")