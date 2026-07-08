"""Training page for the Streamlit thesis application."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from app.streamlit_app import render_header, render_wrapped_key_value_table, render_wrapped_table
from src.application_state import has_dataset, has_profile, has_task_detection, initialize_session_state
from src.display_formatting import format_count, format_cv_score, format_metric
from src.model_trainer import train_pipeline
from src.pipeline_builder import build_pipeline

TRAIN_TEST_RANDOM_STATE = 42
TRAIN_TEST_RATIO_TEXT = "80% / 20%"


def _format_model_name(model_family: str) -> str:
    """Convert a model-family identifier into a readable ML model name."""

    normalized = model_family.replace("_", " ").strip().lower()
    mapping = {
        "logistic regression": "Logistic Regression",
        "random forest": "Random Forest",
        "decision tree": "Decision Tree",
        "knn": "KNN",
        "naive bayes": "Naive Bayes",
        "svm": "SVM",
        "xgboost": "XGBoost",
        "linear regression": "Linear Regression",
        "ridge": "Ridge",
        "lasso": "Lasso",
        "polynomial regression": "Polynomial Regression",
        "random forest regressor": "Random Forest Regressor",
        "logisticregression": "Logistic Regression",
        "randomforest": "Random Forest",
        "decisiontree": "Decision Tree",
        "kneighborsclassifier": "KNN",
        "gaussiannb": "Naive Bayes",
        "linearregression": "Linear Regression",
        "randomforestregressor": "Random Forest Regressor",
        "decisiontreeregressor": "Decision Tree Regressor",
        "xgbclassifier": "XGBoost",
        "xgbregressor": "XGBoost",
    }
    return mapping.get(normalized, model_family.replace("_", " ").title())


def _format_percentage(value: float | None) -> str:
    """Format a percentage value for display."""

    if value is None:
        return "N/A"
    return f"{value:.0f}%"


def _metric_display_name(metric_name: str | None) -> str:
    """Return a readable label for a scoring metric."""

    if metric_name == "f1_weighted":
        return "F1-weighted"
    if metric_name == "r2":
        return "R²"
    return metric_name or "N/A"


def _normalize_text(value: str | None) -> str:
    """Normalize text for simple keyword matching."""

    return "" if value is None else "".join(character for character in value.lower() if character.isalnum())


def _build_cross_validation_table(training_result: Any) -> pd.DataFrame:
    """Create a table with individual fold scores."""

    scores = getattr(training_result, "cross_validation_scores", []) or []
    if not scores:
        return pd.DataFrame(columns=["Fold", "Score"])
    return pd.DataFrame({"Fold": list(range(1, len(scores) + 1)), "Score": scores})


def _training_summary_rows(training_result: Any, dataframe: pd.DataFrame, selected_pipeline: Any) -> list[tuple[str, object]]:
    """Build the single training summary table shown after fitting."""

    total_rows = len(dataframe)
    train_rows = getattr(training_result, "train_row_count", None)
    test_rows = getattr(training_result, "test_row_count", None)
    split_text = (
        f"{TRAIN_TEST_RATIO_TEXT}"
        if train_rows is None or test_rows is None
        else f"{TRAIN_TEST_RATIO_TEXT} ({train_rows} train / {test_rows} test)"
    )

    scoring_metric = getattr(training_result, "scoring_metric", None)
    if getattr(training_result, "mean_cv_score", None) is not None:
        cv_summary = f"Mean {_metric_display_name(scoring_metric)}: {format_cv_score(training_result.mean_cv_score)}"
        if getattr(training_result, "std_cv_score", None) is not None:
            cv_summary = f"{cv_summary}; std: {format_cv_score(training_result.std_cv_score)}"
        if getattr(training_result, "cross_validation_scores", None):
            cv_summary = f"{cv_summary}; folds: {format_count(len(training_result.cross_validation_scores))}"
    else:
        cv_summary = "Not available"

    preprocessing_steps = getattr(selected_pipeline, "preprocessing_steps", []) or []
    if isinstance(preprocessing_steps, (list, tuple, set)):
        preprocessing_text = ", ".join(str(step) for step in preprocessing_steps) or "Not specified"
    else:
        preprocessing_text = str(preprocessing_steps) if preprocessing_steps else "Not specified"

    return [
        ("Selected Pipeline", getattr(selected_pipeline, "pipeline_name", "N/A")),
        ("Machine Learning Model", _format_model_name(getattr(selected_pipeline, "recommended_model_family", ""))),
        ("Applied Preprocessing", preprocessing_text),
        ("Features Removed", "None"),
        ("Missing Value Handling Applied", f"{total_rows} rows processed; missing target rows removed before split if present"),
        ("Train/Test Split", split_text),
        ("Cross-validation Summary", cv_summary),
        ("Training Status", "Completed successfully"),
    ]


def render() -> None:
    """Render the training page."""

    initialize_session_state(st.session_state)

    render_header(
        "Train Selected Machine Learning Pipeline",
        "Build and fit the selected pipeline.",
        "This page explains how the model is trained.",
        "After training, evaluate the model and inspect the results.",
    )

    if not has_dataset(st.session_state):
        st.warning("Please upload a dataset first.")
        return

    if not has_profile(st.session_state):
        st.warning("Please generate the dataset profile first.")
        return

    if not has_task_detection(st.session_state):
        st.warning("Please detect the task type first.")
        return

    if st.session_state.get("selected_pipeline") is None:
        st.warning("Please select a pipeline from the recommendations first.")
        return

    if st.session_state.get("target_column") is None:
        st.warning("Please select a target column first.")
        return

    selected_pipeline = st.session_state["selected_pipeline"]

    st.write("Train the selected pipeline.")

    if st.button("Train Selected Pipeline"):
        built_pipeline_result = build_pipeline(
            st.session_state["selected_pipeline"],
            st.session_state["profile_result"],
            st.session_state["task_result"],
        )
        st.session_state["built_pipeline_result"] = built_pipeline_result

        if not built_pipeline_result.success:
            for error in built_pipeline_result.errors:
                st.error(error)
            return

        training_result = train_pipeline(
            built_pipeline_result,
            st.session_state["dataframe"],
            st.session_state["target_column"],
            st.session_state["task_result"].task_type,
        )
        st.session_state["training_result"] = training_result

        if not training_result.success:
            for error in training_result.errors:
                st.error(error)
            return

        st.success("Training completed successfully.")
        for warning in training_result.warnings:
            st.warning(warning)

        metrics_left, metrics_right, metrics_third = st.columns(3)
        metrics_left.metric("Trained Model", training_result.model_name or "N/A")
        metrics_right.metric("Training Time", f"{training_result.training_time_seconds:.3f} s" if training_result.training_time_seconds is not None else "N/A")
        metrics_third.metric("Cross-validation Metric", training_result.scoring_metric or "N/A")

        st.subheader("Training Summary")
        render_wrapped_key_value_table(
            _training_summary_rows(
                training_result,
                st.session_state["dataframe"],
                selected_pipeline,
            )
        )

        st.subheader("Cross-validation summary")
        cv_summary_left, cv_summary_middle, cv_summary_right = st.columns(3)
        cv_summary_left.metric("Mean CV score", format_cv_score(training_result.mean_cv_score))
        cv_summary_middle.metric("CV standard deviation", format_cv_score(training_result.std_cv_score))
        cv_summary_right.metric("Number of folds", format_count(len(training_result.cross_validation_scores)) if training_result.cross_validation_scores else "N/A")
        st.write(f"Scoring metric used: {training_result.scoring_metric or 'N/A'}")

        with st.expander("View individual fold scores"):
            cv_table = _build_cross_validation_table(training_result)
            if cv_table.empty:
                st.info("Cross-validation scores were not available for this training run.")
            else:
                render_wrapped_table(cv_table, hide_index=True)

        st.info("Next: open Evaluation.")