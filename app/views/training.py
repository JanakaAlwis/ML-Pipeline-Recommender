"""Training page for the Streamlit thesis application."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.streamlit_app import render_header, render_wrapped_key_value_table
from src.application_state import has_dataset, has_profile, has_task_detection, initialize_session_state
from src.model_trainer import train_pipeline
from src.pipeline_builder import build_pipeline


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


def _bullet_points(items: list[str]) -> str:
    """Format a list as Markdown bullet points."""

    if not items:
        return "- Not specified"
    return "\n".join(f"- {item}" for item in items)


def render() -> None:
    """Render the training page."""

    initialize_session_state(st.session_state)

    render_header(
        "Train Selected Machine Learning Pipeline",
        "Build and fit the selected pipeline.",
        "After training, evaluate the model and inspect the results.",
        "This step turns the recommendation into a fitted model.",
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
    st.subheader("Selected pipeline summary")
    render_wrapped_key_value_table(
        [
            ("Selected Pipeline", selected_pipeline.pipeline_name),
            ("Machine Learning Model", _format_model_name(selected_pipeline.recommended_model_family)),
            ("Task Type", selected_pipeline.task_type),
            ("Confidence", f"{selected_pipeline.confidence_score:.2f}"),
            ("Complexity", selected_pipeline.estimated_training_complexity),
            ("Evaluation Metric / CV Metric", selected_pipeline.evaluation_metrics),
        ]
    )

    st.subheader("Preprocessing Steps")
    st.markdown(_bullet_points(selected_pipeline.preprocessing_steps))

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
        metrics_left, metrics_right, metrics_third = st.columns(3)
        metrics_left.metric("Trained Model", training_result.model_name or "N/A")
        metrics_right.metric("Training Time", f"{training_result.training_time_seconds:.3f} s" if training_result.training_time_seconds is not None else "N/A")
        metrics_third.metric("Cross-validation Metric", training_result.scoring_metric or "N/A")

        if training_result.cross_validation_scores:
            cv_table = pd.DataFrame({"fold": list(range(1, len(training_result.cross_validation_scores) + 1)), "cv_score": training_result.cross_validation_scores})
            st.subheader("Cross-validation scores")
            st.dataframe(cv_table, use_container_width=True)
            st.bar_chart(cv_table.set_index("fold"))

        st.write(f"Mean CV Score: {training_result.mean_cv_score:.3f}" if training_result.mean_cv_score is not None else "Mean CV Score: N/A")
        st.write(f"Standard Deviation: {training_result.std_cv_score:.3f}" if training_result.std_cv_score is not None else "Standard Deviation: N/A")

        st.info("Next: open Evaluation.")