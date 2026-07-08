"""Pipeline Recommendations page for the Streamlit thesis application."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.streamlit_app import render_header, render_wrapped_key_value_table, render_wrapped_table
from src.application_state import has_profile, has_task_detection, initialize_session_state
from src.display_formatting import format_confidence
from src.recommendation_engine import recommend_pipelines


def _format_model_name(model_family: str) -> str:
    """Convert a model-family identifier into a readable ML model name."""

    normalized = model_family.replace("_", " ").strip().lower()
    mapping = {
        "logistic regression": "Logistic Regression",
        "random forest": "Random Forest",
        "decision tree": "Decision Tree",
        "knn": "KNN",
        "k neighbors": "KNN",
        "naive bayes": "Naive Bayes",
        "svm": "SVM",
        "support vector machine": "SVM",
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
    }
    return mapping.get(normalized, model_family.replace("_", " ").title())


def _explainability_rating(pipeline: object) -> str:
    """Return a simple explainability rating for a recommendation."""

    model_name = getattr(pipeline, "recommended_model_family", "").lower()
    if any(keyword in model_name for keyword in ["tree", "linear", "ridge", "lasso", "logistic", "naive", "knn"]):
        return "High"
    if any(keyword in model_name for keyword in ["svm", "randomforest"]):
        return "Medium"
    return "Low"


def _why_recommended(pipeline: object) -> str:
    """Create a short explanation for why the pipeline is recommended."""

    explanation = getattr(pipeline, "explanation", "") or "This pipeline matches the detected task and dataset characteristics."
    if explanation.endswith("."):
        return explanation
    return f"{explanation}."


def _preprocessing_strategy_items(preprocessing_steps: object) -> list[str]:
    """Return preprocessing steps as a concise bullet list."""

    if preprocessing_steps is None:
        return ["Not specified"]
    if isinstance(preprocessing_steps, (list, tuple, set)):
        return [str(step) for step in preprocessing_steps] or ["Not specified"]

    text = str(preprocessing_steps).strip()
    if not text:
        return ["Not specified"]

    parts = [part.strip() for part in text.replace(";", "\n").splitlines() if part.strip()]
    return parts or [text]


def render() -> None:
    """Render the pipeline recommendation page."""

    initialize_session_state(st.session_state)

    render_header(
        "Pipeline Recommendations",
        "Review pipeline candidates for the detected task.",
        "Select a pipeline, then continue to training.",
        "This page explains why each pipeline is recommended.",
    )

    if not has_profile(st.session_state):
        st.warning("Please generate the dataset profile first.")
        return

    if not has_task_detection(st.session_state):
        st.warning("Please detect the task type first.")
        return

    st.write("Review the recommended pipelines and choose one.")

    if st.button("Generate Recommendations"):
        recommendation_result = recommend_pipelines(
            st.session_state["profile_result"],
            st.session_state["task_result"],
        )
        st.session_state["recommendation_result"] = recommendation_result

        if not recommendation_result.success:
            for error in recommendation_result.errors:
                st.error(error)
            return

    recommendation_result = st.session_state.get("recommendation_result")
    if recommendation_result is None:
        st.markdown(
            """
            Generate recommendations to see the top pipeline candidates.
            """
        )
        return

    if not recommendation_result.success:
        for error in recommendation_result.errors:
            st.error(error)
        return

    top_pipelines = recommendation_result.top_pipelines[:5]
    if not top_pipelines:
        st.warning("No pipeline recommendations were generated.")
        return

    pipeline_options = {pipeline.pipeline_name: pipeline for pipeline in top_pipelines}
    selected_name = st.selectbox(
        "Select a pipeline",
        options=list(pipeline_options.keys()),
        key="recommendations_selected_pipeline",
    )
    selected_pipeline = pipeline_options[selected_name]
    st.session_state["selected_pipeline"] = selected_pipeline

    summary_table = pd.DataFrame(
        [
            {
                "Rank": pipeline.pipeline_rank,
                "Pipeline": pipeline.pipeline_name,
                "Task": pipeline.task_type,
                "ML Model": _format_model_name(pipeline.recommended_model_family),
                "Why Recommended": _why_recommended(pipeline),
                "Explainability": _explainability_rating(pipeline),
                "Confidence": format_confidence(pipeline.confidence_score),
                "Complexity": pipeline.estimated_training_complexity,
            }
            for pipeline in top_pipelines
        ]
    )[
        ["Rank", "Pipeline", "Task", "ML Model", "Why Recommended", "Explainability", "Confidence", "Complexity"]
    ]

    st.subheader("Top 5 pipelines")
    render_wrapped_table(summary_table)

    detail_left, detail_right, detail_third = st.columns(3)
    detail_left.metric("Selected rank", selected_pipeline.pipeline_rank)
    detail_right.metric("Confidence", format_confidence(selected_pipeline.confidence_score))
    detail_third.metric("Complexity", selected_pipeline.estimated_training_complexity)

    st.subheader("Selected pipeline summary")
    render_wrapped_key_value_table(
        [
            ("Pipeline Name", selected_pipeline.pipeline_name),
            ("ML Model", _format_model_name(selected_pipeline.recommended_model_family)),
            ("Confidence", format_confidence(selected_pipeline.confidence_score)),
            ("Why Recommended", _why_recommended(selected_pipeline)),
            ("Advantages", selected_pipeline.advantages),
            ("Limitations", selected_pipeline.limitations),
        ]
    )

    st.subheader("Selected preprocessing strategy")
    for step in _preprocessing_strategy_items(getattr(selected_pipeline, "preprocessing_steps", None)):
        st.markdown(f"- {step}")

    st.info("Next: open Training.")