"""Pipeline Recommendations page for the Streamlit thesis application."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.streamlit_app import render_header, render_wrapped_key_value_table, render_wrapped_table
from src.application_state import has_profile, has_task_detection, initialize_session_state, invalidate_downstream_results
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
    """Create a concise presentation explanation from recommendation fields."""

    task_type = str(getattr(pipeline, "task_type", "") or "the detected").lower()
    model_family = str(getattr(pipeline, "recommended_model_family", "") or "").strip()
    model_name = _format_model_name(model_family) if model_family else "the selected model"
    model_key = model_family.replace("_", "").replace(" ", "").lower()

    model_reason = {
        "logisticregression": "an interpretable linear baseline",
        "randomforest": "a flexible ensemble for nonlinear patterns",
        "decisiontree": "a simple tree model for nonlinear patterns",
        "knn": "a distance-based alternative for local patterns",
        "kneighborsclassifier": "a distance-based alternative for local patterns",
        "gaussiannb": "a fast probabilistic baseline",
        "linearregression": "a clear linear baseline",
        "ridge": "a regularized linear model for correlated predictors",
        "lasso": "a regularized linear model that can reduce model complexity",
        "svr": "a margin-based model for nonlinear regression",
        "randomforestregressor": "a flexible ensemble for nonlinear regression",
        "decisiontreeregressor": "a tree model for nonlinear regression",
    }.get(model_key, model_name)

    details: list[str] = []
    missing_strategy = str(getattr(pipeline, "missing_value_strategy", "") or "").strip()
    if missing_strategy and "no special imputation" not in missing_strategy.lower():
        details.append(missing_strategy.rstrip("."))

    scaling_strategy = str(getattr(pipeline, "scaling_strategy", "") or "").strip()
    if scaling_strategy:
        details.append(scaling_strategy.rstrip("."))

    imbalance_strategy = str(getattr(pipeline, "imbalance_handling_strategy", "") or "").strip()
    if imbalance_strategy:
        details.append(imbalance_strategy.rstrip("."))

    dimensionality_strategy = str(getattr(pipeline, "dimensionality_reduction_strategy", "") or "").strip()
    if dimensionality_strategy and "no dimensionality" not in dimensionality_strategy.lower():
        details.append(dimensionality_strategy.rstrip("."))

    explanation = f"Recommended as {model_name} for {task_type}; it provides {model_reason}."
    if details:
        explanation += f" Strategy: {'; '.join(details)}."
    return explanation


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
    previous_pipeline = st.session_state.get("selected_pipeline")
    if previous_pipeline is not None and (
        getattr(previous_pipeline, "pipeline_name", None) != getattr(selected_pipeline, "pipeline_name", None)
        or getattr(previous_pipeline, "recommended_model_family", None)
        != getattr(selected_pipeline, "recommended_model_family", None)
    ):
        invalidate_downstream_results(st.session_state)
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