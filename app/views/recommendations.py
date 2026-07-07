"""Pipeline Recommendations page for the Streamlit thesis application."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.streamlit_app import render_header, render_wrapped_key_value_table, render_wrapped_table
from src.application_state import has_profile, has_task_detection, initialize_session_state
from src.missing_value_analysis import build_missing_value_overview
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


def render() -> None:
    """Render the pipeline recommendation page."""

    initialize_session_state(st.session_state)

    render_header(
        "Pipeline Recommendations",
        "Review pipeline candidates for the detected task.",
        "Select a pipeline, then continue to training.",
        "This page shows the most suitable pipeline options.",
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

    missing_value_overview = build_missing_value_overview(st.session_state["profile_result"])
    st.subheader("Missing Value Handling")
    if missing_value_overview["has_missing_values"]:
        st.warning(
            "Missing values were detected in the dataset. Imputation is recommended before model training to ensure reliable model performance."
        )
        st.write(missing_value_overview["strategy_summary"])
        st.caption(missing_value_overview["explanation"])
        with st.expander("View missing value handling details"):
            analysis_frame = missing_value_overview["analysis_frame"]
            visible_columns = [
                "Column",
                "Data type",
                "Missing percentage",
                "Recommended action",
                "Applied action during preprocessing",
            ]
            render_wrapped_table(analysis_frame[visible_columns], hide_index=True)
            render_wrapped_key_value_table(
                [
                    ("Detected missing values", "Yes"),
                    ("Affected columns", missing_value_overview["columns"]),
                    ("Features recommended for dropping", missing_value_overview["drop_feature_columns"] or ["None"]),
                    ("Features recommended for imputation", missing_value_overview["imputed_feature_columns"] or ["None"]),
                    ("Features recommended for review", missing_value_overview["review_feature_columns"] or ["None"]),
                ]
            )
    else:
        st.info("No missing values were detected. Missing value handling is not required.")

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
                "Confidence": pipeline.confidence_score,
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
    detail_right.metric("Confidence", f"{selected_pipeline.confidence_score:.2f}")
    detail_third.metric("Complexity", selected_pipeline.estimated_training_complexity)

    st.subheader("Selected pipeline summary")
    render_wrapped_key_value_table(
        [
            ("Pipeline Name", selected_pipeline.pipeline_name),
            ("ML Model", _format_model_name(selected_pipeline.recommended_model_family)),
            ("Confidence", f"{selected_pipeline.confidence_score:.2f}"),
            ("Why Recommended", _why_recommended(selected_pipeline)),
            ("Advantages", selected_pipeline.advantages),
            ("Limitations", selected_pipeline.limitations),
            ("Preprocessing Steps", selected_pipeline.preprocessing_steps),
            ("Expected Evaluation Metrics", selected_pipeline.evaluation_metrics),
        ]
    )

    st.info("Next: open Training.")