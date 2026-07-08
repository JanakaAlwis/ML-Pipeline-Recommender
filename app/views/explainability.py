"""Explainability page for the Streamlit thesis application."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app.streamlit_app import render_header, render_wrapped_key_value_table, render_wrapped_table
from src.application_state import (
    has_dataset,
    has_evaluation_result,
    has_recommendations,
    has_task_detection,
    has_training_result,
    initialize_session_state,
)
from src.explainability import SHAP_DEFAULT_SAMPLE_SIZE, explain_model, generate_shap_explanation


def _format_model_name(model_family: str) -> str:
    """Convert a model-family identifier into a readable model name."""

    normalized = model_family.replace("_", " ").strip().lower()
    mapping = {
        "logistic regression": "Logistic Regression",
        "random forest": "Random Forest",
        "decision tree": "Decision Tree",
        "knn": "KNN",
        "naive bayes": "Naive Bayes",
        "svm": "SVM",
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


def _metric_display_value(evaluation_result: object) -> tuple[str, str, str | None]:
    """Return the primary metric label, value, and optional secondary metric text."""

    task_type = getattr(evaluation_result, "task_type", None)
    if task_type == "regression":
        regression_metrics = getattr(evaluation_result, "regression_metrics", {}) or {}
        r2_value = regression_metrics.get("r2")
        rmse_value = regression_metrics.get("rmse")
        primary_value = f"{r2_value:.3f}" if isinstance(r2_value, (int, float)) else "N/A"
        secondary_text = f"RMSE: {rmse_value:.3f}" if isinstance(rmse_value, (int, float)) else None
        return "R²", primary_value, secondary_text

    classification_metrics = getattr(evaluation_result, "classification_metrics", {}) or {}
    accuracy_value = classification_metrics.get("accuracy")
    primary_value = f"{accuracy_value:.3f}" if isinstance(accuracy_value, (int, float)) else "N/A"
    return "Accuracy", primary_value, None


def _mean_cv_score(training_result: object) -> str:
    """Format the cross-validation score for display."""

    mean_score = getattr(training_result, "mean_cv_score", None)
    if isinstance(mean_score, (int, float)):
        return f"{mean_score:.3f}"
    return "N/A"


def _training_time(training_result: object) -> str:
    """Format the training time for display."""

    training_time = getattr(training_result, "training_time_seconds", None)
    if isinstance(training_time, (int, float)):
        return f"{training_time:.3f} s"
    return "N/A"


def _count_numeric_columns(profile_result: object) -> int:
    """Count numeric columns in the profiled dataset."""

    return len(getattr(profile_result, "numeric_columns", []) or [])


def _count_categorical_columns(profile_result: object) -> int:
    """Count categorical columns in the profiled dataset."""

    return len(getattr(profile_result, "categorical_columns", []) or [])


def _missing_ratio(profile_result: object) -> float:
    """Calculate the overall missing-value ratio."""

    missing_values = getattr(profile_result, "missing_values", {}) or {}
    row_count = getattr(profile_result, "row_count", 0) or 0
    column_count = getattr(profile_result, "column_count", 0) or 0
    total_cells = row_count * column_count
    if total_cells <= 0:
        return 0.0
    return sum(int(value or 0) for value in missing_values.values()) / total_cells


def _recommendation_reason_items(profile_result: object, selected_pipeline: object, task_result: object) -> list[str]:
    """Build readable reasons for why the pipeline was recommended."""

    reasons: list[str] = []

    task_type = getattr(task_result, "task_type", "unknown")
    reasons.append(f"The task was detected as a {task_type} problem, so the system chose a pipeline suited to that task.")

    row_count = getattr(profile_result, "row_count", None)
    column_count = getattr(profile_result, "column_count", None)
    if row_count is not None and column_count is not None:
        reasons.append(f"The dataset is structured tabular data with {row_count} rows and {column_count} columns.")

    numeric_count = _count_numeric_columns(profile_result)
    categorical_count = _count_categorical_columns(profile_result)
    if numeric_count >= categorical_count:
        reasons.append("The dataset is mostly numerical, which supports models that work well with numeric features.")
    elif categorical_count > 0:
        reasons.append("The dataset contains categorical information, so preprocessing and encoding are included in the pipeline.")

    if _missing_ratio(profile_result) <= 0.1:
        reasons.append("The dataset has low missingness, which keeps the training pipeline stable and predictable.")

    preprocessing_steps = getattr(selected_pipeline, "preprocessing_steps", []) or []
    preprocessing_text = " ".join(preprocessing_steps).lower()
    if "scal" in preprocessing_text:
        reasons.append("Scaling is included because it can improve performance for the selected model family.")

    model_family = getattr(selected_pipeline, "recommended_model_family", "")
    readable_model = _format_model_name(model_family)
    if any(keyword in model_family.lower() for keyword in ["logistic", "linear", "ridge", "lasso", "tree", "naive", "knn"]):
        reasons.append(f"{readable_model} is comparatively easy to interpret, which supports a thesis-friendly explanation.")

    explanation = getattr(selected_pipeline, "explanation", "")
    if explanation:
        reasons.append(explanation.rstrip("." ) + ".")

    evidence = getattr(selected_pipeline, "evidence", []) or []
    for item in evidence:
        reasons.append(str(item).rstrip(".") + ".")

    deduplicated: list[str] = []
    for reason in reasons:
        cleaned = reason.strip()
        if cleaned and cleaned not in deduplicated:
            deduplicated.append(cleaned)
    return deduplicated


def _importance_level(score: float, max_score: float) -> str:
    """Map a relative importance score to a qualitative level."""

    if max_score <= 0:
        return "Low"

    ratio = score / max_score
    if ratio >= 0.75:
        return "Very High"
    if ratio >= 0.5:
        return "High"
    if ratio >= 0.25:
        return "Medium"
    return "Low"


def _feature_importance_table(top_features: list[tuple[str, float]]) -> pd.DataFrame:
    """Create a presentation table for feature importance values."""

    if not top_features:
        return pd.DataFrame(columns=["Rank", "Feature", "Importance Score", "Influence Level"])

    max_score = max(score for _, score in top_features)
    rows = []
    for rank, (feature, score) in enumerate(top_features, start=1):
        rows.append(
            {
                "Rank": rank,
                "Feature": feature,
                "Importance Score": f"{score:.4f}",
                "Influence Level": _importance_level(score, max_score),
            }
        )
    return pd.DataFrame(rows, columns=["Rank", "Feature", "Importance Score", "Influence Level"])


def _feature_importance_chart_frame(top_features: list[tuple[str, float]]) -> pd.DataFrame:
    """Create a dataframe that is suitable for a horizontal bar chart."""

    if not top_features:
        return pd.DataFrame(columns=["Feature", "Importance"])

    chart_frame = pd.DataFrame(top_features, columns=["Feature", "Importance"])
    return chart_frame.sort_values("Importance", ascending=True)


def _interpretation_text(top_features: list[tuple[str, float]]) -> str:
    """Build a short plain-English interpretation of the ranked features."""

    if not top_features:
        return "The model explainability step did not return ranked feature importance values."

    top_names = [feature for feature, _ in top_features[:3]]
    top_names_text = ", ".join(top_names)
    if len(top_features) == 1:
        return (
            f"The trained model relies primarily on {top_names_text}. "
            "This feature has the strongest influence on the model's predictions, while other features contribute less."
        )

    return (
        f"The trained model relies primarily on {top_names_text}. "
        "The highest-ranked features contribute most to the predictions, and lower-ranked features have progressively smaller influence."
    )


def _pipeline_explanation_rows(selected_pipeline: object, task_result: object) -> list[tuple[str, object]]:
    """Build a short beginner-friendly pipeline explanation table."""

    preprocessing_steps = getattr(selected_pipeline, "preprocessing_steps", []) or []
    preprocessing_text = ", ".join(str(step) for step in preprocessing_steps) if isinstance(preprocessing_steps, (list, tuple, set)) else str(preprocessing_steps)
    model_name = _format_model_name(getattr(selected_pipeline, "recommended_model_family", ""))
    if any(keyword in getattr(selected_pipeline, "recommended_model_family", "").lower() for keyword in ["tree", "forest", "xgb"]):
        importance_text = "Intrinsic tree structure plus estimated permutation importance and optional SHAP"
    else:
        importance_text = "Estimated permutation importance; SHAP is not supported for this model"

    return [
        ("Selected Model", model_name),
        ("Preprocessing", preprocessing_text or "Not specified"),
        ("Why Explainable", "The pipeline is readable because preprocessing is explicit and feature importance can be inspected."),
        ("Importance Method", importance_text),
        ("Task Type", getattr(task_result, "task_type", "N/A")),
    ]


def _technical_detail_rows(explainability_result: object, shap_result: object | None) -> list[tuple[str, object]]:
    """Build compact technical details for the expander."""

    rows: list[tuple[str, object]] = [
        ("Explainability method", getattr(explainability_result, "method_used", "N/A") or "N/A"),
        ("Top features shown", len(getattr(explainability_result, "top_features", []) or [])),
        ("Sample size used for feature importance", "Up to 5,000 rows"),
    ]

    if shap_result is None:
        rows.append(("SHAP status", "Not generated yet"))
        return rows

    rows.extend(
        [
            ("SHAP status", "Success" if getattr(shap_result, "success", False) else "Unavailable"),
            ("SHAP sample size", getattr(shap_result, "sample_size", "N/A") or "N/A"),
            ("SHAP method", getattr(shap_result, "method_used", "N/A") or "N/A"),
        ]
    )
    return rows


def render() -> None:
    """Render the explainability page."""

    initialize_session_state(st.session_state)

    render_header(
        "Explainability",
        "Show why the pipeline was recommended and how the trained model behaves.",
        "Review the pipeline reasoning, feature importance, and optional SHAP.",
        "This page keeps explainability compact and beginner-friendly.",
    )

    if not has_dataset(st.session_state):
        st.warning("Please upload a dataset first.")
        return

    if not has_recommendations(st.session_state):
        st.warning("Please generate pipeline recommendations first.")
        return

    if not has_task_detection(st.session_state):
        st.warning("Please detect the task type first.")
        return

    if not has_training_result(st.session_state):
        st.warning("Please train a model first.")
        return

    if not has_evaluation_result(st.session_state):
        st.warning("Please evaluate the model first.")
        return

    if st.session_state.get("selected_pipeline") is None:
        st.warning("Please select a pipeline from the recommendations first.")
        return

    if st.session_state.get("target_column") is None:
        st.warning("Please select a target column first.")
        return

    selected_pipeline = st.session_state["selected_pipeline"]
    training_result = st.session_state["training_result"]
    evaluation_result = st.session_state["evaluation_result"]
    profile_result = st.session_state["profile_result"]
    task_result = st.session_state["task_result"]

    st.write("Generate the explanation for the trained model.")

    if st.button("Generate Explainability"):
        explainability_result = explain_model(
            training_result,
            st.session_state["dataframe"],
            st.session_state["target_column"],
            task_result.task_type,
        )
        st.session_state["explainability_result"] = explainability_result
        st.session_state.pop("shap_explanation_result", None)

        if not explainability_result.success:
            st.warning(explainability_result.errors[0] if explainability_result.errors else "Explainability could not be generated.")
            return

    explainability_result = st.session_state.get("explainability_result")
    if explainability_result is None:
        st.info("Generate explainability to see feature importance and optional SHAP output.")
        return

    if not explainability_result.success:
        st.warning(explainability_result.errors[0] if explainability_result.errors else "Feature importance could not be generated.")
        return

    primary_metric_label, primary_metric_value, secondary_metric_text = _metric_display_value(evaluation_result)
    summary_left, summary_middle, summary_right = st.columns(3)
    summary_left.metric("Selected Pipeline", selected_pipeline.pipeline_name)
    summary_middle.metric("Machine Learning Model", _format_model_name(selected_pipeline.recommended_model_family))
    summary_right.metric("Task Type", task_result.task_type)

    performance_left, performance_middle, performance_right = st.columns(3)
    performance_left.metric(primary_metric_label, primary_metric_value)
    performance_middle.metric("Cross Validation Score", _mean_cv_score(training_result))
    performance_right.metric("Training Time", _training_time(training_result))
    if secondary_metric_text is not None:
        st.caption(secondary_metric_text)

    st.subheader("Explainability Summary")
    render_wrapped_key_value_table(
        [
            ("Selected Pipeline", selected_pipeline.pipeline_name),
            ("Machine Learning Model", _format_model_name(selected_pipeline.recommended_model_family)),
            ("Task Type", task_result.task_type),
            (primary_metric_label, primary_metric_value),
            ("Cross Validation Score", _mean_cv_score(training_result)),
            ("Training Time", _training_time(training_result)),
        ]
    )

    st.subheader("Pipeline Explanation")
    render_wrapped_key_value_table(_pipeline_explanation_rows(selected_pipeline, task_result))
    st.caption("Feature importance is estimated with permutation importance. SHAP is optional and only available for supported tree-based models.")

    st.subheader("Feature Importance")
    st.write("Feature importance shows which input features changed the model output the most when they were varied in the sampled data.")

    if explainability_result.top_features:
        top_features = explainability_result.top_features[:10]
        display_table = _feature_importance_table(top_features)
        chart_frame = _feature_importance_chart_frame(top_features)
        if not chart_frame.empty:
            feature_chart = px.bar(
                chart_frame,
                x="Importance",
                y="Feature",
                orientation="h",
                text_auto=".3f",
                title="Top Features",
            )
            feature_chart.update_layout(margin={"l": 10, "r": 10, "t": 40, "b": 10}, yaxis_title="", xaxis_title="Importance")
            st.plotly_chart(feature_chart, use_container_width=True)
        render_wrapped_table(display_table, hide_index=True)
    else:
        st.info("Feature importance could not be calculated for this model. A SHAP explanation may still be available for supported models.")

    st.subheader("SHAP Explanation")
    st.warning("SHAP explanations may take time for large datasets. A sample of the data will be used.")

    shap_sample_size = st.slider("SHAP sample size", min_value=50, max_value=1000, value=SHAP_DEFAULT_SAMPLE_SIZE, step=50)
    shap_result = st.session_state.get("shap_explanation_result")

    if st.button("Generate SHAP Explanation"):
        shap_result = generate_shap_explanation(
            training_result,
            st.session_state["dataframe"],
            st.session_state["target_column"],
            task_result.task_type,
            sample_size=shap_sample_size,
        )
        st.session_state["shap_explanation_result"] = shap_result

        if not shap_result.success:
            st.warning(shap_result.errors[0] if shap_result.errors else "SHAP explanation is not available for this model. Feature importance is shown instead.")

    shap_result = st.session_state.get("shap_explanation_result")
    if shap_result is not None and shap_result.success:
        st.caption(shap_result.explanation_summary or "")
        if shap_result.warnings:
            for warning in shap_result.warnings:
                st.caption(warning)

        shap_features_df = pd.DataFrame(shap_result.top_features, columns=["Feature", "SHAP Importance"])
        shap_features_df = shap_features_df.sort_values("SHAP Importance", ascending=True)
        shap_chart = px.bar(
            shap_features_df,
            x="SHAP Importance",
            y="Feature",
            orientation="h",
            text_auto=".3f",
            title="Top SHAP Features",
        )
        shap_chart.update_layout(margin={"l": 10, "r": 10, "t": 40, "b": 10}, yaxis_title="", xaxis_title="SHAP Importance")
        st.plotly_chart(shap_chart, use_container_width=True)
        st.write(shap_result.interpretation)
    elif shap_result is not None and not shap_result.success:
        st.info(shap_result.errors[0] if shap_result.errors else "SHAP explanation is not available for this model. Feature importance is shown instead.")
    else:
        st.info("Run SHAP only if you want an additional optional explanation for a supported model.")

    st.subheader("Top Feature Interpretation")
    st.write(_interpretation_text(explainability_result.top_features))

    with st.expander("Technical Details", expanded=False):
        render_wrapped_key_value_table(_technical_detail_rows(explainability_result, shap_result))
        st.write("Pipeline reason summary:")
        for reason in _recommendation_reason_items(profile_result, selected_pipeline, task_result):
            st.markdown(f"- {reason}")

    st.info("Next: open Report.")