"""Training page for the Streamlit thesis application."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd
import streamlit as st

from app.streamlit_app import render_header, render_wrapped_key_value_table, render_wrapped_table
from src.application_state import has_dataset, has_profile, has_task_detection, initialize_session_state
from src.model_trainer import train_pipeline
from src.pipeline_builder import build_pipeline
from src.missing_value_analysis import build_high_missingness_warnings, build_missing_value_overview
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


def _normalize_text(value: str | None) -> str:
    """Normalize text for simple keyword matching."""

    return "" if value is None else "".join(character for character in value.lower() if character.isalnum())


def _selected_pipeline_summary_rows(selected_pipeline: Any, target_column: str | None, task_type: str | None) -> list[tuple[str, object]]:
    """Build the compact summary for the selected pipeline."""

    return [
        ("Selected Pipeline", getattr(selected_pipeline, "pipeline_name", "N/A")),
        ("Machine Learning Model", _format_model_name(getattr(selected_pipeline, "recommended_model_family", ""))),
        ("Target Variable", target_column or "Not specified"),
        ("Detected Learning Task", task_type or getattr(selected_pipeline, "task_type", "N/A")),
        ("Preprocessing Steps", getattr(selected_pipeline, "preprocessing_steps", []) or []),
        ("Confidence", f"{getattr(selected_pipeline, 'confidence_score', 0.0):.2f}"),
        ("Complexity", getattr(selected_pipeline, "estimated_training_complexity", "N/A")),
        ("Evaluation Metric / CV Metric", getattr(selected_pipeline, "evaluation_metrics", []) or []),
    ]


def _stratified_sampling_label(task_type: str | None, target: pd.Series) -> str:
    """Explain whether stratified sampling was used for the train/test split."""

    if task_type != "classification":
        return "Not applicable"

    class_counts = Counter(target.dropna())
    if len(class_counts) < 2:
        return "No"
    if min(class_counts.values()) < 2:
        return "No"
    return "Yes"


def _split_information_rows(training_result: Any, dataframe: pd.DataFrame, target_column: str, task_type: str | None) -> list[tuple[str, object]]:
    """Summarize the train/test split used during training."""

    total_rows = len(dataframe)
    train_rows = getattr(training_result, "train_row_count", None)
    test_rows = getattr(training_result, "test_row_count", None)
    train_percentage = (train_rows / total_rows * 100.0) if train_rows is not None and total_rows else None
    test_percentage = (test_rows / total_rows * 100.0) if test_rows is not None and total_rows else None
    stratified_label = _stratified_sampling_label(task_type, dataframe[target_column])

    return [
        ("Original dataset rows", total_rows),
        ("Training dataset rows", train_rows if train_rows is not None else "N/A"),
        ("Testing dataset rows", test_rows if test_rows is not None else "N/A"),
        ("Train/Test split ratio", f"{TRAIN_TEST_RATIO_TEXT} ({_format_percentage(train_percentage)} / {_format_percentage(test_percentage)})"),
        ("Random state", TRAIN_TEST_RANDOM_STATE),
        ("Stratified split", stratified_label),
    ]


def _missing_after_preprocessing(
    training_result: Any,
    dataframe: pd.DataFrame,
    target_column: str,
) -> int | None:
    """Estimate remaining missing values after fitted preprocessing."""

    pipeline = getattr(training_result, "trained_pipeline", None)
    if pipeline is None or not hasattr(pipeline, "named_steps"):
        return None

    preprocessor = pipeline.named_steps.get("preprocessor") if hasattr(pipeline, "named_steps") else None
    if preprocessor is None:
        return None

    features = dataframe.drop(columns=[target_column])
    try:
        transformed = preprocessor.transform(features)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        transformed_frame = pd.DataFrame(transformed)
        return int(transformed_frame.isna().sum().sum())
    except Exception:
        return None


def _rows_removed_for_missing_target(dataframe: pd.DataFrame, target_column: str, training_result: Any) -> int:
    """Compute how many rows were removed because the target value was missing."""

    trained_rows = 0
    if getattr(training_result, "train_row_count", None) is not None:
        trained_rows += int(training_result.train_row_count)
    if getattr(training_result, "test_row_count", None) is not None:
        trained_rows += int(training_result.test_row_count)
    if trained_rows <= 0:
        return 0
    return max(int(len(dataframe) - trained_rows), 0)


def _build_cross_validation_table(training_result: Any) -> pd.DataFrame:
    """Create a table with individual fold scores."""

    scores = getattr(training_result, "cross_validation_scores", []) or []
    if not scores:
        return pd.DataFrame(columns=["Fold", "Score"])
    return pd.DataFrame({"Fold": list(range(1, len(scores) + 1)), "Score": scores})


def _model_parameter_rows(training_result: Any) -> list[tuple[str, object]]:
    """Extract important selected-model parameters for display."""

    pipeline = getattr(training_result, "trained_pipeline", None)
    steps = getattr(pipeline, "steps", []) or []
    if not steps:
        return [("Model parameters", "Not available")]

    estimator = steps[-1][1]
    estimator_name = estimator.__class__.__name__
    params = estimator.get_params(deep=False) if hasattr(estimator, "get_params") else {}

    rows: list[tuple[str, object]] = []

    def add_row(label: str, key: str) -> None:
        value = params.get(key, "Not specified") if isinstance(params, dict) else "Not specified"
        rows.append((label, value))

    model_key = _normalize_text(estimator_name)

    if model_key == "logisticregression":
        add_row("C", "C")
        add_row("Penalty", "penalty")
        add_row("Solver", "solver")
        add_row("Max iterations", "max_iter")
    elif model_key in {"decisiontreeclassifier", "decisiontreeregressor"}:
        add_row("Criterion", "criterion")
        add_row("Max depth", "max_depth")
        add_row("Min samples leaf", "min_samples_leaf")
    elif model_key in {"randomforestclassifier", "randomforestregressor"}:
        add_row("Number of estimators", "n_estimators")
        add_row("Max depth", "max_depth")
        add_row("Min samples leaf", "min_samples_leaf")
    elif model_key == "kneighborsclassifier":
        add_row("Number of neighbors", "n_neighbors")
        add_row("Weights", "weights")
        add_row("Distance metric", "metric")
    elif model_key == "svc":
        add_row("C", "C")
        add_row("Kernel", "kernel")
        add_row("Gamma", "gamma")
    elif model_key == "ridge":
        add_row("Alpha", "alpha")
        add_row("Fit intercept", "fit_intercept")
        add_row("Solver", "solver")
    elif model_key == "lasso":
        add_row("Alpha", "alpha")
        add_row("Selection", "selection")
        add_row("Max iterations", "max_iter")
    elif model_key == "linearregression":
        add_row("Fit intercept", "fit_intercept")
        add_row("Positive coefficients", "positive")
        add_row("Number of jobs", "n_jobs")
    elif model_key == "sgdregressor":
        add_row("Alpha", "alpha")
        add_row("Penalty", "penalty")
        add_row("Initial learning rate", "eta0")
        add_row("Max iterations", "max_iter")
    else:
        add_row("Parameters", "Not available in this view")

    for step_name, step in steps:
        if _normalize_text(step.__class__.__name__) == "polynomialfeatures":
            degree = step.get_params(deep=False).get("degree", "Not specified") if hasattr(step, "get_params") else "Not specified"
            rows.append(("Polynomial degree", degree))
            break

    return rows


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
        _selected_pipeline_summary_rows(
            selected_pipeline,
            st.session_state.get("target_column"),
            st.session_state.get("task_result").task_type if st.session_state.get("task_result") is not None else None,
        )
    )

    missing_value_overview = build_missing_value_overview(st.session_state["profile_result"])
    st.subheader("Missing Value Handling")
    if missing_value_overview["has_missing_values"]:
        analysis_frame = missing_value_overview["analysis_frame"]
        render_wrapped_key_value_table(
            [
                ("Original dataset rows", len(st.session_state["dataframe"])),
                ("Missing values detected", "Yes"),
                ("Imputation applied", "Yes"),
                ("Numerical imputation strategy", missing_value_overview["numerical_strategy"]),
                ("Categorical imputation strategy", missing_value_overview["categorical_strategy"]),
                ("Affected columns", missing_value_overview["columns"]),
                ("Features recommended for dropping", missing_value_overview["drop_feature_columns"] or ["None"]),
                ("Total missing values before preprocessing", missing_value_overview["total_missing_values"]),
            ]
        )
        st.caption(missing_value_overview["explanation"])
        with st.expander("View missing value actions by column", expanded=False):
            visible_columns = [
                "Column",
                "Data type",
                "Missing percentage",
                "Recommended action",
                "Applied action during preprocessing",
            ]
            render_wrapped_table(analysis_frame[visible_columns], hide_index=True)
        for warning_message in build_high_missingness_warnings(st.session_state["profile_result"]):
            st.warning(warning_message)
    else:
        st.info("No missing value handling was required.")

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

        st.subheader("Preprocessing Transparency Summary")
        missing_after_preprocessing = _missing_after_preprocessing(
            training_result,
            st.session_state["dataframe"],
            st.session_state["target_column"],
        )
        rows_removed_missing_target = _rows_removed_for_missing_target(
            st.session_state["dataframe"],
            st.session_state["target_column"],
            training_result,
        )
        confirmation_text = (
            "Missing values were successfully handled by the selected preprocessing pipeline."
            if missing_after_preprocessing == 0
            else "Missing value handling completed. Please review the remaining missing values count below."
        )
        if not missing_value_overview["has_missing_values"]:
            confirmation_text = "No missing values were detected, so no missing value handling was required."

        render_wrapped_key_value_table(
            [
                ("Original dataset rows", len(st.session_state["dataframe"])),
                ("Missing values before preprocessing", missing_value_overview["total_missing_values"]),
                (
                    "Missing values remaining after preprocessing",
                    missing_after_preprocessing if missing_after_preprocessing is not None else "Not available",
                ),
                ("Rows removed because of missing target values", rows_removed_missing_target),
                ("Confirmation", confirmation_text),
            ]
        )
        st.caption("Rows are not removed because of missing values. Row-count changes come from the train/test split.")

        st.subheader("Dataset Processing Summary")
        render_wrapped_key_value_table(
            [
                ("Original dataset rows", len(st.session_state["dataframe"])),
                ("Feature columns removed", missing_value_overview["drop_feature_columns"] or ["None"]),
                ("Missing value handling applied", missing_value_overview["strategy_summary"]),
                ("Rows removed because of missing target values", rows_removed_missing_target),
                ("Training dataset size", training_result.train_row_count if training_result.train_row_count is not None else "N/A"),
                ("Testing dataset size", training_result.test_row_count if training_result.test_row_count is not None else "N/A"),
                ("Train/Test split ratio", TRAIN_TEST_RATIO_TEXT),
            ]
        )

        st.subheader("Dataset Split Summary")
        render_wrapped_key_value_table(
            _split_information_rows(
                training_result,
                st.session_state["dataframe"],
                st.session_state["target_column"],
                st.session_state.get("task_result").task_type if st.session_state.get("task_result") is not None else None,
            )
        )

        st.subheader("Cross-validation summary")
        cv_summary_left, cv_summary_middle, cv_summary_right = st.columns(3)
        cv_summary_left.metric("Mean CV score", f"{training_result.mean_cv_score:.3f}" if training_result.mean_cv_score is not None else "N/A")
        cv_summary_middle.metric("CV standard deviation", f"{training_result.std_cv_score:.3f}" if training_result.std_cv_score is not None else "N/A")
        cv_summary_right.metric("Number of folds", str(len(training_result.cross_validation_scores)) if training_result.cross_validation_scores else "N/A")
        st.write(f"Scoring metric used: {training_result.scoring_metric or 'N/A'}")

        with st.expander("View individual fold scores"):
            cv_table = _build_cross_validation_table(training_result)
            if cv_table.empty:
                st.info("Cross-validation scores were not available for this training run.")
            else:
                render_wrapped_table(cv_table, hide_index=True)

        st.subheader("Selected model parameters")
        with st.expander("View important model parameters"):
            render_wrapped_key_value_table(_model_parameter_rows(training_result))

        st.info("Next: open Evaluation.")