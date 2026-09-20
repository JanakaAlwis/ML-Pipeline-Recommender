"""Evaluation page for the Streamlit thesis application."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, get_scorer, precision_score, r2_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from app.streamlit_app import render_header, render_wrapped_key_value_table, render_wrapped_table
from src.application_state import has_dataset, has_task_detection, has_training_result, initialize_session_state, invalidate_downstream_results
from src.display_formatting import format_count, format_cv_score, format_metric
from src.model_evaluator import evaluate_model


EVALUATION_TEST_SIZE = 0.2
EVALUATION_RANDOM_STATE = 42


def _metric_display_name(metric_name: str | None) -> str:
    """Return a readable label for a scoring metric."""

    if metric_name == "f1_weighted":
        return "F1-weighted"
    if metric_name == "r2":
        return "R²"
    return metric_name or "N/A"


def _main_metric_label(task_type: str | None) -> str:
    """Return the primary evaluation metric label for the task."""

    if task_type == "regression":
        return "R²"
    return "Accuracy"


def _build_stratify_target(task_type: str | None, target_series: pd.Series) -> pd.Series | None:
    """Return a safe stratification target for the hold-out split."""

    if task_type != "classification":
        return None

    class_counts = Counter(target_series.dropna())
    if len(class_counts) < 2 or min(class_counts.values()) < 2:
        return None
    return target_series


def _split_holdout_data(
    dataframe: pd.DataFrame,
    target_column: str,
    task_type: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create the same hold-out split used by the trainer."""

    features = dataframe.drop(columns=[target_column])
    target = dataframe[target_column]
    stratify = _build_stratify_target(task_type, target)
    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=EVALUATION_TEST_SIZE,
        random_state=EVALUATION_RANDOM_STATE,
        stratify=stratify,
    )
    return train_features, test_features, train_target, test_target


def _holdout_split_from_training_result(training_result: Any, dataframe: pd.DataFrame, target_column: str, task_type: str | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return the hold-out split captured during training when available."""

    train_features = getattr(training_result, "train_features", None)
    test_features = getattr(training_result, "test_features", None)
    train_target = getattr(training_result, "train_target", None)
    test_target = getattr(training_result, "test_target", None)

    if all(value is not None for value in (train_features, test_features, train_target, test_target)):
        return train_features, test_features, train_target, test_target

    return _split_holdout_data(dataframe, target_column, task_type)


def _score_pipeline(training_result: Any, features: pd.DataFrame, target: pd.Series, scoring_metric: str | None) -> float | None:
    """Score the trained pipeline using the requested scoring metric."""

    if scoring_metric is None:
        return None

    pipeline = getattr(training_result, "trained_pipeline", None)
    if pipeline is None:
        return None

    try:
        scorer = get_scorer(scoring_metric)
        return float(scorer(pipeline, features, target))
    except Exception:
        return None


def _predictions_on_holdout(training_result: Any, features: pd.DataFrame) -> pd.Series:
    """Predict labels for the hold-out split."""

    pipeline = getattr(training_result, "trained_pipeline", None)
    if pipeline is None:
        return pd.Series(dtype=object)
    return pd.Series(pipeline.predict(features))


def _binary_roc_auc(training_result: Any, features: pd.DataFrame, target: pd.Series) -> float | None:
    """Compute ROC-AUC for binary classification when scores are available."""

    pipeline = getattr(training_result, "trained_pipeline", None)
    if pipeline is None or target.nunique(dropna=True) != 2:
        return None

    score_values: Any | None = None
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(features)
        if getattr(probabilities, "ndim", 0) == 2 and probabilities.shape[1] >= 2:
            score_values = probabilities[:, 1]
    elif hasattr(pipeline, "decision_function"):
        score_values = pipeline.decision_function(features)

    if score_values is None:
        return None

    try:
        return float(roc_auc_score(target, score_values))
    except Exception:
        return None


def _build_classification_metrics_table(
    y_true: pd.Series,
    y_pred: pd.Series,
    roc_auc_value: float | None,
) -> pd.DataFrame:
    """Build a concise classification metrics table."""

    metrics_rows = [
        ("Accuracy", format_metric(accuracy_score(y_true, y_pred))),
        ("Precision (weighted)", format_metric(precision_score(y_true, y_pred, average="weighted", zero_division=0))),
        ("Recall (weighted)", format_metric(recall_score(y_true, y_pred, average="weighted", zero_division=0))),
        ("Macro F1", format_metric(f1_score(y_true, y_pred, average="macro", zero_division=0))),
        ("Weighted F1", format_metric(f1_score(y_true, y_pred, average="weighted", zero_division=0))),
    ]
    if roc_auc_value is not None:
        metrics_rows.append(("ROC-AUC", format_metric(roc_auc_value)))

    return pd.DataFrame(metrics_rows, columns=["Metric", "Value"])


def _build_regression_metrics_table(evaluation_result: Any) -> pd.DataFrame:
    """Build a concise regression metrics table."""

    regression_metrics = getattr(evaluation_result, "regression_metrics", {}) or {}
    ordered_metrics = ["mae", "mse", "rmse", "r2"]
    rows = []
    for metric_name in ordered_metrics:
        if metric_name in regression_metrics:
            rows.append((metric_name.upper() if metric_name != "r2" else "R²", format_metric(regression_metrics[metric_name])))
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def _build_classification_report_frame(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    """Build a readable classification report table."""

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_frame = pd.DataFrame(report).transpose()
    columns = [column for column in ["precision", "recall", "f1-score", "support"] if column in report_frame.columns]
    visible_rows = [index for index in report_frame.index if index not in {"accuracy"}]
    report_frame = report_frame.loc[visible_rows, columns]
    if "support" in report_frame.columns:
        report_frame["support"] = report_frame["support"].map(format_count)
    for column in ["precision", "recall", "f1-score"]:
        if column in report_frame.columns:
            report_frame[column] = report_frame[column].map(format_metric)
    return report_frame


def _build_confusion_matrix_frame(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    """Build a labeled confusion matrix table."""

    labels = sorted(pd.unique(pd.concat([y_true, y_pred])))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    label_text = [str(label) for label in labels]
    return pd.DataFrame(matrix, index=label_text, columns=label_text)


def _build_cross_validation_table(training_result: Any) -> pd.DataFrame:
    """Build the cross-validation fold table."""

    scores = getattr(training_result, "cross_validation_scores", []) or []
    return pd.DataFrame(
        {"Fold": list(range(1, len(scores) + 1)), "Score": scores},
    )


def _build_generalization_summary(
    training_result: Any,
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    train_target: pd.Series,
    test_target: pd.Series,
) -> dict[str, float | str | None]:
    """Compute training and test scores for the configured scoring metric."""

    scoring_metric = getattr(training_result, "scoring_metric", None)
    train_score = _score_pipeline(training_result, train_features, train_target, scoring_metric)
    test_score = _score_pipeline(training_result, test_features, test_target, scoring_metric)
    gap = (train_score - test_score) if train_score is not None and test_score is not None else None
    return {
        "scoring_metric": scoring_metric,
        "train_score": train_score,
        "test_score": test_score,
        "generalization_gap": gap,
        "train_row_count": int(len(train_features)),
        "test_row_count": int(len(test_features)),
    }


def _interpret_generalization(generalization_gap: float | None, scoring_metric: str | None) -> str:
    """Create a short interpretation of the generalization gap."""

    metric_label = _metric_display_name(scoring_metric)
    if generalization_gap is None:
        return "Generalization could not be computed for this run."
    if generalization_gap <= 0.05:
        return f"The training and test scores are close, so the model appears stable on the {metric_label} metric."
    if generalization_gap <= 0.15:
        return f"The model shows a moderate gap between training and test performance on {metric_label}, so some overfitting is possible."
    return f"The model performs noticeably better on training data than on test data, which suggests overfitting on the {metric_label} metric."


def _evaluation_summary_rows(
    training_result: Any,
    task_type: str | None,
    main_score: float | None,
    rows_evaluated: int | None,
) -> list[tuple[str, object]]:
    """Build the rows shown in the overall evaluation summary."""

    return [
        ("Model Name", getattr(training_result, "model_name", "N/A")),
        ("Learning Task", task_type or "N/A"),
        ("Main Evaluation Score", format_metric(main_score)),
        ("Mean CV Score", format_cv_score(getattr(training_result, "mean_cv_score", None))),
        ("CV Standard Deviation", format_cv_score(getattr(training_result, "std_cv_score", None))),
        ("Rows Evaluated", format_count(rows_evaluated)),
    ]


def _model_information_rows(training_result: Any, split_summary: dict[str, float | str | None]) -> list[tuple[str, object]]:
    """Build the model information table shown on the page."""

    train_rows = split_summary.get("train_row_count")
    test_rows = split_summary.get("test_row_count")
    split_text = f"80/20 ({format_count(train_rows)} train / {format_count(test_rows)} test)"
    return [
        ("Selected Algorithm", getattr(training_result, "model_name", "N/A")),
        ("Learning Task", getattr(training_result, "task_type", "N/A")),
        ("Target Variable", getattr(training_result, "target_column", "N/A")),
        ("Train/Test Split", split_text),
        ("Random State", EVALUATION_RANDOM_STATE),
        ("Scoring Metric", _metric_display_name(getattr(training_result, "scoring_metric", None))),
    ]


def _render_classification_section(
    training_result: Any,
    test_features: pd.DataFrame,
    test_target: pd.Series,
    holdout_predictions: pd.Series,
    split_summary: dict[str, float | str | None],
) -> None:
    """Render the classification evaluation details."""

    roc_auc_value = _binary_roc_auc(training_result, test_features, test_target)

    metrics_table = _build_classification_metrics_table(test_target, holdout_predictions, roc_auc_value)
    st.subheader("Classification Evaluation")
    st.caption("Metrics and the confusion matrix use the held-out test set.")
    render_wrapped_table(metrics_table, hide_index=True)

    detailed_metrics = _build_classification_report_frame(test_target, holdout_predictions)
    confusion_matrix_frame = _build_confusion_matrix_frame(test_target, holdout_predictions)

    with st.expander("Classification report", expanded=False):
        render_wrapped_table(detailed_metrics, hide_index=False)

    with st.expander("Confusion matrix", expanded=False):
        st.dataframe(confusion_matrix_frame.style.background_gradient(cmap="Blues"), use_container_width=True)

    _render_generalization_section(split_summary)


def _render_regression_section(
    training_result: Any,
    split_summary: dict[str, float | str | None],
    evaluation_result: Any,
) -> None:
    """Render the regression evaluation details."""

    metrics_table = _build_regression_metrics_table(evaluation_result)

    st.subheader("Regression Evaluation")
    st.caption("Metrics below use the held-out test set.")
    render_wrapped_table(metrics_table, hide_index=True)
    st.caption("Lower MAE, MSE, and RMSE are better. Higher R² is better.")

    _render_generalization_section(split_summary)


def _render_generalization_section(split_summary: dict[str, float | str | None]) -> None:
    """Render the generalization performance summary."""

    scoring_metric = split_summary.get("scoring_metric")
    train_score = split_summary.get("train_score")
    test_score = split_summary.get("test_score")
    generalization_gap = split_summary.get("generalization_gap")

    st.subheader("Generalization Performance")
    summary_left, summary_middle, summary_right = st.columns(3)
    summary_left.metric(f"Training score ({_metric_display_name(scoring_metric)})", format_metric(train_score))
    summary_middle.metric(f"Test score ({_metric_display_name(scoring_metric)})", format_metric(test_score))
    summary_right.metric("Generalization gap", format_metric(generalization_gap))

    st.write(_interpret_generalization(generalization_gap if isinstance(generalization_gap, (int, float)) else None, scoring_metric if isinstance(scoring_metric, str) else None))


def _render_cross_validation_section(training_result: Any) -> None:
    """Render the cross-validation summary."""

    scores = getattr(training_result, "cross_validation_scores", []) or []
    st.subheader("Cross-Validation Summary")

    summary_left, summary_middle, summary_right = st.columns(3)
    summary_left.metric("Mean CV score", format_cv_score(getattr(training_result, "mean_cv_score", None)))
    summary_middle.metric("CV standard deviation", format_cv_score(getattr(training_result, "std_cv_score", None)))
    summary_right.metric("Number of folds", format_count(len(scores)) if scores else "N/A")

    with st.expander("Fold scores", expanded=False):
        fold_scores_table = _build_cross_validation_table(training_result)
        if fold_scores_table.empty:
            st.info("Cross-validation scores are not available for this run.")
        else:
            render_wrapped_table(fold_scores_table, hide_index=True)


def _render_interpretation(evaluation_result: Any, training_result: Any) -> None:
    """Render a short plain-English interpretation of the evaluation output."""

    if getattr(evaluation_result, "task_type", None) == "regression":
        main_score = getattr(evaluation_result, "regression_metrics", {}).get("r2")
        score_text = f"R² of {format_metric(main_score)}"
    else:
        main_score = getattr(evaluation_result, "classification_metrics", {}).get("accuracy")
        score_text = f"accuracy of {format_metric(main_score)}"

    st.subheader("Interpretation")
    st.write(
        f"The evaluation shows a {score_text}. The cross-validation and generalization summary help judge whether the model is stable or whether its performance drops on unseen data."
    )


def render() -> None:
    """Render the evaluation page."""

    initialize_session_state(st.session_state)

    render_header(
        "Evaluation",
        "Review the fitted model using a concise hold-out summary.",
        "Focus on results rather than training details.",
        "These metrics show how well the model performs on unseen data.",
    )

    if not has_dataset(st.session_state):
        st.warning("Please upload a dataset first.")
        return

    if not has_training_result(st.session_state):
        st.warning("Please train a model first.")
        return

    if not has_task_detection(st.session_state):
        st.warning("Please detect the task type first.")
        return

    if st.session_state.get("target_column") is None:
        st.warning("Please select a target column first.")
        return

    st.write("Evaluate the trained pipeline.")

    if st.button("Evaluate Model"):
        invalidate_downstream_results(st.session_state, include_training=False)
        evaluation_result = evaluate_model(
            st.session_state["training_result"],
            st.session_state["dataframe"],
            st.session_state["target_column"],
            st.session_state["task_result"].task_type,
        )
        st.session_state["evaluation_result"] = evaluation_result

        if not evaluation_result.success:
            for error in evaluation_result.errors:
                st.error(error)
            return

        st.success("Evaluation completed successfully.")

    evaluation_result = st.session_state.get("evaluation_result")
    if evaluation_result is None:
        st.info("Run the evaluation to view model results.")
        return

    if not evaluation_result.success:
        for error in evaluation_result.errors:
            st.error(error)
        return

    training_result = st.session_state["training_result"]
    dataframe = st.session_state["dataframe"]
    target_column = st.session_state["target_column"]
    task_type = st.session_state["task_result"].task_type

    train_features, test_features, train_target, test_target = _holdout_split_from_training_result(training_result, dataframe, target_column, task_type)
    holdout_predictions = _predictions_on_holdout(training_result, test_features)
    main_score = accuracy_score(test_target, holdout_predictions) if task_type == "classification" else r2_score(test_target, holdout_predictions)
    split_summary = _build_generalization_summary(training_result, train_features, test_features, train_target, test_target)

    summary_rows = _evaluation_summary_rows(training_result, task_type, main_score, split_summary.get("test_row_count") if isinstance(split_summary.get("test_row_count"), int) else None)
    st.subheader("Overall Evaluation Summary")
    summary_top_left, summary_top_middle, summary_top_right = st.columns(3)
    summary_bottom_left, summary_bottom_middle, summary_bottom_right = st.columns(3)

    summary_top_left.metric(summary_rows[0][0], summary_rows[0][1])
    summary_top_middle.metric(summary_rows[1][0], summary_rows[1][1])
    summary_top_right.metric(summary_rows[2][0], summary_rows[2][1])
    summary_bottom_left.metric(summary_rows[3][0], summary_rows[3][1])
    summary_bottom_middle.metric(summary_rows[4][0], summary_rows[4][1])
    summary_bottom_right.metric(summary_rows[5][0], summary_rows[5][1])

    st.subheader("Model Information")
    render_wrapped_key_value_table(_model_information_rows(training_result, split_summary))

    _render_cross_validation_section(training_result)

    if task_type == "classification":
        _render_classification_section(training_result, test_features, test_target, holdout_predictions, split_summary)
    else:
        _render_regression_section(training_result, split_summary, evaluation_result)

    _render_interpretation(evaluation_result, training_result)

    st.info("Next: open Explainability.")