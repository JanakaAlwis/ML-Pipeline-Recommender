"""Session state helpers for the Streamlit application.

This module keeps the application workflow state in one place so the Streamlit
pages can share a consistent set of keys.
"""

from __future__ import annotations

from typing import Any, MutableMapping

EXPECTED_SESSION_KEYS: tuple[str, ...] = (
    "dataframe",
    "target_column",
    "profile_result",
    "task_result",
    "recommendation_result",
    "selected_pipeline",
    "built_pipeline_result",
    "training_result",
    "evaluation_result",
)

DOWNSTREAM_RESULT_KEYS: tuple[str, ...] = (
    "built_pipeline_result",
    "training_result",
    "evaluation_result",
    "explainability_result",
    "report_result",
)

TRANSIENT_RESULT_KEYS: tuple[str, ...] = (
    "report_markdown_result",
    "shap_explanation_result",
)


def initialize_session_state(session_state: MutableMapping[str, Any]) -> None:
    """Initialize missing workflow keys without overwriting existing values.

    Args:
        session_state: A Streamlit-like session state mapping.
    """

    for key in EXPECTED_SESSION_KEYS:
        session_state.setdefault(key, None)


def reset_workflow_state(session_state: MutableMapping[str, Any]) -> None:
    """Reset all workflow keys to ``None``.

    Args:
        session_state: A Streamlit-like session state mapping.
    """

    for key in EXPECTED_SESSION_KEYS:
        session_state[key] = None


def invalidate_downstream_results(
    session_state: MutableMapping[str, Any],
    *,
    include_training: bool = True,
) -> None:
    """Clear results that depend on the selected or newly trained pipeline."""

    keys = DOWNSTREAM_RESULT_KEYS if include_training else DOWNSTREAM_RESULT_KEYS[2:]
    for key in keys:
        session_state[key] = None
    for key in TRANSIENT_RESULT_KEYS:
        session_state.pop(key, None)

def invalidate_explainability_dependents(session_state: MutableMapping[str, Any]) -> None:
    """Clear report and SHAP artifacts when explainability is regenerated."""

    session_state["report_result"] = None
    session_state.pop("report_markdown_result", None)
    session_state.pop("shap_explanation_result", None)

def has_dataset(session_state: MutableMapping[str, Any]) -> bool:
    """Return True when a dataset has been loaded."""

    return session_state.get("dataframe") is not None


def has_profile(session_state: MutableMapping[str, Any]) -> bool:
    """Return True when a dataset profile result exists."""

    return session_state.get("profile_result") is not None


def has_task_detection(session_state: MutableMapping[str, Any]) -> bool:
    """Return True when a task detection result exists."""

    return session_state.get("task_result") is not None


def has_recommendations(session_state: MutableMapping[str, Any]) -> bool:
    """Return True when pipeline recommendations exist."""

    return session_state.get("recommendation_result") is not None


def has_training_result(session_state: MutableMapping[str, Any]) -> bool:
    """Return True when a training result exists."""

    return session_state.get("training_result") is not None


def has_evaluation_result(session_state: MutableMapping[str, Any]) -> bool:
    """Return True when an evaluation result exists."""

    return session_state.get("evaluation_result") is not None