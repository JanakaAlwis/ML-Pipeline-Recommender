"""Tests for the application state helper module."""

from __future__ import annotations

from src.application_state import (
    EXPECTED_SESSION_KEYS,
    has_dataset,
    has_evaluation_result,
    has_profile,
    has_recommendations,
    has_task_detection,
    has_training_result,
    invalidate_downstream_results,
    invalidate_explainability_dependents,
    initialize_session_state,
    reset_workflow_state,
)


def test_expected_session_keys_exist() -> None:
    """The helper should define the expected workflow keys."""

    assert EXPECTED_SESSION_KEYS == (
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


def test_initialize_session_state_adds_missing_keys() -> None:
    """Initialization should add missing keys with None values."""

    session_state: dict[str, object] = {}

    initialize_session_state(session_state)

    for key in EXPECTED_SESSION_KEYS:
        assert key in session_state
        assert session_state[key] is None


def test_initialize_session_state_does_not_overwrite_existing_values() -> None:
    """Initialization should preserve values that already exist."""

    session_state: dict[str, object] = {"dataframe": "existing", "target_column": "target"}

    initialize_session_state(session_state)

    assert session_state["dataframe"] == "existing"
    assert session_state["target_column"] == "target"
    assert session_state["profile_result"] is None


def test_reset_workflow_state_clears_values() -> None:
    """Reset should clear all workflow values back to None."""

    session_state: dict[str, object] = {key: f"value-for-{key}" for key in EXPECTED_SESSION_KEYS}

    reset_workflow_state(session_state)

    for key in EXPECTED_SESSION_KEYS:
        assert session_state[key] is None


def test_helper_functions_return_expected_booleans() -> None:
    """Presence helpers should report workflow state correctly."""

    session_state: dict[str, object] = {key: None for key in EXPECTED_SESSION_KEYS}

    assert has_dataset(session_state) is False
    assert has_profile(session_state) is False
    assert has_task_detection(session_state) is False
    assert has_recommendations(session_state) is False
    assert has_training_result(session_state) is False
    assert has_evaluation_result(session_state) is False

    session_state["dataframe"] = object()
    session_state["profile_result"] = object()
    session_state["task_result"] = object()
    session_state["recommendation_result"] = object()
    session_state["training_result"] = object()
    session_state["evaluation_result"] = object()

    assert has_dataset(session_state) is True
    assert has_profile(session_state) is True
    assert has_task_detection(session_state) is True
    assert has_recommendations(session_state) is True
    assert has_training_result(session_state) is True
    assert has_evaluation_result(session_state) is True


def test_invalidate_downstream_results_preserves_upstream_state() -> None:
    """Invalidation should clear dependent artifacts without clearing workflow inputs."""

    session_state: dict[str, object] = {
        "dataframe": "data",
        "profile_result": "profile",
        "task_result": "task",
        "recommendation_result": "recommendations",
        "selected_pipeline": "selected",
        "built_pipeline_result": "built",
        "training_result": "training",
        "evaluation_result": "evaluation",
        "explainability_result": "explainability",
        "report_result": "report",
        "report_markdown_result": "markdown",
        "shap_explanation_result": "shap",
    }

    invalidate_downstream_results(session_state)

    assert session_state["dataframe"] == "data"
    assert session_state["profile_result"] == "profile"
    assert session_state["task_result"] == "task"
    assert session_state["recommendation_result"] == "recommendations"
    assert session_state["selected_pipeline"] == "selected"
    assert session_state["built_pipeline_result"] is None
    assert session_state["training_result"] is None
    assert session_state["evaluation_result"] is None
    assert session_state["explainability_result"] is None
    assert session_state["report_result"] is None
    assert "report_markdown_result" not in session_state
    assert "shap_explanation_result" not in session_state


def test_invalidate_downstream_results_without_training_preserves_training() -> None:
    """Evaluation/explainability regeneration should preserve the current training result."""

    session_state: dict[str, object] = {
        "training_result": "training",
        "evaluation_result": "evaluation",
        "explainability_result": "explainability",
        "report_result": "report",
        "report_markdown_result": "markdown",
        "shap_explanation_result": "shap",
    }

    invalidate_downstream_results(session_state, include_training=False)

    assert session_state["training_result"] == "training"
    assert session_state["evaluation_result"] is None
    assert session_state["explainability_result"] is None
    assert session_state["report_result"] is None
    assert "report_markdown_result" not in session_state
    assert "shap_explanation_result" not in session_state

def test_invalidate_explainability_dependents_preserves_evaluation() -> None:
    """Explainability regeneration should preserve evaluation and training results."""

    session_state: dict[str, object] = {
        "selected_pipeline": "selected",
        "built_pipeline_result": "built",
        "training_result": "training",
        "evaluation_result": "evaluation",
        "explainability_result": "old explainability",
        "report_result": "report",
        "report_markdown_result": "markdown",
        "shap_explanation_result": "old shap",
    }

    invalidate_explainability_dependents(session_state)

    assert session_state["selected_pipeline"] == "selected"
    assert session_state["built_pipeline_result"] == "built"
    assert session_state["training_result"] == "training"
    assert session_state["evaluation_result"] == "evaluation"
    assert session_state["explainability_result"] == "old explainability"
    assert session_state["report_result"] is None
    assert "report_markdown_result" not in session_state
    assert "shap_explanation_result" not in session_state