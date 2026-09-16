"""Tests for presentation-only recommendation explanations."""

from __future__ import annotations

from dataclasses import replace

import pytest

from app.views.recommendations import _why_recommended
from src.recommendation_engine import PipelineRecommendation


def _pipeline(model_family: str, **overrides: object) -> PipelineRecommendation:
    """Build a recommendation suitable for display formatter tests."""

    values: dict[str, object] = {
        "pipeline_rank": 1,
        "pipeline_name": "Test Pipeline",
        "task_type": "classification",
        "preprocessing_steps": ["Impute missing values", "Standard scaling"],
        "encoding_strategy": "One-hot encoding for categorical features",
        "scaling_strategy": "Standard scaling for numeric features",
        "missing_value_strategy": "Median imputation for numeric features",
        "imbalance_handling_strategy": None,
        "dimensionality_reduction_strategy": None,
        "recommended_model_family": model_family,
        "evaluation_metrics": ["Accuracy"],
        "cross_validation_recommendation": "Use cross-validation",
        "confidence_score": 0.8,
        "priority": 1,
        "explanation": "Original stored explanation must remain unchanged.",
        "evidence": ["test evidence"],
        "advantages": ["Aggregated advantage"],
        "limitations": ["Aggregated limitation"],
        "estimated_training_complexity": "low",
        "supporting_recommendation_ids": [],
    }
    values.update(overrides)
    return PipelineRecommendation(**values)


@pytest.mark.parametrize(
    ("model_family", "expected_text"),
    [
        ("logistic_regression", "interpretable linear baseline"),
        ("random_forest", "flexible ensemble"),
        ("knn", "distance-based alternative"),
        ("gaussiannb", "fast probabilistic baseline"),
        ("linear_regression", "clear linear baseline"),
        ("ridge", "regularized linear model"),
        ("svr", "margin-based model"),
        ("random_forest_regressor", "flexible ensemble"),
    ],
)
def test_formatter_is_specific_to_reachable_model_families(model_family: str, expected_text: str) -> None:
    """Reachable model families should produce distinct useful rationale text."""

    pipeline = _pipeline(model_family)

    explanation = _why_recommended(pipeline)

    assert expected_text in explanation
    assert "Aggregated advantage" not in explanation
    assert "Aggregated limitation" not in explanation


def test_formatter_includes_strategy_fields_without_modifying_stored_explanation() -> None:
    """Display formatting should use strategies and preserve backend state."""

    pipeline = _pipeline(
        "random_forest",
        task_type="classification",
        missing_value_strategy="No special imputation required",
        scaling_strategy="Scaling is optional",
        imbalance_handling_strategy="SMOTE inside training folds",
        dimensionality_reduction_strategy="PCA or feature selection",
    )
    original = replace(pipeline)

    explanation = _why_recommended(pipeline)

    assert "Scaling is optional" in explanation
    assert "SMOTE inside training folds" in explanation
    assert "PCA or feature selection" in explanation
    assert pipeline == original
    assert pipeline.explanation == "Original stored explanation must remain unchanged."


def test_formatter_has_safe_fallback_for_empty_optional_fields() -> None:
    """Missing display fields should still produce readable text."""

    pipeline = _pipeline(
        "",
        task_type="",
        missing_value_strategy=None,
        scaling_strategy=None,
        imbalance_handling_strategy=None,
        dimensionality_reduction_strategy=None,
    )

    explanation = _why_recommended(pipeline)

    assert explanation == "Recommended as the selected model for the detected; it provides the selected model."