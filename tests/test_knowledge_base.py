"""Tests for the Knowledge Base loader and validator."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import pytest

from knowledge_base import (
    KnowledgeBase,
    KnowledgeBaseCategory,
    KnowledgeBaseCondition,
    KnowledgeBaseData,
    KnowledgeBaseFileError,
    KnowledgeBaseMetadata,
    KnowledgeBaseNotLoadedError,
    KnowledgeBaseRule,
    KnowledgeBaseValidationError,
    PipelineArchetype,
)


def _write_json(file_path: Path, payload: object) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_valid_knowledge_base(base_path: Path) -> None:
    _write_json(
        base_path / "metadata.json",
        {
            "version": "1.0",
            "project_title": "Explainable ML Pipeline Recommendation System",
            "description": "Test knowledge base.",
            "author": "Tester",
            "created_date": "2026-06-25",
            "knowledge_base_version": "1.0",
            "engine_scope": "Recommendation knowledge only",
            "notes": "Test data.",
        },
    )
    _write_json(
        base_path / "categories.json",
        [
            {
                "category_id": "imputation",
                "name": "Imputation",
                "description": "Missing value handling.",
                "task_scope": ["classification", "regression"],
            },
            {
                "category_id": "encoding",
                "name": "Encoding",
                "description": "Categorical feature handling.",
                "task_scope": ["classification", "regression"],
            },
            {
                "category_id": "scaling",
                "name": "Scaling",
                "description": "Feature scaling.",
                "task_scope": ["classification", "regression"],
            },
            {
                "category_id": "imbalance_handling",
                "name": "Imbalance Handling",
                "description": "Class imbalance strategies.",
                "task_scope": ["classification"],
            },
            {
                "category_id": "dimensionality_reduction",
                "name": "Dimensionality Reduction",
                "description": "Dimensionality reduction.",
                "task_scope": ["classification", "regression"],
            },
            {
                "category_id": "feature_selection",
                "name": "Feature Selection",
                "description": "Feature selection.",
                "task_scope": ["classification", "regression"],
            },
            {
                "category_id": "model_family_classification",
                "name": "Classification Model Families",
                "description": "Classification models.",
                "task_scope": ["classification"],
            },
            {
                "category_id": "model_family_regression",
                "name": "Regression Model Families",
                "description": "Regression models.",
                "task_scope": ["regression"],
            },
            {
                "category_id": "evaluation_metrics_classification",
                "name": "Classification Metrics",
                "description": "Metrics for classification.",
                "task_scope": ["classification"],
            },
            {
                "category_id": "evaluation_metrics_regression",
                "name": "Regression Metrics",
                "description": "Metrics for regression.",
                "task_scope": ["regression"],
            },
            {
                "category_id": "cross_validation",
                "name": "Cross Validation",
                "description": "Cross validation and baseline comparison.",
                "task_scope": ["classification", "regression"],
            },
        ],
    )
    _write_json(
        base_path / "pipeline_archetypes.json",
        [
            {
                "archetype_id": "balanced_linear_classification",
                "name": "Balanced Linear Classification Pipeline",
                "task_type": "classification",
                "goal": "Baseline classification.",
                "preferred_preprocessing": ["imputation", "one_hot_encoding", "scaling"],
                "preferred_model_families": ["logistic_regression"],
                "typical_metrics": ["accuracy", "f1_score"],
                "when_to_use": ["Balanced classes"],
                "when_not_to_use": ["Severe imbalance"],
                "advantages": ["Simple"],
                "limitations": ["May underfit"],
                "complexity_label": "low",
            }
        ],
    )

    rule_payloads = {
        "imputation_rules.json": [
            {
                "rule_id": "IMP-A-001",
                "category": "imputation",
                "name": "Impute missing values when any missingness exists",
                "applies_to_task": ["classification", "regression"],
                "conditions": [
                    {
                        "field": "missing_values.total_count",
                        "operator": ">",
                        "value": 0,
                        "description": "Missing values exist.",
                    }
                ],
                "evidence_fields": ["missing_values"],
                "recommendation": "Use imputation.",
                "reason_template": "Missing values require imputation.",
                "pipeline_tags": ["imputation"],
                "confidence_weight": 0.95,
                "severity": "high",
                "priority": 1,
                "advantages": ["Preserves rows"],
                "limitations": ["May introduce bias"],
                "notes": None,
            }
        ],
        "encoding_rules.json": [
            {
                "rule_id": "ENC-A-001",
                "category": "encoding",
                "name": "Use one-hot encoding for categorical features",
                "applies_to_task": ["classification", "regression"],
                "conditions": [
                    {
                        "field": "categorical_columns.count",
                        "operator": ">",
                        "value": 0,
                        "description": "Categorical features exist.",
                    }
                ],
                "evidence_fields": ["categorical_columns"],
                "recommendation": "Use one-hot encoding.",
                "reason_template": "Categorical features require numeric representation.",
                "pipeline_tags": ["encoding"],
                "confidence_weight": 0.92,
                "severity": "high",
                "priority": 1,
                "advantages": ["Simple"],
                "limitations": ["Can expand feature space"],
                "notes": None,
            }
        ],
        "scaling_rules.json": [
            {
                "rule_id": "SCL-A-001",
                "category": "scaling",
                "name": "Scale features for scale-sensitive models",
                "applies_to_task": ["classification", "regression"],
                "conditions": [
                    {
                        "field": "numeric_columns.count",
                        "operator": ">",
                        "value": 0,
                        "description": "Numeric features exist.",
                    }
                ],
                "evidence_fields": ["numeric_columns"],
                "recommendation": "Use standard scaling.",
                "reason_template": "Numeric features benefit from scaling for some models.",
                "pipeline_tags": ["scaling"],
                "confidence_weight": 0.88,
                "severity": "medium",
                "priority": 2,
                "advantages": ["Helps linear models"],
                "limitations": ["May be optional for tree models"],
                "notes": None,
            }
        ],
        "imbalance_rules.json": [
            {
                "rule_id": "IMB-C-001",
                "category": "imbalance_handling",
                "name": "Use class weights for moderate imbalance",
                "applies_to_task": ["classification"],
                "conditions": [
                    {
                        "field": "class_imbalance_ratio",
                        "operator": "<=",
                        "value": 0.4,
                        "description": "Imbalance exists.",
                    }
                ],
                "evidence_fields": ["class_imbalance_ratio"],
                "recommendation": "Use class weights.",
                "reason_template": "Imbalance can be handled with class weights.",
                "pipeline_tags": ["imbalance_handling"],
                "confidence_weight": 0.86,
                "severity": "high",
                "priority": 1,
                "advantages": ["Simple"],
                "limitations": ["May not solve severe imbalance"],
                "notes": None,
            }
        ],
        "dimensionality_rules.json": [
            {
                "rule_id": "DIM-A-001",
                "category": "dimensionality_reduction",
                "name": "Recommend PCA for correlated features",
                "applies_to_task": ["classification", "regression"],
                "conditions": [
                    {
                        "field": "high_correlation_pairs.count",
                        "operator": ">",
                        "value": 0,
                        "description": "Correlated features exist.",
                    }
                ],
                "evidence_fields": ["high_correlation_pairs"],
                "recommendation": "Consider PCA.",
                "reason_template": "PCA reduces redundancy.",
                "pipeline_tags": ["pca"],
                "confidence_weight": 0.84,
                "severity": "medium",
                "priority": 2,
                "advantages": ["Reduces redundancy"],
                "limitations": ["Reduces interpretability"],
                "notes": None,
            },
            {
                "rule_id": "DIM-A-002",
                "category": "feature_selection",
                "name": "Recommend feature selection for many numeric columns",
                "applies_to_task": ["classification", "regression"],
                "conditions": [
                    {
                        "field": "numeric_columns.count",
                        "operator": ">=",
                        "value": 10,
                        "description": "Many numeric features exist.",
                    }
                ],
                "evidence_fields": ["numeric_columns"],
                "recommendation": "Consider feature selection.",
                "reason_template": "Feature selection can simplify a wide numeric feature set.",
                "pipeline_tags": ["feature_selection"],
                "confidence_weight": 0.7,
                "severity": "low",
                "priority": 3,
                "advantages": ["Simplifies the feature set"],
                "limitations": ["Does not name a specific feature selection method"],
                "notes": None,
            }
        ],
        "classification_model_rules.json": [
            {
                "rule_id": "MOD-C-001",
                "category": "model_family_classification",
                "name": "Recommend logistic regression",
                "applies_to_task": ["classification"],
                "conditions": [
                    {
                        "field": "task_type",
                        "operator": "=",
                        "value": "classification",
                        "description": "Classification task.",
                    }
                ],
                "evidence_fields": ["task_type"],
                "recommendation": "Use logistic regression.",
                "reason_template": "Logistic regression is a simple classification baseline.",
                "pipeline_tags": ["logistic_regression"],
                "confidence_weight": 0.9,
                "severity": "high",
                "priority": 1,
                "advantages": ["Simple"],
                "limitations": ["May underfit nonlinear problems"],
                "notes": None,
            }
        ],
        "regression_model_rules.json": [
            {
                "rule_id": "MOD-R-001",
                "category": "model_family_regression",
                "name": "Recommend linear regression",
                "applies_to_task": ["regression"],
                "conditions": [
                    {
                        "field": "task_type",
                        "operator": "=",
                        "value": "regression",
                        "description": "Regression task.",
                    }
                ],
                "evidence_fields": ["task_type"],
                "recommendation": "Use linear regression.",
                "reason_template": "Linear regression is a simple regression baseline.",
                "pipeline_tags": ["linear_regression"],
                "confidence_weight": 0.9,
                "severity": "high",
                "priority": 1,
                "advantages": ["Simple"],
                "limitations": ["May underfit nonlinear problems"],
                "notes": None,
            }
        ],
        "evaluation_metric_rules.json": [
            {
                "rule_id": "MET-C-001",
                "category": "evaluation_metrics_classification",
                "name": "Use classification metrics",
                "applies_to_task": ["classification"],
                "conditions": [
                    {
                        "field": "task_type",
                        "operator": "=",
                        "value": "classification",
                        "description": "Classification task.",
                    }
                ],
                "evidence_fields": ["task_type"],
                "recommendation": "Use accuracy, precision, recall, F1-score, and confusion matrix.",
                "reason_template": "Classification needs class-aware metrics.",
                "pipeline_tags": ["classification_metrics"],
                "confidence_weight": 0.95,
                "severity": "high",
                "priority": 1,
                "advantages": ["Clear evaluation"],
                "limitations": ["Accuracy alone may mislead"],
                "notes": None,
            }
        ],
        "cross_validation_rules.json": [
            {
                "rule_id": "CV-A-001",
                "category": "cross_validation",
                "name": "Recommend cross-validation",
                "applies_to_task": ["classification", "regression"],
                "conditions": [
                    {
                        "field": "row_count",
                        "operator": ">",
                        "value": 50,
                        "description": "Enough rows for validation.",
                    }
                ],
                "evidence_fields": ["row_count"],
                "recommendation": "Use cross-validation.",
                "reason_template": "Cross-validation provides a more stable evaluation.",
                "pipeline_tags": ["cross_validation"],
                "confidence_weight": 0.85,
                "severity": "medium",
                "priority": 2,
                "advantages": ["More stable evaluation"],
                "limitations": ["Takes longer"],
                "notes": None,
            }
        ],
    }

    for file_name, payload in rule_payloads.items():
        _write_json(base_path / "rules" / file_name, payload)


def test_knowledge_base_dataclasses_have_expected_fields() -> None:
    """The loader should expose the agreed dataclasses."""

    assert [field.name for field in fields(KnowledgeBaseMetadata)] == [
        "version",
        "project_title",
        "description",
        "author",
        "created_date",
        "knowledge_base_version",
        "engine_scope",
        "notes",
    ]
    assert [field.name for field in fields(KnowledgeBaseCategory)] == [
        "category_id",
        "name",
        "description",
        "task_scope",
    ]
    assert [field.name for field in fields(KnowledgeBaseCondition)] == [
        "field",
        "operator",
        "value",
        "description",
    ]
    assert [field.name for field in fields(KnowledgeBaseRule)] == [
        "rule_id",
        "category",
        "name",
        "applies_to_task",
        "conditions",
        "evidence_fields",
        "recommendation",
        "reason_template",
        "pipeline_tags",
        "confidence_weight",
        "severity",
        "priority",
        "advantages",
        "limitations",
        "notes",
    ]
    assert [field.name for field in fields(PipelineArchetype)] == [
        "archetype_id",
        "name",
        "task_type",
        "goal",
        "preferred_preprocessing",
        "preferred_model_families",
        "typical_metrics",
        "when_to_use",
        "when_not_to_use",
        "advantages",
        "limitations",
        "complexity_label",
    ]
    assert [field.name for field in fields(KnowledgeBaseData)] == [
        "metadata",
        "categories",
        "pipeline_archetypes",
        "rules_by_category",
    ]


def test_knowledge_base_loads_successfully(tmp_path: Path) -> None:
    """A valid knowledge base should load successfully."""

    base_path = tmp_path / "knowledge_base"
    _write_valid_knowledge_base(base_path)

    knowledge_base = KnowledgeBase(base_path)
    loaded = knowledge_base.load()

    assert isinstance(loaded, KnowledgeBaseData)
    assert isinstance(knowledge_base.get_metadata(), KnowledgeBaseMetadata)
    assert len(knowledge_base.get_pipeline_archetypes()) == 1
    assert len(knowledge_base.get_rules()) == 11
    assert len(knowledge_base.get_rules("imputation")) == 1


def test_knowledge_base_returns_structured_objects(tmp_path: Path) -> None:
    """Loaded content should be returned as structured dataclasses."""

    base_path = tmp_path / "knowledge_base"
    _write_valid_knowledge_base(base_path)

    knowledge_base = KnowledgeBase(base_path)
    knowledge_base.load()

    assert isinstance(knowledge_base.get_metadata(), KnowledgeBaseMetadata)
    assert all(isinstance(item, PipelineArchetype) for item in knowledge_base.get_pipeline_archetypes())
    assert all(isinstance(item, KnowledgeBaseRule) for item in knowledge_base.get_rules())


def test_get_rules_before_load_raises_error(tmp_path: Path) -> None:
    """Accessing rules before load() should raise a not-loaded error."""

    base_path = tmp_path / "knowledge_base"
    _write_valid_knowledge_base(base_path)

    knowledge_base = KnowledgeBase(base_path)

    with pytest.raises(KnowledgeBaseNotLoadedError):
        knowledge_base.get_rules()


def test_get_rules_unknown_category_raises_error(tmp_path: Path) -> None:
    """Unknown categories should raise a validation error."""

    base_path = tmp_path / "knowledge_base"
    _write_valid_knowledge_base(base_path)

    knowledge_base = KnowledgeBase(base_path)
    knowledge_base.load()

    with pytest.raises(KnowledgeBaseValidationError):
        knowledge_base.get_rules("unknown")


def test_load_raises_for_missing_rule_file(tmp_path: Path) -> None:
    """Missing rule files should raise a file error."""

    base_path = tmp_path / "knowledge_base"
    _write_valid_knowledge_base(base_path)
    (base_path / "rules" / "scaling_rules.json").unlink()

    knowledge_base = KnowledgeBase(base_path)

    with pytest.raises(KnowledgeBaseFileError):
        knowledge_base.load()


def test_load_raises_for_invalid_metadata_schema(tmp_path: Path) -> None:
    """Metadata missing required fields should fail validation."""

    base_path = tmp_path / "knowledge_base"
    _write_valid_knowledge_base(base_path)
    _write_json(base_path / "metadata.json", {"version": "1.0"})

    knowledge_base = KnowledgeBase(base_path)

    with pytest.raises(KnowledgeBaseValidationError):
        knowledge_base.load()


def test_load_raises_for_invalid_rule_schema(tmp_path: Path) -> None:
    """A rule with invalid schema should fail validation."""

    base_path = tmp_path / "knowledge_base"
    _write_valid_knowledge_base(base_path)
    _write_json(
        base_path / "rules" / "encoding_rules.json",
        [
            {
                "rule_id": "IMP-A-999",
                "category": "encoding",
                "name": "Invalid confidence",
                "applies_to_task": ["classification"],
                "conditions": [
                    {
                        "field": "missing_values.total_count",
                        "operator": ">",
                        "value": 0,
                        "description": "Mismatch.",
                    }
                ],
                "evidence_fields": ["missing_values"],
                "recommendation": "Wrong.",
                "reason_template": "Wrong.",
                "pipeline_tags": ["wrong"],
                "confidence_weight": 1.5,
                "severity": "high",
                "priority": 1,
                "advantages": ["A"],
                "limitations": ["B"],
                "notes": None,
            }
        ],
    )

    knowledge_base = KnowledgeBase(base_path)

    with pytest.raises(KnowledgeBaseValidationError):
        knowledge_base.load()
