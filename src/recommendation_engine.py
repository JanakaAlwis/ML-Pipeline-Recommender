"""Explainable ML pipeline recommendation engine.

This module turns dataset profiling evidence and task detection output into
ranked, explainable pipeline recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from statistics import mean
from typing import Any

from src.data_profiler import DatasetProfileResult
from src.knowledge_base import KnowledgeBase, KnowledgeBaseError, KnowledgeBaseRule, PipelineArchetype
from src.task_detector import TaskDetectionResult

logger = logging.getLogger(__name__)

ENGINE_VERSION = "1.0"
TOP_PIPELINE_COUNT = 3
SUPPORTED_TASK_TYPES = {"classification", "regression"}


@dataclass(slots=True)
class SupportingRecommendationItem:
    """A rule-backed supporting recommendation item."""

    recommendation_id: str
    category: str
    name: str
    recommendation: str
    reason: str
    confidence_score: float
    severity: str
    priority: int
    evidence: list[str]
    applies_to: str | None
    advantages: list[str]
    limitations: list[str]


@dataclass(slots=True)
class PipelineRecommendation:
    """A complete explainable ML pipeline recommendation."""

    pipeline_rank: int
    pipeline_name: str
    task_type: str
    preprocessing_steps: list[str]
    encoding_strategy: str
    scaling_strategy: str
    missing_value_strategy: str
    imbalance_handling_strategy: str | None
    dimensionality_reduction_strategy: str | None
    recommended_model_family: str
    evaluation_metrics: list[str]
    cross_validation_recommendation: str
    confidence_score: float
    priority: int
    explanation: str
    evidence: list[str]
    advantages: list[str]
    limitations: list[str]
    estimated_training_complexity: str
    supporting_recommendation_ids: list[str]


@dataclass(slots=True)
class PipelineRecommendationResult:
    """Structured result of the recommendation engine."""

    success: bool
    task_type: str | None
    messages: list[str]
    warnings: list[str]
    errors: list[str]
    top_pipelines: list[PipelineRecommendation]
    supporting_recommendations: list[SupportingRecommendationItem]
    profile_summary: dict[str, str | int | float | None]
    task_summary: dict[str, str | int | float | None]
    engine_version: str
    knowledge_base_version: str


class RecommendationEngineError(Exception):
    """Base class for recommendation engine errors."""


class InvalidRecommendationInputError(RecommendationEngineError):
    """Raised when the upstream inputs are invalid."""


class RecommendationComputationError(RecommendationEngineError):
    """Raised when recommendation generation fails."""


class RecommendationEngine:
    """Generate explainable pipeline recommendations from upstream evidence."""

    def __init__(self, knowledge_base: KnowledgeBase | None = None) -> None:
        self._knowledge_base = knowledge_base or KnowledgeBase()
        self._cached_data = None

    def recommend(
        self,
        profile_result: DatasetProfileResult,
        task_result: TaskDetectionResult,
    ) -> PipelineRecommendationResult:
        """Generate ranked pipeline recommendations."""

        try:
            validate_recommendation_input(profile_result, task_result)
            knowledge_base_data = self._load_knowledge_base()
            task_type = task_result.task_type
            if task_type is None:
                raise RecommendationComputationError("Task type is missing.")

            profile_context = _build_profile_context(profile_result)
            task_context = _build_task_context(task_result)
            combined_context = {**profile_context, **task_context}

            supporting_items: dict[str, SupportingRecommendationItem] = {}
            pipelines: list[PipelineRecommendation] = []

            archetypes = [
                archetype
                for archetype in knowledge_base_data.pipeline_archetypes
                if archetype.task_type == task_type
            ]
            if not archetypes:
                raise RecommendationComputationError(f"No pipeline archetypes available for task type '{task_type}'.")

            for archetype in archetypes:
                model_family = _select_model_family(archetype, profile_result, task_result)
                pipeline_context = {**combined_context, "preferred_model_family": model_family}
                matched_items = _match_rules(
                    knowledge_base_data.rules_by_category,
                    pipeline_context,
                    task_type,
                    archetype.name,
                )
                pipelines.append(
                    _build_pipeline_recommendation(
                        archetype=archetype,
                        task_result=task_result,
                        profile_result=profile_result,
                        model_family=model_family,
                        matched_items=matched_items,
                        pipeline_context=pipeline_context,
                    )
                )
                for item in matched_items:
                    supporting_items.setdefault(item.recommendation_id, item)

            pipelines = _rank_pipelines(pipelines)[:TOP_PIPELINE_COUNT]
            for rank, pipeline in enumerate(pipelines, start=1):
                pipeline.pipeline_rank = rank
                pipeline.priority = rank

            support_list = sorted(
                supporting_items.values(),
                key=lambda item: (item.priority, -item.confidence_score, item.recommendation_id),
            )

            return PipelineRecommendationResult(
                success=True,
                task_type=task_type,
                messages=[f"Generated {len(pipelines)} pipeline recommendations."],
                warnings=list(dict.fromkeys(profile_result.warnings + task_result.warnings)),
                errors=[],
                top_pipelines=pipelines,
                supporting_recommendations=support_list,
                profile_summary=_build_profile_summary(profile_result),
                task_summary=_build_task_summary(task_result),
                engine_version=ENGINE_VERSION,
                knowledge_base_version=knowledge_base_data.metadata.knowledge_base_version,
            )
        except (InvalidRecommendationInputError, RecommendationComputationError, KnowledgeBaseError) as exc:
            logger.error("Recommendation engine failed: %s", exc)
            return PipelineRecommendationResult(
                success=False,
                task_type=task_result.task_type if isinstance(task_result, TaskDetectionResult) else None,
                messages=[str(exc)],
                warnings=[],
                errors=[str(exc)],
                top_pipelines=[],
                supporting_recommendations=[],
                profile_summary=_build_profile_summary(profile_result)
                if isinstance(profile_result, DatasetProfileResult)
                else {},
                task_summary=_build_task_summary(task_result)
                if isinstance(task_result, TaskDetectionResult)
                else {},
                engine_version=ENGINE_VERSION,
                knowledge_base_version="unknown",
            )

    def _load_knowledge_base(self):
        """Load and cache the knowledge base."""

        if self._cached_data is None:
            self._cached_data = self._knowledge_base.load()
        return self._cached_data


def recommend_pipelines(
    profile_result: DatasetProfileResult,
    task_result: TaskDetectionResult,
) -> PipelineRecommendationResult:
    """Convenience wrapper around RecommendationEngine.recommend()."""

    return RecommendationEngine().recommend(profile_result, task_result)


def validate_recommendation_input(
    profile_result: DatasetProfileResult,
    task_result: TaskDetectionResult,
) -> None:
    """Validate upstream inputs before recommendation generation."""

    if not isinstance(profile_result, DatasetProfileResult):
        raise InvalidRecommendationInputError("Profile result must be a DatasetProfileResult instance.")
    if not isinstance(task_result, TaskDetectionResult):
        raise InvalidRecommendationInputError("Task result must be a TaskDetectionResult instance.")
    if not profile_result.success:
        raise InvalidRecommendationInputError("Profile result is not successful.")
    if not task_result.success:
        raise InvalidRecommendationInputError("Task result is not successful.")
    if task_result.task_type not in SUPPORTED_TASK_TYPES:
        raise InvalidRecommendationInputError(f"Unsupported task type: {task_result.task_type}")
    if profile_result.row_count is None or profile_result.column_count is None:
        raise InvalidRecommendationInputError("Profile result is missing shape information.")
    if profile_result.target_column is not None and profile_result.target_column != task_result.target_column:
        raise InvalidRecommendationInputError("Profile and task target columns do not match.")


def _build_pipeline_recommendation(
    *,
    archetype: PipelineArchetype,
    task_result: TaskDetectionResult,
    profile_result: DatasetProfileResult,
    model_family: str,
    matched_items: list[SupportingRecommendationItem],
    pipeline_context: dict[str, Any],
) -> PipelineRecommendation:
    """Build a single pipeline recommendation."""

    support_ids = [item.recommendation_id for item in matched_items]
    support_scores = [item.confidence_score for item in matched_items]
    support_average = mean(support_scores) if support_scores else 0.5
    fit_score = _score_archetype_fit(archetype, profile_result, task_result, model_family)
    confidence = _clamp(0.4 * (task_result.confidence or 0.5) + 0.35 * support_average + 0.25 * fit_score)

    preprocessing_steps = _build_preprocessing_steps(profile_result, task_result, model_family, archetype)
    explanation = _build_explanation(archetype, task_result, profile_result, matched_items, model_family)
    evidence = _build_evidence(profile_result, task_result, matched_items)
    advantages = _merge_unique(archetype.advantages, [item.advantages[0] for item in matched_items if item.advantages])
    limitations = _merge_unique(archetype.limitations, [item.limitations[0] for item in matched_items if item.limitations])

    return PipelineRecommendation(
        pipeline_rank=0,
        pipeline_name=archetype.name,
        task_type=task_result.task_type or archetype.task_type,
        preprocessing_steps=preprocessing_steps,
        encoding_strategy=_encoding_strategy(profile_result),
        scaling_strategy=_scaling_strategy(model_family, archetype),
        missing_value_strategy=_missing_value_strategy(profile_result),
        imbalance_handling_strategy=_imbalance_strategy(profile_result, task_result),
        dimensionality_reduction_strategy=_dimensionality_strategy(archetype, profile_result, model_family),
        recommended_model_family=model_family,
        evaluation_metrics=_evaluation_metrics(task_result),
        cross_validation_recommendation=_cross_validation_recommendation(profile_result),
        confidence_score=confidence,
        priority=0,
        explanation=explanation,
        evidence=evidence,
        advantages=advantages,
        limitations=limitations,
        estimated_training_complexity=archetype.complexity_label,
        supporting_recommendation_ids=support_ids,
    )


def _match_rules(
    rules_by_category: dict[str, list[KnowledgeBaseRule]],
    context: dict[str, Any],
    task_type: str,
    pipeline_name: str,
) -> list[SupportingRecommendationItem]:
    """Return supporting recommendations for the current context."""

    matched: list[SupportingRecommendationItem] = []
    for rule_list in rules_by_category.values():
        for rule in rule_list:
            if task_type not in rule.applies_to_task:
                continue
            if not _rule_matches(rule, context):
                continue
            matched.append(_build_supporting_item(rule, context, pipeline_name))
    return matched


def _build_supporting_item(
    rule: KnowledgeBaseRule,
    context: dict[str, Any],
    pipeline_name: str,
) -> SupportingRecommendationItem:
    """Convert a matched rule into a supporting recommendation item."""

    return SupportingRecommendationItem(
        recommendation_id=rule.rule_id,
        category=rule.category,
        name=rule.name,
        recommendation=rule.recommendation,
        reason=rule.reason_template,
        confidence_score=_score_rule_confidence(rule, context),
        severity=rule.severity,
        priority=rule.priority,
        evidence=_rule_evidence(rule, context),
        applies_to=pipeline_name,
        advantages=list(rule.advantages),
        limitations=list(rule.limitations),
    )


def _rule_matches(rule: KnowledgeBaseRule, context: dict[str, Any]) -> bool:
    """Return True when all rule conditions are satisfied."""

    return all(_condition_matches(condition, context) for condition in rule.conditions)


def _condition_matches(condition: Any, context: dict[str, Any]) -> bool:
    """Evaluate a single knowledge base condition."""

    actual_value = _lookup(context, condition.field)
    if actual_value is None:
        return False

    operator = condition.operator.lower()
    expected_value = condition.value

    if operator in {"=", "=="}:
        return actual_value == expected_value
    if operator == ">":
        return _numeric_compare(actual_value, expected_value, lambda left, right: left > right)
    if operator == ">=":
        return _numeric_compare(actual_value, expected_value, lambda left, right: left >= right)
    if operator == "<":
        return _numeric_compare(actual_value, expected_value, lambda left, right: left < right)
    if operator == "<=":
        return _numeric_compare(actual_value, expected_value, lambda left, right: left <= right)
    if operator == "in":
        return actual_value in expected_value
    return False


def _score_rule_confidence(rule: KnowledgeBaseRule, context: dict[str, Any]) -> float:
    """Score how strongly a rule matches the evidence."""

    if not rule.conditions:
        return _clamp(rule.confidence_weight)
    strengths = [_condition_strength(condition, context) for condition in rule.conditions]
    return _clamp(rule.confidence_weight * mean(strengths))


def _condition_strength(condition: Any, context: dict[str, Any]) -> float:
    """Estimate how well one condition is satisfied."""

    actual_value = _lookup(context, condition.field)
    if actual_value is None:
        return 0.0

    operator = condition.operator.lower()
    expected_value = condition.value
    if operator in {"=", "=="}:
        return 1.0 if actual_value == expected_value else 0.0
    if operator == "in":
        return 1.0 if actual_value in expected_value else 0.0

    try:
        actual_number = float(actual_value)
        expected_number = float(expected_value)
    except (TypeError, ValueError):
        return 0.0

    if operator in {">", ">="}:
        return 1.0 if actual_number > expected_number else 0.0
    if operator in {"<", "<="}:
        return 1.0 if actual_number < expected_number else 0.0
    return 0.0


def _rule_evidence(rule: KnowledgeBaseRule, context: dict[str, Any]) -> list[str]:
    """Build evidence strings for a matched rule."""

    evidence: list[str] = []
    for field_name in rule.evidence_fields:
        value = _lookup(context, field_name)
        if value is None:
            continue
        evidence.append(f"{field_name}={value}")
    return evidence


def _build_profile_context(profile_result: DatasetProfileResult) -> dict[str, Any]:
    """Build a flat evidence context from profile output."""

    missing_total = int(sum(profile_result.missing_values.values()))
    numeric_missing = int(sum(profile_result.missing_values.get(column, 0) for column in profile_result.numeric_columns))
    categorical_missing = int(sum(profile_result.missing_values.get(column, 0) for column in profile_result.categorical_columns))
    return {
        "row_count": profile_result.row_count or 0,
        "column_count": profile_result.column_count or 0,
        "numeric_columns.count": len(profile_result.numeric_columns),
        "categorical_columns.count": len(profile_result.categorical_columns),
        "high_cardinality_columns.count": len(profile_result.high_cardinality_columns),
        "high_correlation_pairs.count": len(profile_result.high_correlation_pairs),
        "missing_values.total_count": missing_total,
        "numeric_columns.missing_count": numeric_missing,
        "categorical_columns.missing_count": categorical_missing,
        "class_imbalance_ratio": profile_result.class_imbalance_ratio,
        "target_column": profile_result.target_column,
        "target_distribution": profile_result.target_distribution,
    }


def _build_task_context(task_result: TaskDetectionResult) -> dict[str, Any]:
    """Build a flat evidence context from task detection output."""

    return {
        "task_type": task_result.task_type,
        "task_confidence": task_result.confidence or 0.0,
        "target_column": task_result.target_column,
        "target_dtype": task_result.target_dtype,
        "target_unique_count": task_result.target_unique_count or 0,
        "target_unique_ratio": task_result.target_unique_ratio or 0.0,
        "is_numeric_target": task_result.is_numeric_target,
        "is_categorical_target": task_result.is_categorical_target,
        "is_boolean_target": task_result.is_boolean_target,
        "class_count": task_result.class_count,
    }


def _build_profile_summary(profile_result: DatasetProfileResult) -> dict[str, str | int | float | None]:
    """Create a compact profile summary for the output payload."""

    return {
        "row_count": profile_result.row_count,
        "column_count": profile_result.column_count,
        "numeric_columns": len(profile_result.numeric_columns),
        "categorical_columns": len(profile_result.categorical_columns),
        "missing_values_total": int(sum(profile_result.missing_values.values())),
        "high_correlation_pairs": len(profile_result.high_correlation_pairs),
        "high_cardinality_columns": len(profile_result.high_cardinality_columns),
        "class_imbalance_ratio": profile_result.class_imbalance_ratio,
    }


def _build_task_summary(task_result: TaskDetectionResult) -> dict[str, str | int | float | None]:
    """Create a compact task summary for the output payload."""

    return {
        "task_type": task_result.task_type,
        "confidence": task_result.confidence,
        "target_unique_count": task_result.target_unique_count,
        "target_unique_ratio": task_result.target_unique_ratio,
        "class_count": task_result.class_count,
    }


def _build_preprocessing_steps(
    profile_result: DatasetProfileResult,
    task_result: TaskDetectionResult,
    model_family: str,
    archetype: PipelineArchetype,
) -> list[str]:
    """Build a readable preprocessing sequence."""

    steps = ["Impute missing values"] if sum(profile_result.missing_values.values()) > 0 else ["No special imputation required"]
    if profile_result.categorical_columns:
        steps.append(_encoding_strategy(profile_result))
    scaling = _scaling_strategy(model_family, archetype)
    if scaling:
        steps.append(scaling)
    imbalance = _imbalance_strategy(profile_result, task_result)
    if imbalance:
        steps.append(imbalance)
    dimensionality = _dimensionality_strategy(archetype, profile_result, model_family)
    if dimensionality:
        steps.append(dimensionality)
    steps.append(f"Train {model_family}")
    return steps


def _encoding_strategy(profile_result: DatasetProfileResult) -> str:
    """Describe encoding for categorical features."""

    if profile_result.high_cardinality_columns:
        return "Careful encoding for high-cardinality categorical features"
    return "One-hot encoding for categorical features"


def _scaling_strategy(model_family: str, archetype: PipelineArchetype) -> str:
    """Describe scaling for the selected model family."""

    if model_family.lower().startswith(("decision tree", "random forest")):
        return "Scaling is optional"
    if archetype.archetype_id in {"robust_tree_classification", "robust_tree_regression"}:
        return "Scaling is optional"
    return "Standard scaling for numeric features"


def _missing_value_strategy(profile_result: DatasetProfileResult) -> str:
    """Describe the missing-value strategy for the dataset."""

    numeric_missing = int(sum(profile_result.missing_values.get(column, 0) for column in profile_result.numeric_columns))
    categorical_missing = int(sum(profile_result.missing_values.get(column, 0) for column in profile_result.categorical_columns))
    if numeric_missing and categorical_missing:
        return "Median imputation for numeric features and most-frequent imputation for categorical features"
    if numeric_missing:
        return "Median imputation for numeric features"
    if categorical_missing:
        return "Most-frequent imputation for categorical features"
    return "No special imputation required"


def _imbalance_strategy(profile_result: DatasetProfileResult, task_result: TaskDetectionResult) -> str | None:
    """Describe imbalance handling for classification tasks."""

    if task_result.task_type != "classification" or profile_result.class_imbalance_ratio is None:
        return None
    if profile_result.class_imbalance_ratio < 0.2:
        return "SMOTE inside training folds"
    if profile_result.class_imbalance_ratio < 0.4:
        return "Class weights or threshold tuning"
    return None


def _dimensionality_strategy(
    archetype: PipelineArchetype,
    profile_result: DatasetProfileResult,
    model_family: str,
) -> str | None:
    """Describe dimensionality reduction or feature selection."""

    high_corr = len(profile_result.high_correlation_pairs)
    numeric_count = len(profile_result.numeric_columns)
    if high_corr == 0 and numeric_count < 10:
        return None
    if archetype.archetype_id == "regularized_regression":
        return "Feature selection or regularization-aware dimensionality reduction"
    if "tree" in archetype.archetype_id:
        return "Optional feature selection"
    return "PCA or feature selection"


def _evaluation_metrics(task_result: TaskDetectionResult) -> list[str]:
    """Return task-appropriate evaluation metrics."""

    if task_result.task_type == "classification":
        metrics = ["Accuracy", "Precision", "Recall", "F1-score", "Confusion matrix"]
        if task_result.target_unique_count == 2:
            metrics.append("ROC-AUC")
        return metrics
    return ["MAE", "MSE", "RMSE", "R2"]


def _cross_validation_recommendation(profile_result: DatasetProfileResult) -> str:
    """Return a simple cross-validation recommendation."""

    if (profile_result.row_count or 0) >= 50:
        return "Use 5-fold cross-validation and compare with a baseline pipeline."
    return "Use a train/validation split and compare with a baseline pipeline."


def _select_model_family(
    archetype: PipelineArchetype,
    profile_result: DatasetProfileResult,
    task_result: TaskDetectionResult,
) -> str:
    """Select a representative model family for the archetype."""

    if task_result.task_type == "classification":
        if archetype.archetype_id == "balanced_linear_classification":
            return "logistic_regression"
        if archetype.archetype_id == "robust_tree_classification":
            return "random_forest"
        if archetype.archetype_id == "imbalanced_classification":
            if profile_result.class_imbalance_ratio is not None and profile_result.class_imbalance_ratio < 0.2:
                return "random_forest"
            return "logistic_regression"
    if task_result.task_type == "regression":
        if archetype.archetype_id == "balanced_linear_regression":
            return "linear_regression"
        if archetype.archetype_id == "robust_tree_regression":
            return "random_forest_regressor"
        if archetype.archetype_id == "regularized_regression":
            return "ridge"
    return archetype.preferred_model_families[0]


def _score_archetype_fit(
    archetype: PipelineArchetype,
    profile_result: DatasetProfileResult,
    task_result: TaskDetectionResult,
    model_family: str,
) -> float:
    """Estimate how well an archetype fits the current dataset."""

    numeric_count = len(profile_result.numeric_columns)
    categorical_count = len(profile_result.categorical_columns)
    missing_total = int(sum(profile_result.missing_values.values()))
    high_corr = len(profile_result.high_correlation_pairs)
    imbalance_ratio = profile_result.class_imbalance_ratio

    score = 0.6
    if archetype.task_type == task_result.task_type:
        score += 0.1
    if missing_total > 0:
        score += 0.05
    if categorical_count > 0:
        score += 0.05
    if numeric_count > 0:
        score += 0.05

    if archetype.archetype_id == "imbalanced_classification":
        if imbalance_ratio is not None and imbalance_ratio < 0.2:
            score += 0.2
    if archetype.archetype_id == "regularized_regression" and high_corr > 0:
        score += 0.2
    if model_family in {"logistic_regression", "linear_regression", "ridge"}:
        score += 0.05
    if model_family in {"random_forest", "random_forest_regressor"}:
        score += 0.03
    return _clamp(score)


def _build_explanation(
    archetype: PipelineArchetype,
    task_result: TaskDetectionResult,
    profile_result: DatasetProfileResult,
    matched_items: list[SupportingRecommendationItem],
    model_family: str,
) -> str:
    """Create a readable explanation for a recommendation."""

    evidence_parts = [
        f"task={task_result.task_type}",
        f"rows={profile_result.row_count}",
        f"numeric_columns={len(profile_result.numeric_columns)}",
        f"categorical_columns={len(profile_result.categorical_columns)}",
        f"missing_values={sum(profile_result.missing_values.values())}",
    ]
    if profile_result.class_imbalance_ratio is not None:
        evidence_parts.append(f"class_imbalance_ratio={profile_result.class_imbalance_ratio:.2f}")
    if matched_items:
        evidence_parts.append(f"matched_rules={len(matched_items)}")
    return (
        f"{archetype.name} fits because {', '.join(evidence_parts)}. "
        f"It uses {model_family} with a transparent preprocessing sequence."
    )


def _build_evidence(
    profile_result: DatasetProfileResult,
    task_result: TaskDetectionResult,
    matched_items: list[SupportingRecommendationItem],
) -> list[str]:
    """Create a readable evidence list for a recommendation."""

    evidence = [
        f"rows={profile_result.row_count}",
        f"columns={profile_result.column_count}",
        f"numeric_columns={len(profile_result.numeric_columns)}",
        f"categorical_columns={len(profile_result.categorical_columns)}",
        f"missing_values={sum(profile_result.missing_values.values())}",
        f"task_type={task_result.task_type}",
    ]
    if profile_result.class_imbalance_ratio is not None:
        evidence.append(f"class_imbalance_ratio={profile_result.class_imbalance_ratio:.2f}")
    for item in matched_items[:5]:
        evidence.append(item.reason)
    return list(dict.fromkeys(evidence))


def _rank_pipelines(pipelines: list[PipelineRecommendation]) -> list[PipelineRecommendation]:
    """Rank pipelines by confidence and deterministic tie-breakers."""

    return sorted(pipelines, key=lambda pipeline: (-pipeline.confidence_score, pipeline.pipeline_name))


def _build_profile_summary(profile_result: DatasetProfileResult) -> dict[str, str | int | float | None]:
    """Create a compact profile summary for the result payload."""

    return {
        "row_count": profile_result.row_count,
        "column_count": profile_result.column_count,
        "numeric_columns": len(profile_result.numeric_columns),
        "categorical_columns": len(profile_result.categorical_columns),
        "missing_values_total": int(sum(profile_result.missing_values.values())),
        "high_correlation_pairs": len(profile_result.high_correlation_pairs),
        "high_cardinality_columns": len(profile_result.high_cardinality_columns),
        "class_imbalance_ratio": profile_result.class_imbalance_ratio,
    }


def _build_task_summary(task_result: TaskDetectionResult) -> dict[str, str | int | float | None]:
    """Create a compact task summary for the result payload."""

    return {
        "task_type": task_result.task_type,
        "confidence": task_result.confidence,
        "target_unique_count": task_result.target_unique_count,
        "target_unique_ratio": task_result.target_unique_ratio,
        "class_count": task_result.class_count,
    }


def _lookup(context: dict[str, Any], field_path: str) -> Any:
    """Read a dotted field path from a nested evidence dictionary."""

    if field_path in context:
        return context[field_path]

    current: Any = context
    for part in field_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _numeric_compare(actual_value: Any, expected_value: Any, comparator) -> bool:
    """Safely compare numeric values."""

    try:
        return bool(comparator(float(actual_value), float(expected_value)))
    except (TypeError, ValueError):
        return False


def _merge_unique(primary: list[str], secondary: list[str]) -> list[str]:
    """Combine two lists while preserving order and removing duplicates."""

    merged: list[str] = []
    seen: set[str] = set()
    for value in [*primary, *secondary]:
        if value not in seen:
            merged.append(value)
            seen.add(value)
    return merged


def _clamp(value: float) -> float:
    """Clamp a numeric value to the 0..1 range."""

    return max(0.0, min(1.0, value))
