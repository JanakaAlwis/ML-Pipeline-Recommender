"""Knowledge base loader and validator for the recommendation system.

This module loads declarative JSON data that describes recommendation
knowledge. It does not implement recommendation logic; it only validates and
returns structured dataclasses for later use by the recommendation engine.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

METADATA_FILE = "metadata.json"
CATEGORIES_FILE = "categories.json"
PIPELINE_ARCHETYPES_FILE = "pipeline_archetypes.json"
RULE_FILES = {
    "imputation": "imputation_rules.json",
    "encoding": "encoding_rules.json",
    "scaling": "scaling_rules.json",
    "imbalance_handling": "imbalance_rules.json",
    "dimensionality_reduction": "dimensionality_rules.json",
    "model_family_classification": "classification_model_rules.json",
    "model_family_regression": "regression_model_rules.json",
    "evaluation_metrics_classification": "evaluation_metric_rules.json",
    "evaluation_metrics_regression": "evaluation_metric_rules.json",
    "cross_validation": "cross_validation_rules.json",
}
PROFILE_VERSION = "1.0"


@dataclass(slots=True)
class KnowledgeBaseMetadata:
    """Metadata describing the knowledge base."""

    version: str
    project_title: str
    description: str
    author: str
    created_date: str
    knowledge_base_version: str
    engine_scope: str
    notes: str


@dataclass(slots=True)
class KnowledgeBaseCategory:
    """A category of knowledge base rules."""

    category_id: str
    name: str
    description: str
    task_scope: list[str]


@dataclass(slots=True)
class KnowledgeBaseCondition:
    """One rule condition in the knowledge base."""

    field: str
    operator: str
    value: Any
    description: str | None


@dataclass(slots=True)
class KnowledgeBaseRule:
    """A declarative recommendation rule."""

    rule_id: str
    category: str
    name: str
    applies_to_task: list[str]
    conditions: list[KnowledgeBaseCondition]
    evidence_fields: list[str]
    recommendation: str
    reason_template: str
    pipeline_tags: list[str]
    confidence_weight: float
    severity: str
    priority: int
    advantages: list[str]
    limitations: list[str]
    notes: str | None


@dataclass(slots=True)
class PipelineArchetype:
    """A complete pipeline archetype that can be recommended."""

    archetype_id: str
    name: str
    task_type: str
    goal: str
    preferred_preprocessing: list[str]
    preferred_model_families: list[str]
    typical_metrics: list[str]
    when_to_use: list[str]
    when_not_to_use: list[str]
    advantages: list[str]
    limitations: list[str]
    complexity_label: str


@dataclass(slots=True)
class KnowledgeBaseData:
    """Loaded knowledge base content."""

    metadata: KnowledgeBaseMetadata
    categories: list[KnowledgeBaseCategory]
    pipeline_archetypes: list[PipelineArchetype]
    rules_by_category: dict[str, list[KnowledgeBaseRule]]


class KnowledgeBaseError(Exception):
    """Base class for knowledge base errors."""


class KnowledgeBaseNotLoadedError(KnowledgeBaseError):
    """Raised when data is requested before load() runs."""


class KnowledgeBaseValidationError(KnowledgeBaseError):
    """Raised when the JSON structure is invalid."""


class KnowledgeBaseFileError(KnowledgeBaseError):
    """Raised when a knowledge base file cannot be read."""


class KnowledgeBase:
    """Load and validate the declarative recommendation knowledge base."""

    def __init__(self, base_path: str | Path | None = None) -> None:
        """Initialize the knowledge base loader.

        Args:
            base_path: Optional path to the knowledge_base directory.
        """

        default_base_path = Path(__file__).resolve().parents[1] / "knowledge_base"
        self.base_path = Path(base_path) if base_path is not None else default_base_path
        self._data: KnowledgeBaseData | None = None

    def load(self) -> KnowledgeBaseData:
        """Load, validate, and cache the knowledge base.

        Returns:
            Loaded knowledge base data as structured dataclasses.

        Raises:
            KnowledgeBaseFileError: If a required file is missing or unreadable.
            KnowledgeBaseValidationError: If the JSON structure is invalid.
        """

        logger.info("Loading knowledge base from %s", self.base_path)
        self._validate_base_directory()

        metadata = self._load_metadata()
        categories = self._load_categories()
        pipeline_archetypes = self._load_pipeline_archetypes()
        rules_by_category = self._load_rules_by_category()

        self._data = KnowledgeBaseData(
            metadata=metadata,
            categories=categories,
            pipeline_archetypes=pipeline_archetypes,
            rules_by_category=rules_by_category,
        )
        logger.info(
            "Knowledge base loaded: categories=%s, archetypes=%s, rule_groups=%s",
            len(categories),
            len(pipeline_archetypes),
            len(rules_by_category),
        )
        return self._data

    def get_rules(self, category: str | None = None) -> list[KnowledgeBaseRule]:
        """Return rules, optionally filtered by category.

        Args:
            category: Optional category name.

        Returns:
            A list of knowledge base rules.

        Raises:
            KnowledgeBaseNotLoadedError: If load() has not run.
            KnowledgeBaseValidationError: If the category is unknown.
        """

        data = self._require_loaded()
        if category is None:
            ordered_rules: list[KnowledgeBaseRule] = []
            for rule_list in data.rules_by_category.values():
                ordered_rules.extend(rule_list)
            return ordered_rules

        if category not in data.rules_by_category:
            valid_categories = ", ".join(sorted(data.rules_by_category))
            raise KnowledgeBaseValidationError(
                f"Unknown rule category: {category}. Available categories: {valid_categories}"
            )
        return list(data.rules_by_category[category])

    def get_pipeline_archetypes(self) -> list[PipelineArchetype]:
        """Return the loaded pipeline archetypes."""

        data = self._require_loaded()
        return list(data.pipeline_archetypes)

    def get_metadata(self) -> KnowledgeBaseMetadata:
        """Return the loaded knowledge base metadata."""

        data = self._require_loaded()
        return data.metadata

    def _require_loaded(self) -> KnowledgeBaseData:
        """Return loaded data or raise a not-loaded error."""

        if self._data is None:
            raise KnowledgeBaseNotLoadedError("Knowledge base has not been loaded yet.")
        return self._data

    def _validate_base_directory(self) -> None:
        """Validate that the base knowledge base directory and files exist."""

        if not self.base_path.exists() or not self.base_path.is_dir():
            raise KnowledgeBaseFileError(f"Knowledge base directory not found: {self.base_path}")

        required_files = [METADATA_FILE, CATEGORIES_FILE, PIPELINE_ARCHETYPES_FILE]
        for file_name in required_files:
            file_path = self.base_path / file_name
            if not file_path.exists():
                raise KnowledgeBaseFileError(f"Missing knowledge base file: {file_name}")

        rules_directory = self.base_path / "rules"
        if not rules_directory.exists() or not rules_directory.is_dir():
            raise KnowledgeBaseFileError(f"Missing knowledge base rules directory: {rules_directory}")

        for category, file_name in RULE_FILES.items():
            file_path = rules_directory / file_name
            if not file_path.exists():
                raise KnowledgeBaseFileError(f"Missing rule file for category '{category}': {file_name}")

    def _read_json(self, file_path: Path) -> Any:
        """Read JSON from a file path."""

        try:
            with file_path.open("r", encoding="utf-8") as file_handle:
                return json.load(file_handle)
        except OSError as exc:
            raise KnowledgeBaseFileError(f"Could not read file: {file_path}") from exc
        except json.JSONDecodeError as exc:
            raise KnowledgeBaseValidationError(f"Invalid JSON in file: {file_path.name}") from exc

    def _load_metadata(self) -> KnowledgeBaseMetadata:
        """Load and validate metadata.json."""

        metadata_path = self.base_path / METADATA_FILE
        data = self._read_json(metadata_path)
        required_fields = {
            "version": str,
            "project_title": str,
            "description": str,
            "author": str,
            "created_date": str,
            "knowledge_base_version": str,
            "engine_scope": str,
            "notes": str,
        }
        self._validate_dict_schema(data, required_fields, metadata_path.name)
        return KnowledgeBaseMetadata(**data)

    def _load_categories(self) -> list[KnowledgeBaseCategory]:
        """Load and validate categories.json."""

        categories_path = self.base_path / CATEGORIES_FILE
        data = self._read_json(categories_path)
        if not isinstance(data, list) or not data:
            raise KnowledgeBaseValidationError(f"{categories_path.name} must contain a non-empty list.")

        categories: list[KnowledgeBaseCategory] = []
        for index, item in enumerate(data):
            context = f"{categories_path.name}[{index}]"
            required_fields = {
                "category_id": str,
                "name": str,
                "description": str,
                "task_scope": list,
            }
            self._validate_dict_schema(item, required_fields, context)
            self._validate_list_of_strings(item["task_scope"], f"{context}.task_scope")
            categories.append(KnowledgeBaseCategory(**item))
        return categories

    def _load_pipeline_archetypes(self) -> list[PipelineArchetype]:
        """Load and validate pipeline_archetypes.json."""

        archetypes_path = self.base_path / PIPELINE_ARCHETYPES_FILE
        data = self._read_json(archetypes_path)
        if not isinstance(data, list) or not data:
            raise KnowledgeBaseValidationError(
                f"{archetypes_path.name} must contain a non-empty list."
            )

        archetypes: list[PipelineArchetype] = []
        for index, item in enumerate(data):
            context = f"{archetypes_path.name}[{index}]"
            required_fields = {
                "archetype_id": str,
                "name": str,
                "task_type": str,
                "goal": str,
                "preferred_preprocessing": list,
                "preferred_model_families": list,
                "typical_metrics": list,
                "when_to_use": list,
                "when_not_to_use": list,
                "advantages": list,
                "limitations": list,
                "complexity_label": str,
            }
            self._validate_dict_schema(item, required_fields, context)
            for key in [
                "preferred_preprocessing",
                "preferred_model_families",
                "typical_metrics",
                "when_to_use",
                "when_not_to_use",
                "advantages",
                "limitations",
            ]:
                self._validate_list_of_strings(item[key], f"{context}.{key}")
            archetypes.append(PipelineArchetype(**item))
        return archetypes

    def _load_rules_by_category(self) -> dict[str, list[KnowledgeBaseRule]]:
        """Load and validate all rule files."""

        rules_directory = self.base_path / "rules"
        rules_by_category: dict[str, list[KnowledgeBaseRule]] = {}
        seen_rule_ids: set[str] = set()
        loaded_files: set[str] = set()

        for category, file_name in RULE_FILES.items():
            if file_name in loaded_files:
                continue
            loaded_files.add(file_name)

            file_path = rules_directory / file_name
            data = self._read_json(file_path)
            if not isinstance(data, list) or not data:
                raise KnowledgeBaseValidationError(f"{file_path.name} must contain a non-empty list.")

            for index, item in enumerate(data):
                context = f"{file_path.name}[{index}]"
                rule = self._parse_rule(item, context)
                if rule.rule_id in seen_rule_ids:
                    raise KnowledgeBaseValidationError(
                        f"Duplicate rule_id detected: {rule.rule_id} in {file_path.name}."
                    )
                seen_rule_ids.add(rule.rule_id)
                rules_by_category.setdefault(rule.category, []).append(rule)

        if "dimensionality_reduction" not in rules_by_category:
            raise KnowledgeBaseValidationError(
                "dimensionality_reduction rules are required but were not loaded."
            )
        if "feature_selection" not in rules_by_category:
            raise KnowledgeBaseValidationError(
                "feature_selection rules are required but were not loaded."
            )

        return rules_by_category

    def _parse_rule(
        self,
        item: Any,
        context: str,
    ) -> KnowledgeBaseRule:
        """Parse and validate a rule entry."""

        required_fields = {
            "rule_id": str,
            "category": str,
            "name": str,
            "applies_to_task": list,
            "conditions": list,
            "evidence_fields": list,
            "recommendation": str,
            "reason_template": str,
            "pipeline_tags": list,
            "confidence_weight": (int, float),
            "severity": str,
            "priority": int,
            "advantages": list,
            "limitations": list,
            "notes": (str, type(None)),
        }
        self._validate_dict_schema(item, required_fields, context)
        self._validate_list_of_strings(item["applies_to_task"], f"{context}.applies_to_task")
        self._validate_list_of_strings(item["evidence_fields"], f"{context}.evidence_fields")
        self._validate_list_of_strings(item["pipeline_tags"], f"{context}.pipeline_tags")
        self._validate_list_of_strings(item["advantages"], f"{context}.advantages")
        self._validate_list_of_strings(item["limitations"], f"{context}.limitations")
        if not isinstance(item["conditions"], list) or not item["conditions"]:
            raise KnowledgeBaseValidationError(f"{context}.conditions must be a non-empty list.")

        conditions = [self._parse_condition(condition, f"{context}.conditions[{index}]") for index, condition in enumerate(item["conditions"])]

        confidence_weight = float(item["confidence_weight"])
        if not 0.0 <= confidence_weight <= 1.0:
            raise KnowledgeBaseValidationError(
                f"{context}.confidence_weight must be between 0 and 1."
            )
        if item["priority"] < 1:
            raise KnowledgeBaseValidationError(f"{context}.priority must be at least 1.")
        if item["severity"] not in {"high", "medium", "low"}:
            raise KnowledgeBaseValidationError(
                f"{context}.severity must be one of: high, medium, low."
            )

        return KnowledgeBaseRule(
            rule_id=item["rule_id"],
            category=item["category"],
            name=item["name"],
            applies_to_task=list(item["applies_to_task"]),
            conditions=conditions,
            evidence_fields=list(item["evidence_fields"]),
            recommendation=item["recommendation"],
            reason_template=item["reason_template"],
            pipeline_tags=list(item["pipeline_tags"]),
            confidence_weight=confidence_weight,
            severity=item["severity"],
            priority=item["priority"],
            advantages=list(item["advantages"]),
            limitations=list(item["limitations"]),
            notes=item["notes"],
        )

    def _parse_condition(self, item: Any, context: str) -> KnowledgeBaseCondition:
        """Parse and validate one condition entry."""

        required_fields = {
            "field": str,
            "operator": str,
            "value": object,
            "description": (str, type(None)),
        }
        self._validate_dict_schema(item, required_fields, context)
        return KnowledgeBaseCondition(
            field=item["field"],
            operator=item["operator"],
            value=item["value"],
            description=item.get("description"),
        )

    def _validate_dict_schema(
        self,
        item: Any,
        required_fields: dict[str, type | tuple[type, ...] | object],
        context: str,
    ) -> None:
        """Validate that an item is a dictionary with expected fields."""

        if not isinstance(item, dict):
            raise KnowledgeBaseValidationError(f"{context} must be a JSON object.")

        for field_name, expected_type in required_fields.items():
            if field_name not in item:
                raise KnowledgeBaseValidationError(f"{context} is missing required field '{field_name}'.")
            if expected_type is object:
                continue
            if not isinstance(item[field_name], expected_type):
                raise KnowledgeBaseValidationError(
                    f"{context}.{field_name} must be of type {self._describe_type(expected_type)}."
                )

    def _validate_list_of_strings(self, value: Any, context: str) -> None:
        """Validate that a value is a list of strings."""

        if not isinstance(value, list) or not value:
            raise KnowledgeBaseValidationError(f"{context} must be a non-empty list.")
        for index, item in enumerate(value):
            if not isinstance(item, str):
                raise KnowledgeBaseValidationError(f"{context}[{index}] must be a string.")

    def _describe_type(self, expected_type: type | tuple[type, ...]) -> str:
        """Return a human-readable type description."""

        if isinstance(expected_type, tuple):
            return ", ".join(type_item.__name__ for type_item in expected_type)
        return expected_type.__name__


def _default_knowledge_base() -> KnowledgeBase:
    """Create a knowledge base instance using the default repository path."""

    return KnowledgeBase()