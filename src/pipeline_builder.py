"""Executable pipeline builder for the recommendation system.

This module converts an already decided pipeline recommendation into a concrete
scikit-learn or imbalanced-learn pipeline that can be fitted later by the model
trainer.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Any

from data_profiler import DatasetProfileResult
from recommendation_engine import PipelineRecommendation
from task_detector import TaskDetectionResult

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:  # Optional dependency.
    from xgboost import XGBClassifier, XGBRegressor  # type: ignore

    HAS_XGBOOST = True
except Exception:  # pragma: no cover - optional dependency fallback
    XGBClassifier = None  # type: ignore[assignment]
    XGBRegressor = None  # type: ignore[assignment]
    HAS_XGBOOST = False

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.naive_bayes import GaussianNB

logger = logging.getLogger(__name__)

_NUMERIC_COLUMN_NAME = "numeric"
_CATEGORICAL_COLUMN_NAME = "categorical"
_BOOLEAN_COLUMN_NAME = "boolean"


@dataclass(slots=True)
class BuiltPipelineResult:
    """Structured result of building an executable pipeline."""

    success: bool
    pipeline: object | None
    pipeline_name: str | None
    task_type: str | None
    model_name: str | None
    preprocessing_steps: list[str]
    messages: list[str]
    warnings: list[str]
    errors: list[str]


class PipelineBuilderError(Exception):
    """Base class for pipeline builder errors."""


class InvalidPipelineRecommendationError(PipelineBuilderError):
    """Raised when the recommendation or upstream results are invalid."""


class UnsupportedModelError(PipelineBuilderError):
    """Raised when the requested model family is not supported."""


class PipelineConstructionError(PipelineBuilderError):
    """Raised when the executable pipeline cannot be assembled."""


class PipelineBuilder:
    """Build executable pipelines from recommendation objects."""

    def build(
        self,
        recommendation: PipelineRecommendation,
        profile_result: DatasetProfileResult,
        task_result: TaskDetectionResult,
    ) -> BuiltPipelineResult:
        """Build an executable pipeline from a recommendation.

        Args:
            recommendation: The selected pipeline recommendation.
            profile_result: Dataset profile output.
            task_result: Task detection output.

        Returns:
            A BuiltPipelineResult containing the executable pipeline or errors.
        """

        try:
            _validate_inputs(recommendation, profile_result, task_result)
            logger.info(
                "Building pipeline: recommendation=%s task=%s model=%s",
                recommendation.pipeline_name,
                recommendation.task_type,
                recommendation.recommended_model_family,
            )

            feature_columns = _select_feature_columns(profile_result, task_result)
            if not any(feature_columns.values()):
                raise PipelineConstructionError("No usable feature columns were found in the profile result.")

            model = _build_model(recommendation, task_result)
            preprocessor = _build_preprocessor(recommendation, feature_columns, task_result)
            use_smote = _should_use_smote(recommendation, task_result)
            include_pca = _should_include_pca(recommendation)

            steps: list[tuple[str, Any]] = [("preprocessor", preprocessor)]
            if include_pca:
                steps.append(("pca", PCA(n_components=0.95)))
            if use_smote:
                steps.append(("smote", SMOTE(random_state=42)))
            steps.append(("model", model))

            pipeline = ImbPipeline(steps) if use_smote else SklearnPipeline(steps)
            messages = [f"Built pipeline '{recommendation.pipeline_name}' successfully."]
            if include_pca:
                messages.append("Included PCA as requested by the recommendation.")
            if use_smote:
                messages.append("Included SMOTE for classification imbalance handling.")

            return BuiltPipelineResult(
                success=True,
                pipeline=pipeline,
                pipeline_name=recommendation.pipeline_name,
                task_type=recommendation.task_type,
                model_name=model.__class__.__name__,
                preprocessing_steps=list(recommendation.preprocessing_steps),
                messages=messages,
                warnings=[],
                errors=[],
            )
        except PipelineBuilderError as exc:
            logger.error("Pipeline build failed: %s", exc)
            return _build_failure_result(recommendation, exc)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.exception("Unexpected pipeline build failure")
            return _failure_result(recommendation, PipelineConstructionError(str(exc)))


def build_pipeline(
    recommendation: PipelineRecommendation,
    profile_result: DatasetProfileResult,
    task_result: TaskDetectionResult,
) -> BuiltPipelineResult:
    """Build an executable pipeline using the default builder."""

    return PipelineBuilder().build(recommendation, profile_result, task_result)


def _validate_inputs(
    recommendation: PipelineRecommendation,
    profile_result: DatasetProfileResult,
    task_result: TaskDetectionResult,
) -> None:
    """Validate the builder inputs."""

    if not isinstance(recommendation, PipelineRecommendation):
        raise InvalidPipelineRecommendationError("recommendation must be a PipelineRecommendation instance.")
    if not isinstance(profile_result, DatasetProfileResult):
        raise InvalidPipelineRecommendationError("profile_result must be a DatasetProfileResult instance.")
    if not isinstance(task_result, TaskDetectionResult):
        raise InvalidPipelineRecommendationError("task_result must be a TaskDetectionResult instance.")
    if not recommendation.pipeline_name:
        raise InvalidPipelineRecommendationError("recommendation.pipeline_name is required.")
    if recommendation.task_type not in {"classification", "regression"}:
        raise InvalidPipelineRecommendationError("recommendation.task_type must be classification or regression.")
    if not profile_result.success:
        raise InvalidPipelineRecommendationError("profile_result must be successful before building a pipeline.")
    if not task_result.success:
        raise InvalidPipelineRecommendationError("task_result must be successful before building a pipeline.")
    if task_result.task_type != recommendation.task_type:
        raise InvalidPipelineRecommendationError("recommendation.task_type and task_result.task_type must match.")
    if profile_result.target_column is not None and profile_result.target_column != task_result.target_column:
        raise InvalidPipelineRecommendationError("profile_result and task_result must reference the same target column.")


def _select_feature_columns(profile_result: DatasetProfileResult, task_result: TaskDetectionResult) -> dict[str, list[str]]:
    """Select feature columns, excluding the target column when present."""

    target_column = task_result.target_column
    numeric_columns = _unique_preserve_order(
        column for column in profile_result.numeric_columns if column != target_column
    )
    categorical_columns = _unique_preserve_order(
        column for column in profile_result.categorical_columns if column != target_column
    )
    boolean_columns = _unique_preserve_order(
        column for column in profile_result.boolean_columns if column != target_column
    )
    return {
        _NUMERIC_COLUMN_NAME: numeric_columns,
        _CATEGORICAL_COLUMN_NAME: categorical_columns,
        _BOOLEAN_COLUMN_NAME: boolean_columns,
    }


def _build_preprocessor(
    recommendation: PipelineRecommendation,
    feature_columns: dict[str, list[str]],
    task_result: TaskDetectionResult,
) -> ColumnTransformer:
    """Build a ColumnTransformer for numeric, categorical, and boolean features."""

    transformers: list[tuple[str, Any, list[str]]] = []

    numeric_columns = feature_columns[_NUMERIC_COLUMN_NAME]
    if numeric_columns:
        numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
        scaler = _build_scaler(recommendation.scaling_strategy)
        if scaler is not None:
            numeric_steps.append(("scaler", scaler))
        transformers.append(("numeric", SklearnPipeline(numeric_steps), numeric_columns))

    categorical_columns = feature_columns[_CATEGORICAL_COLUMN_NAME]
    if categorical_columns:
        categorical_steps = [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _build_one_hot_encoder()),
        ]
        transformers.append(("categorical", SklearnPipeline(categorical_steps), categorical_columns))

    boolean_columns = feature_columns[_BOOLEAN_COLUMN_NAME]
    if boolean_columns:
        boolean_steps = [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("to_int", FunctionTransformer(lambda values: values.astype(int), validate=False)),
        ]
        transformers.append(("boolean", SklearnPipeline(boolean_steps), boolean_columns))

    if not transformers:
        raise PipelineConstructionError("No feature transformers could be created from the provided profile result.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _build_scaler(scaling_strategy: str) -> Any | None:
    """Map the recommendation's scaling strategy to a scaler instance."""

    normalized = _normalize_text(scaling_strategy)
    if not normalized or any(keyword in normalized for keyword in ["none", "noscaling", "optional", "notrequired"]):
        return None
    if "minmax" in normalized:
        return MinMaxScaler()
    if "robust" in normalized:
        return RobustScaler()
    if "standard" in normalized or "scale" in normalized:
        return StandardScaler()
    raise InvalidPipelineRecommendationError(f"Unsupported scaling strategy: {scaling_strategy}")


def _build_one_hot_encoder() -> OneHotEncoder:
    """Create a OneHotEncoder compatible with the installed scikit-learn version."""

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - older scikit-learn fallback
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _should_include_pca(recommendation: PipelineRecommendation) -> bool:
    """Return True when PCA should be added to the pipeline."""

    strategy = _normalize_text(recommendation.dimensionality_reduction_strategy or "")
    return "pca" in strategy


def _should_use_smote(recommendation: PipelineRecommendation, task_result: TaskDetectionResult) -> bool:
    """Return True when SMOTE should be inserted into the pipeline."""

    strategy = _normalize_text(recommendation.imbalance_handling_strategy or "")
    use_smote = "smote" in strategy
    if use_smote and task_result.task_type != "classification":
        raise InvalidPipelineRecommendationError("SMOTE can only be used for classification pipelines.")
    return use_smote


def _build_model(
    recommendation: PipelineRecommendation,
    task_result: TaskDetectionResult,
) -> Any:
    """Map the recommended model family to a concrete estimator."""

    model_key = _normalize_text(recommendation.recommended_model_family)
    task_type = task_result.task_type

    classification_models: dict[str, Any] = {
        "logisticregression": lambda: LogisticRegression(max_iter=1000),
        "randomforest": lambda: RandomForestClassifier(random_state=42),
        "decisiontree": lambda: DecisionTreeClassifier(random_state=42),
        "svc": lambda: SVC(),
        "svm": lambda: SVC(),
        "kneighborsclassifier": lambda: KNeighborsClassifier(),
        "knn": lambda: KNeighborsClassifier(),
        "gaussiannb": lambda: GaussianNB(),
        "xgbclassifier": lambda: _build_xgb_classifier(),
        "xgb": lambda: _build_xgb_classifier(),
    }

    regression_models: dict[str, Any] = {
        "linearregression": lambda: LinearRegression(),
        "ridge": lambda: Ridge(),
        "lasso": lambda: Lasso(max_iter=10000),
        "randomforestregressor": lambda: RandomForestRegressor(random_state=42),
        "decisiontreeregressor": lambda: DecisionTreeRegressor(random_state=42),
        "svr": lambda: SVR(),
        "xgbregressor": lambda: _build_xgb_regressor(),
        "xgb": lambda: _build_xgb_regressor(),
    }

    if task_type == "classification":
        factory = classification_models.get(model_key)
    else:
        factory = regression_models.get(model_key)

    if factory is None:
        raise UnsupportedModelError(f"Unsupported model family: {recommendation.recommended_model_family}")

    model = factory()
    if model is None:
        raise UnsupportedModelError(f"Unsupported model family: {recommendation.recommended_model_family}")
    return model


def _build_xgb_classifier() -> Any:
    """Build an XGBClassifier when xgboost is installed."""

    if not HAS_XGBOOST or XGBClassifier is None:
        raise UnsupportedModelError("xgboost is not installed, so XGBClassifier cannot be used.")
    return XGBClassifier(random_state=42, eval_metric="logloss")


def _build_xgb_regressor() -> Any:
    """Build an XGBRegressor when xgboost is installed."""

    if not HAS_XGBOOST or XGBRegressor is None:
        raise UnsupportedModelError("xgboost is not installed, so XGBRegressor cannot be used.")
    return XGBRegressor(random_state=42)


def _build_failure_result(
    recommendation: PipelineRecommendation | Any,
    exc: Exception,
) -> BuiltPipelineResult:
    """Build a failed pipeline result from an exception."""

    pipeline_name = recommendation.pipeline_name if isinstance(recommendation, PipelineRecommendation) else None
    task_type = recommendation.task_type if isinstance(recommendation, PipelineRecommendation) else None
    return BuiltPipelineResult(
        success=False,
        pipeline=None,
        pipeline_name=pipeline_name,
        task_type=task_type,
        model_name=None,
        preprocessing_steps=list(recommendation.preprocessing_steps)
        if isinstance(recommendation, PipelineRecommendation)
        else [],
        messages=[str(exc)],
        warnings=[],
        errors=[str(exc)],
    )


def _unique_preserve_order(values: Any) -> list[str]:
    """Return unique string values while preserving order."""

    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _normalize_text(value: str) -> str:
    """Normalize text for model and strategy matching."""

    return re.sub(r"[^a-z0-9]+", "", value.lower())