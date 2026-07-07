"""Model trainer and pipeline executor.

This module takes an already-built executable pipeline and fits it on a
tabular dataset. It does not recommend models or construct pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split

from src.data_profiler import DatasetProfileResult
from src.pipeline_builder import BuiltPipelineResult
from src.task_detector import TaskDetectionResult

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TrainingResult:
    """Structured result of a pipeline training run."""

    success: bool
    trained_pipeline: object | None
    model_name: str | None
    task_type: str | None
    target_column: str | None
    train_row_count: int | None
    test_row_count: int | None
    training_time_seconds: float | None
    cross_validation_scores: list[float]
    mean_cv_score: float | None
    std_cv_score: float | None
    scoring_metric: str | None
    messages: list[str]
    warnings: list[str]
    errors: list[str]


class ModelTrainerError(Exception):
    """Base class for model trainer errors."""


class InvalidTrainingInputError(ModelTrainerError):
    """Raised when the training inputs are invalid."""


class TrainingExecutionError(ModelTrainerError):
    """Raised when training or cross-validation fails."""


class ModelTrainer:
    """Fit an already-built pipeline on a dataset."""

    def train(
        self,
        built_pipeline_result: BuiltPipelineResult,
        dataframe: pd.DataFrame,
        target_column: str,
        task_type: str,
        run_cross_validation: bool = True,
    ) -> TrainingResult:
        """Train the pipeline and optionally run cross-validation.

        Args:
            built_pipeline_result: Executable pipeline produced by the builder.
            dataframe: Dataset containing features and target.
            target_column: Name of the target column.
            task_type: Supervised task type, classification or regression.
            run_cross_validation: Whether to run cross-validation.

        Returns:
            A structured TrainingResult.
        """

        try:
            _validate_inputs(built_pipeline_result, dataframe, target_column, task_type)

            pipeline = built_pipeline_result.pipeline
            if pipeline is None:
                raise InvalidTrainingInputError("built_pipeline_result.pipeline must not be None.")

            X = dataframe.drop(columns=[target_column])
            y = dataframe[target_column]

            warnings: list[str] = []
            missing_target_rows = int(y.isna().sum())
            if missing_target_rows > 0:
                valid_target_mask = y.notna()
                X = X.loc[valid_target_mask]
                y = y.loc[valid_target_mask]
                warnings.append(
                    f"Removed {missing_target_rows} row(s) because the target column contains missing values. "
                    "Only target-missing rows were removed; feature-missing rows are handled in preprocessing."
                )

            if len(y) < 2:
                raise InvalidTrainingInputError("Not enough rows remain after removing missing target values.")

            stratify = _build_stratify_target(task_type, y)
            train_test_kwargs: dict[str, Any] = {
                "test_size": 0.2,
                "random_state": 42,
            }
            if stratify is not None:
                train_test_kwargs["stratify"] = stratify

            X_train, X_test, y_train, y_test = train_test_split(X, y, **train_test_kwargs)
            train_row_count = int(len(X_train))
            test_row_count = int(len(X_test))

            scoring_metric = _get_scoring_metric(task_type)
            cross_validation_scores: list[float] = []

            if run_cross_validation:
                cv, cv_warning = _build_cross_validation(task_type, y_train)
                if cv is None:
                    warnings.append(cv_warning or "Cross-validation was skipped.")
                else:
                    try:
                        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring_metric)
                        cross_validation_scores = [float(score) for score in cv_scores]
                    except Exception as exc:
                        warnings.append(f"Cross-validation was skipped: {exc}")

            start_time = time.perf_counter()
            pipeline.fit(X_train, y_train)
            training_time_seconds = time.perf_counter() - start_time

            mean_cv_score = float(np.mean(cross_validation_scores)) if cross_validation_scores else None
            std_cv_score = float(np.std(cross_validation_scores)) if cross_validation_scores else None

            return TrainingResult(
                success=True,
                trained_pipeline=pipeline,
                model_name=pipeline.steps[-1][1].__class__.__name__ if getattr(pipeline, "steps", None) else None,
                task_type=task_type,
                target_column=target_column,
                train_row_count=train_row_count,
                test_row_count=test_row_count,
                training_time_seconds=training_time_seconds,
                cross_validation_scores=cross_validation_scores,
                mean_cv_score=mean_cv_score,
                std_cv_score=std_cv_score,
                scoring_metric=scoring_metric,
                messages=["Training completed successfully."],
                warnings=warnings,
                errors=[],
            )
        except ModelTrainerError as exc:
            logger.error("Training failed: %s", exc)
            return _failure_result(built_pipeline_result, target_column, task_type, exc)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.exception("Unexpected training failure")
            return _failure_result(
                built_pipeline_result,
                target_column,
                task_type,
                TrainingExecutionError(str(exc)),
            )


def train_pipeline(
    built_pipeline_result: BuiltPipelineResult,
    dataframe: pd.DataFrame,
    target_column: str,
    task_type: str,
    run_cross_validation: bool = True,
) -> TrainingResult:
    """Train a built pipeline using the default trainer instance."""

    return ModelTrainer().train(built_pipeline_result, dataframe, target_column, task_type, run_cross_validation)


def _validate_inputs(
    built_pipeline_result: BuiltPipelineResult,
    dataframe: pd.DataFrame,
    target_column: str,
    task_type: str,
) -> None:
    """Validate training inputs before fitting the pipeline."""

    if not isinstance(built_pipeline_result, BuiltPipelineResult):
        raise InvalidTrainingInputError("built_pipeline_result must be a BuiltPipelineResult instance.")
    if not built_pipeline_result.success:
        error_details = "; ".join(built_pipeline_result.errors) if built_pipeline_result.errors else "unknown error"
        raise InvalidTrainingInputError(f"built_pipeline_result.success must be True: {error_details}")
    if built_pipeline_result.pipeline is None:
        raise InvalidTrainingInputError("built_pipeline_result.pipeline must not be None.")
    if not isinstance(dataframe, pd.DataFrame):
        raise InvalidTrainingInputError("dataframe must be a pandas DataFrame.")
    if dataframe.empty:
        raise InvalidTrainingInputError("dataframe must not be empty.")
    if not isinstance(target_column, str) or not target_column.strip():
        raise InvalidTrainingInputError("target_column must be a non-empty string.")
    if target_column not in dataframe.columns:
        raise InvalidTrainingInputError(f"Target column not found: {target_column}")
    if task_type not in {"classification", "regression"}:
        raise InvalidTrainingInputError("task_type must be classification or regression.")


def _build_stratify_target(task_type: str, y: pd.Series) -> pd.Series | None:
    """Return a stratification target when it is safe to do so."""

    if task_type != "classification":
        return None
    class_counts = Counter(y.dropna())
    if len(class_counts) < 2:
        return None
    if min(class_counts.values()) < 2:
        return None
    return y


def _build_cross_validation(task_type: str, y_train: pd.Series) -> tuple[Any | None, str | None]:
    """Create a safe cross-validation splitter or skip with a warning."""

    n_samples = int(len(y_train))
    if n_samples < 2:
        return None, "Cross-validation was skipped because the training set is too small."

    if task_type == "classification":
        class_counts = Counter(y_train.dropna())
        if len(class_counts) < 2:
            return None, "Cross-validation was skipped because classification labels are not diverse enough."
        max_folds = min(5, max(2, n_samples // 2), min(class_counts.values()))
        if max_folds < 2:
            return None, "Cross-validation was skipped because there are not enough samples per class."
        if max_folds < 5:
            warning = f"Cross-validation folds were reduced to {max_folds} because the dataset is small."
        else:
            warning = None
        return StratifiedKFold(n_splits=max_folds, shuffle=True, random_state=42), warning

    max_folds = min(5, max(2, n_samples // 2))
    if max_folds < 2:
        return None, "Cross-validation was skipped because the training set is too small."
    warning = f"Cross-validation folds were reduced to {max_folds} because the dataset is small." if max_folds < 5 else None
    return KFold(n_splits=max_folds, shuffle=True, random_state=42), warning


def _get_scoring_metric(task_type: str) -> str:
    """Return the scoring metric for the given task type."""

    return "f1_weighted" if task_type == "classification" else "r2"


def _failure_result(
    built_pipeline_result: BuiltPipelineResult,
    target_column: str,
    task_type: str,
    exc: Exception,
) -> TrainingResult:
    """Build a failed training result."""

    return TrainingResult(
        success=False,
        trained_pipeline=None,
        model_name=built_pipeline_result.model_name if isinstance(built_pipeline_result, BuiltPipelineResult) else None,
        task_type=task_type if isinstance(task_type, str) else None,
        target_column=target_column if isinstance(target_column, str) else None,
        train_row_count=None,
        test_row_count=None,
        training_time_seconds=None,
        cross_validation_scores=[],
        mean_cv_score=None,
        std_cv_score=None,
        scoring_metric=_get_scoring_metric(task_type) if task_type in {"classification", "regression"} else None,
        messages=[str(exc)],
        warnings=[],
        errors=[str(exc)],
    )