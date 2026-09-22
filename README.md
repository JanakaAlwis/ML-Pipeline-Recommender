# Design and Implementation of an Explainable Machine Learning Pipeline Recommendation System for Structured Tabular Data

This project is a thesis prototype for a knowledge-based machine learning pipeline recommendation system for structured tabular data. The application guides a user from dataset upload through profiling, task detection, pipeline recommendation, training, evaluation, explainability, and final report generation in a single Streamlit workflow.

[Installation and Local Setup](INSTALLATION.md)

## Overview

The system is designed for tabular CSV datasets where the user needs a transparent and explainable starting point for supervised learning tasks. Instead of treating pipeline selection as a black box, the project makes the recommendation process explicit: it profiles the dataset, identifies the learning task, matches dataset characteristics against a declarative knowledge base, and then builds a concrete pipeline for the selected recommendation.

The practical goal is not to guarantee an optimal pipeline or to claim superiority over AutoML systems. It is to provide a beginner-friendly, traceable workflow that supports academic discussion, transparent decision making, and reproducible experimentation on structured data.

## Main Capabilities

The current implementation verifies the following workflow capabilities:

- CSV upload and dataset loading
- Dataset profiling and summary statistics
- Feature assessment before model training
- Classification and regression task detection
- Knowledge-based pipeline recommendation for structured tabular data
- Pipeline construction, selection, and training
- Evaluation metrics for classification and regression
- Model explainability using permutation importance
- Optional SHAP explanations for supported tree-based models
- Final Markdown report generation

## Explainability and Feature Assessment

Explainability is a central design goal of the project.

The workflow includes:

- a pre-modeling feature assessment step that highlights features that may need review,
- recommendation explanations that describe why a pipeline was selected,
- permutation importance after training to assess the most influential features,
- optional SHAP output for supported tree-based models when the model type permits it.

Important detail: feature assessment flags suspicious or weakly suitable features for review, but the implementation does not automatically remove or exclude them from the model. Flagged features remain visible in the assessment output and are only treated as review signals unless a later implementation step explicitly modifies the data or pipeline.

## Application Workflow

The actual user journey in the application is:

1. Upload a CSV dataset.
2. Review the dataset profile and quality indicators.
3. Select the target column and detect the task type.
4. View ranked pipeline recommendations.
5. Choose a recommended pipeline and train it.
6. Evaluate the trained model on the hold-out split.
7. Review explainability outputs and feature importance.
8. Generate the final Markdown report.

The app is implemented as a guided Streamlit workflow and is intended for thesis demonstration and research-oriented experimentation rather than production deployment.

## Thesis Evaluation Summary

The project documentation and implementation evidence support the following evaluation context:

- The system was evaluated on six tabular datasets as part of the thesis workflow review.
- The recommendation process was exercised through 30 pipeline recommendation executions in the evaluation workflow.
- The current automated test evidence for the repository is 173 passing tests, with one existing scikit-learn warning from a regression report test.

This repository does not claim a formal browser-level end-to-end UI test suite, a large-scale user study, or a production benchmark against external AutoML systems.

## Known Limitations

The current implementation is intentionally limited and transparent about these constraints:

- Recommendation logic is rule-based and knowledge-base-driven rather than learned from historical performance data.
- Recommendation consistency depends on the current declarative rules and dataset characteristics; it is not a general-purpose optimization engine.
- Feature assessment can flag features that may be unsuitable, but the system does not automatically enforce exclusion unless a later step explicitly does so.
- Explanation quality is constrained by the current methods: the primary workflow relies on permutation importance, with optional SHAP only for supported models.
- Evaluation is designed for the project’s thesis context and uses project-defined heuristics rather than a universally accepted benchmark standard.
- The executable report flow is Markdown-based; PDF export is not implemented in the current application.

## Future Development

The following directions are realistic future improvements, but they are not presented as current functionality:

- expanding the rule base and pipeline archetypes,
- improving recommendation consistency across more dataset types,
- adding deeper explainability methods and richer reporting,
- improving model persistence and experiment tracking,
- extending the application for broader deployment or demonstration scenarios.

## Repository Layout

- app/ — Streamlit application shell and workflow pages
- src/ — recommendation, preprocessing, training, evaluation, explainability, and reporting logic
- tests/ — automated validation for the core workflow components
- docs/ — thesis-related project documentation and planning materials
- reference/ — prior notebooks and supporting datasets
- models/ — generated model artifacts when produced by the workflow
- outputs/ — generated reports and other workflow outputs

## Validation and Scope

This repository is a thesis prototype and demonstration application for explainable machine learning pipeline recommendation on structured tabular data. It is designed to support transparent, academic evaluation and workflow explanation rather than serving as a production-grade platform.

## Related Documentation

- [Installation and Local Setup](INSTALLATION.md)
- [docs/DEVELOPMENT_GUIDELINES.md](docs/DEVELOPMENT_GUIDELINES.md)
