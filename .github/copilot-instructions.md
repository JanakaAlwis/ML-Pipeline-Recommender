# GitHub Copilot Instructions

This repository is for a bachelor's thesis project.

Project title:
Design and Implementation of an Explainable Machine Learning Pipeline Recommendation System for Tabular Datasets

The primary project specification is:
docs/ML_Pipeline_Recommender_Development_Plan.docx

Use the files in docs/ and reference/ only to understand the project context, thesis expectations, previous machine learning assignments, and coding style.

Do not copy previous assignments directly.

## Development Rules

- Use the active virtual environment thesis_env. The current Python version is acceptable if all required packages work.
- Use modular architecture.
- Do not put all code inside one file.
- Prefer readable beginner-friendly code.
- Use type hints.
- Use Google-style docstrings.
- Use logging where useful.
- Use dataclasses for structured outputs.
- Use scikit-learn Pipelines where possible.
- Write unit tests for each module.
- Keep functions small and easy to explain.
- Every important design decision should be understandable for a thesis defense.

## Main Libraries

Use:
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- xgboost
- shap
- matplotlib
- plotly
- streamlit
- joblib
- pytest

## Development Order

1. Dataset Loader
2. Dataset Profiler
3. Task Detector
4. Recommendation Engine
5. Preprocessing Builder
6. Model Trainer
7. Evaluator
8. Explainability Module
9. Report Generator
10. Streamlit Web Application

## Important Rule

Never generate the whole project at once.

Always implement one module at a time.