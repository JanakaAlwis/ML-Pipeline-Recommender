# ML Pipeline Recommender

Bachelor's thesis prototype for explainable machine learning pipeline recommendation on tabular datasets.

## Project Title

Design and Implementation of an Explainable Machine Learning Pipeline Recommendation System for Tabular Datasets

## Purpose

This system analyzes structured CSV datasets, recommends suitable machine learning pipelines, trains the selected pipeline, evaluates performance, explains the result, and generates a final report.

## Core Capabilities

- CSV upload and validation
- Dataset profiling
- Task detection
- Knowledge-base-driven pipeline recommendation
- Pipeline construction and training
- Model evaluation
- Feature-importance explainability
- Markdown and optional PDF reporting
- Streamlit workflow navigation

## Installation

1. Create or activate the Python environment used by the project.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the application with:

```bash
streamlit run app/streamlit_app.py
```

The workflow is session-state driven. Complete the pages in order:

1. Upload Dataset
2. Dataset Profile
3. Task Detection
4. Recommendations
5. Training
6. Evaluation
7. Explainability
8. Report

## Workflow

The application follows a guided supervised-learning workflow for tabular data:

Upload Dataset → Dataset Profile → Task Detection → Recommendation Engine → Pipeline Builder → Model Training → Model Evaluation → Explainability → Report

## Project Structure

- `app/` - Streamlit shell and page views
- `src/` - machine learning, knowledge base, explainability, and reporting modules
- `knowledge_base/` - declarative recommendation rules and archetypes
- `tests/` - pytest suite
- `docs/` - thesis reference material and planning artifacts
- `docs/DEVELOPMENT_GUIDELINES.md` - general project development guidelines
- `reference/` - previous assignment notebooks and datasets used for context
- `models/` - generated model artifacts
- `outputs/` - generated reports and outputs

## Testing

Run the test suite with:

```bash
python -m pytest -v
```

Current validation status:

- `131` tests passing
- `streamlit run app/streamlit_app.py` verified successfully

## Screenshots

Add thesis screenshots here after the final UI review.

## Future Work

- Broaden the knowledge base with additional archetypes and rules.
- Add model persistence and experiment tracking.
- Add optional SHAP-based explanations.
- Add richer PDF formatting for the final report.
- Add deployment guidance for thesis demonstration environments.
