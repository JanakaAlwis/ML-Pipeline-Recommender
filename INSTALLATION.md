# Installation and Local Setup

[← Back to Project Overview](README.md)

This guide explains how to set up and run the project locally. The repository is a Python-based Streamlit application for explainable machine learning pipeline recommendation on structured tabular data.

## Requirements

The project configuration declares:

- Python 3.12 or newer
- a local virtual environment or Conda environment
- access to the repository files
- no CUDA or NVIDIA GPU requirement

The application is intended for local research and thesis demonstration use rather than production deployment.

## 1. Clone or Download the Repository

Use your preferred method to obtain the project files.

If you are using Git, use the following command:

```bash
git clone https://github.com/JanakaAlwis/ML-Pipeline-Recommender.git
cd ML-Pipeline-Recommender
```

## 2. Create a Python Environment

### Option A: Conda

```bash
conda create -n ml-pipeline-recommender python=3.12
conda activate ml-pipeline-recommender
```

### Option B: venv

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On Windows Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

## 3. Install Dependencies

From the project root, install the dependencies listed in the repository dependency file:

```bash
pip install -r requirements.txt
```

The current dependency file includes the libraries required by the implemented workflow, including pandas, numpy, scikit-learn, imbalanced-learn, streamlit, xgboost, shap, and pytest.

## 4. Required Configuration

No additional project-specific configuration file is required to run the application locally.

The verified app entry point is:

```bash
app/streamlit_app.py
```

The repository does not require a CUDA-enabled environment or any NVIDIA GPU setup.

## 5. Start the Application

From the project root, run:

```bash
streamlit run app/streamlit_app.py
```

This is the verified Streamlit entry point for the project.

## 6. Open the Application in a Browser

After the command starts successfully, Streamlit will print a local URL in the terminal, typically:

```text
Local URL: http://localhost:8501
```

Open that URL in a browser to use the application.

## 7. First-Time Verification Procedure

To validate the application in a simple end-to-end check without using production data:

1. Open the app in the browser.
2. Upload a small CSV file with a clear tabular structure.
3. Select a target column that represents a supervised learning target.
4. Generate the dataset profile.
5. Detect the task type.
6. Review the recommended pipelines.
7. Choose a recommended pipeline and train it.
8. Run evaluation.
9. Review explainability output.
10. Generate the final Markdown report.

A CSV with a single target column and several feature columns is sufficient for a first-use check. The project is designed for structured tabular data and is not intended for multi-table, image, text, or time-series workflows.

## 8. Optional Automated Tests

If you want to run the repository test suite after installation:

```bash
python -m pytest
```

The project documentation and audit evidence currently support a repository test run with 173 passing tests and one existing warning from a scikit-learn regression-report check. This is not a substitute for browser-level UI testing.

## 9. Troubleshooting

### Python version mismatch

If you see version-related errors, make sure you are using Python 3.12 or newer, as declared in the project configuration.

### Dependencies fail to install

Update pip and retry:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Streamlit does not start

Confirm you are in the repository root and that the entry point is correct:

```bash
streamlit run app/streamlit_app.py
```

### Import errors or missing packages

Make sure the active environment is the one in which you installed the project requirements.

### Dataset or workflow issues

The project expects structured tabular CSV input. If a dataset is not suitable, check that:

- the file is a CSV,
- the target column is selected correctly,
- the data contains usable feature columns,
- missing values and feature types are appropriate for the selected workflow.

## Notes

This project is a thesis prototype and local research tool for explainable machine learning pipeline recommendation. It is intended for academic, educational, and demonstration purposes and should not be treated as a full production deployment or as a guarantee of the best possible pipeline for every dataset.

[← Back to Project Overview](README.md)
