"""Structure tests for the Streamlit application shell."""

from __future__ import annotations

from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1] / "app"
PAGE_DIR = APP_DIR / "pages"
EXPECTED_FILES = [
    APP_DIR / "streamlit_app.py",
    PAGE_DIR / "home.py",
    PAGE_DIR / "upload_dataset.py",
    PAGE_DIR / "dataset_profile.py",
    PAGE_DIR / "task_detection.py",
    PAGE_DIR / "recommendations.py",
    PAGE_DIR / "training.py",
    PAGE_DIR / "evaluation.py",
    PAGE_DIR / "explainability.py",
    PAGE_DIR / "report.py",
]

EXPECTED_SESSION_STATE_KEYS = [
    "dataframe",
    "target_column",
    "profile_result",
    "task_result",
    "recommendation_result",
    "selected_pipeline",
    "built_pipeline_result",
    "training_result",
    "evaluation_result",
    "explainability_result",
]


def test_expected_app_files_exist() -> None:
    """All required Streamlit app files should exist."""

    for file_path in EXPECTED_FILES:
        assert file_path.exists(), f"Missing app file: {file_path}"


def test_streamlit_entry_file_exists() -> None:
    """The main Streamlit entry point should exist."""

    assert (APP_DIR / "streamlit_app.py").exists()


def test_expected_session_state_keys_are_defined_in_app() -> None:
    """The app should define the shared session state keys in code."""

    app_source = (APP_DIR / "streamlit_app.py").read_text(encoding="utf-8")
    for key in EXPECTED_SESSION_STATE_KEYS:
        assert key in app_source