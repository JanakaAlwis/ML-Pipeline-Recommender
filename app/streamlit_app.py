"""Streamlit application entry point for the thesis project.

This file only wires the multi-page shell together. The backend workflow is
connected in later phases.
"""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from pages import (
	dataset_profile,
	evaluation,
	explainability,
	home,
	recommendations,
	report,
	task_detection,
	training,
	upload_dataset,
)

APP_SESSION_KEYS = [
	"dataframe",
	"target_column",
	"profile_result",
	"task_result",
	"recommendation_result",
	"selected_pipeline",
	"built_pipeline_result",
	"training_result",
	"evaluation_result",
]


def initialize_session_state() -> None:
	"""Create the shared session state used by the future workflow."""

	for key in APP_SESSION_KEYS:
		st.session_state.setdefault(key, None)


def _get_pages() -> dict[str, callable]:
	"""Return the page render functions keyed by their sidebar labels."""

	return {
		"Home": home.render,
		"Upload Dataset": upload_dataset.render,
		"Dataset Profile": dataset_profile.render,
		"Task Detection": task_detection.render,
		"Pipeline Recommendations": recommendations.render,
		"Training": training.render,
		"Evaluation": evaluation.render,
		"Explainability": explainability.render,
		"Report": report.render,
	}


def main() -> None:
	"""Render the Streamlit application shell."""

	initialize_session_state()

	st.set_page_config(
		page_title="ML Pipeline Recommender",
		page_icon="📊",
		layout="wide",
		initial_sidebar_state="expanded",
	)

	st.sidebar.title("Workflow")
	st.sidebar.caption("Thesis demo navigation")

	pages = _get_pages()
	page_name = st.sidebar.radio("Go to", list(pages.keys()), index=0)

	st.sidebar.divider()
	st.sidebar.subheader("Current state")
	st.sidebar.write(f"Dataset loaded: {'Yes' if st.session_state.dataframe is not None else 'No'}")
	st.sidebar.write(f"Target column: {st.session_state.target_column or 'Not set'}")

	pages[page_name]()


if __name__ == "__main__":
	main()
