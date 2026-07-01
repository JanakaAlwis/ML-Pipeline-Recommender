"""Streamlit application entry point for the thesis project.

This file only wires the multi-page shell together. The backend workflow is
connected in later phases.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Callable

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

APP_TITLE = "Explainable ML Pipeline Recommendation System"
APP_SUBTITLE = (
	"A thesis prototype for recommending, training, evaluating, and explaining "
	"tabular machine learning pipelines."
)
APP_ICON = "🤖📊"

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
	"explainability_result",
]


def initialize_session_state() -> None:
	"""Create the shared session state used by the future workflow."""

	for key in APP_SESSION_KEYS:
		st.session_state.setdefault(key, None)


def render_header(page_title: str, page_description: str, next_step: str | None = None, result_meaning: str | None = None) -> None:
	"""Render a consistent page header for the thesis UI."""

	st.markdown(f"## {page_title}")
	st.caption(page_description)
	if next_step is not None:
		st.info(f"Next step: {next_step}")
	if result_meaning is not None:
		st.write(f"What this means: {result_meaning}")


def _workflow_step_states() -> list[tuple[str, bool]]:
	"""Return the workflow steps and completion state."""

	return [
		("Upload", st.session_state.dataframe is not None),
		("Profile", st.session_state.profile_result is not None),
		("Detect", st.session_state.task_result is not None),
		("Recommend", st.session_state.recommendation_result is not None),
		("Train", st.session_state.training_result is not None),
		("Evaluate", st.session_state.evaluation_result is not None),
		("Explain", st.session_state.explainability_result is not None),
	]


def render_workflow_progress() -> None:
	"""Render a compact workflow progress summary."""

	steps = [f"{'✅' if completed else '⬜'} {label}" for label, completed in _workflow_step_states()]
	st.markdown("**Workflow progress**")
	st.markdown(" → ".join(steps))


def render_sidebar_status() -> None:
	"""Render the branded sidebar with workflow guidance."""

	st.sidebar.markdown(f"### {APP_ICON} {APP_TITLE}")
	st.sidebar.caption("Thesis demo navigation and workflow status")
	st.sidebar.markdown("**Workflow steps**")
	for label, completed in _workflow_step_states():
		st.sidebar.write(f"{'✅' if completed else '⬜'} {label}")
	st.sidebar.divider()
	st.sidebar.markdown("**Current status**")
	st.sidebar.write(f"Dataset loaded: {'Yes' if st.session_state.dataframe is not None else 'No'}")
	st.sidebar.write(f"Target column: {st.session_state.target_column or 'Not set'}")
	st.sidebar.write(f"Task detected: {'Yes' if st.session_state.task_result is not None else 'No'}")
	st.sidebar.write(f"Model trained: {'Yes' if st.session_state.training_result is not None else 'No'}")
	st.sidebar.divider()
	st.sidebar.markdown("**How to use**")
	st.sidebar.write("1. Upload a CSV file.")
	st.sidebar.write("2. Profile the dataset and detect the task.")
	st.sidebar.write("3. Choose a recommendation, train, evaluate, and explain the model.")


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


def _get_pages() -> dict[str, Callable[[], None]]:
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
		page_title=APP_TITLE,
		page_icon=APP_ICON,
		layout="wide",
		initial_sidebar_state="expanded",
	)

	st.markdown(
		"""
		<style>
			.block-container {
				padding-top: 1.5rem;
				padding-bottom: 2rem;
			}
			section[data-testid="stSidebar"] {
				padding-top: 0.5rem;
			}
		</style>
		""",
		unsafe_allow_html=True,
	)

	render_sidebar_status()
	render_workflow_progress()

	pages = _get_pages()
	page_name = st.sidebar.radio("Go to", list(pages.keys()), index=0)

	pages[page_name]()


if __name__ == "__main__":
	main()
