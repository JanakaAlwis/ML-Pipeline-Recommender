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
	"report_result",
	"current_page",
]

PAGE_ORDER = [
	"Home",
	"Upload Dataset",
	"Dataset Profile",
	"Task Detection",
	"Pipeline Recommendations",
	"Training",
	"Evaluation",
	"Explainability",
	"Report",
]

WORKFLOW_PROGRESS_STEPS = [
	("Upload", "Upload Dataset", "dataframe"),
	("Profile", "Dataset Profile", "profile_result"),
	("Detect", "Task Detection", "task_result"),
	("Recommend", "Pipeline Recommendations", "recommendation_result"),
	("Train", "Training", "training_result"),
	("Evaluate", "Evaluation", "evaluation_result"),
	("Explain", "Explainability", "explainability_result"),
	("Report", "Report", "report_result"),
]

PAGE_REQUIREMENTS: dict[str, list[tuple[str, str]]] = {
	"Home": [],
	"Upload Dataset": [],
	"Dataset Profile": [("dataframe", "Upload a dataset first on the Upload Dataset page.")],
	"Task Detection": [
		("dataframe", "Upload a dataset first on the Upload Dataset page."),
		("profile_result", "Generate a dataset profile before detecting task type."),
	],
	"Pipeline Recommendations": [
		("profile_result", "Generate a dataset profile before requesting recommendations."),
		("task_result", "Detect the task type before requesting recommendations."),
	],
	"Training": [
		("dataframe", "Upload a dataset first on the Upload Dataset page."),
		("profile_result", "Generate a dataset profile before training."),
		("task_result", "Detect the task type before training."),
		("selected_pipeline", "Select a recommended pipeline before training."),
		("target_column", "Choose a target column before training."),
	],
	"Evaluation": [
		("dataframe", "Upload a dataset first on the Upload Dataset page."),
		("task_result", "Detect the task type before evaluation."),
		("training_result", "Train a model before evaluation."),
		("target_column", "Choose a target column before evaluation."),
	],
	"Explainability": [
		("dataframe", "Upload a dataset first on the Upload Dataset page."),
		("task_result", "Detect the task type before explainability."),
		("training_result", "Train a model before explainability."),
		("target_column", "Choose a target column before explainability."),
	],
	"Report": [],
}


def initialize_session_state() -> None:
	"""Create the shared session state used by the future workflow."""

	for key in APP_SESSION_KEYS:
		st.session_state.setdefault(key, None)

	if st.session_state.current_page not in PAGE_ORDER:
		st.session_state.current_page = "Home"


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
		(step_label, st.session_state.get(state_key) is not None)
		for step_label, _, state_key in WORKFLOW_PROGRESS_STEPS
	]


def _workflow_progress_text() -> str:
	"""Build the compact workflow progress line."""

	steps = [f"{'✅' if completed else '⬜'} {label}" for label, completed in _workflow_step_states()]
	return " → ".join(steps)


def render_workflow_progress() -> None:
	"""Render workflow progress in the main page."""

	st.markdown("**Workflow progress**")
	st.markdown(_workflow_progress_text())


def _page_status_label(page_name: str) -> str:
	"""Return a compact label for the current navigation state."""

	if st.session_state.current_page == page_name:
		return f"▶ {page_name}"
	return page_name


def render_sidebar_status() -> None:
	"""Render the branded sidebar with workflow guidance."""

	st.sidebar.markdown(f"### {APP_ICON} {APP_TITLE}")
	st.sidebar.caption("Thesis demo navigation and workflow status")
	st.sidebar.markdown("**Workflow navigation**")
	for page_name in PAGE_ORDER:
		button_type = "primary" if st.session_state.current_page == page_name else "secondary"
		if st.sidebar.button(
			_page_status_label(page_name),
			key=f"nav_{page_name}",
			use_container_width=True,
			type=button_type,
		):
			st.session_state.current_page = page_name
			st.rerun()

	st.sidebar.markdown("**Workflow progress**")
	for label, completed in _workflow_step_states():
		if completed:
			status = "✅ Completed"
		elif st.session_state.current_page == next(
			page_name for step_label, page_name, _ in WORKFLOW_PROGRESS_STEPS if step_label == label
		):
			status = "🔄 Current"
		else:
			status = "⭕ Not completed"
		st.sidebar.write(f"{status} {label}")
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


def _missing_requirements(page_name: str) -> list[str]:
	"""Return missing prerequisite messages for a given page."""

	requirements = PAGE_REQUIREMENTS.get(page_name, [])
	missing: list[str] = []
	for key, message in requirements:
		if st.session_state.get(key) is None:
			missing.append(message)
	return missing


def _go_to_previous_page() -> None:
	"""Move to the previous page in the workflow order."""

	current_index = PAGE_ORDER.index(st.session_state.current_page)
	if current_index > 0:
		st.session_state.current_page = PAGE_ORDER[current_index - 1]
		st.rerun()


def _go_to_next_page() -> None:
	"""Move to the next page if prerequisites are satisfied."""

	current_index = PAGE_ORDER.index(st.session_state.current_page)
	if current_index >= len(PAGE_ORDER) - 1:
		return

	next_page = PAGE_ORDER[current_index + 1]
	missing = _missing_requirements(next_page)
	if missing:
		st.warning(f"Cannot move to {next_page} yet. {missing[0]}")
		return

	st.session_state.current_page = next_page
	st.rerun()


def render_navigation_footer() -> None:
	"""Render Previous and Next controls for all pages."""

	st.divider()
	prev_col, center_col, next_col = st.columns([1, 2, 1])
	with prev_col:
		if st.button("Previous", use_container_width=True):
			_go_to_previous_page()
	with center_col:
		st.caption(f"Current page: {st.session_state.current_page}")
	with next_col:
		if st.button("Next", use_container_width=True):
			_go_to_next_page()


from app.views import (
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

	pages = _get_pages()
	current_page = st.session_state.current_page
	pages[current_page]()
	render_navigation_footer()


if __name__ == "__main__":
	main()
