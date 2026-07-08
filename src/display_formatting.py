"""Shared numeric formatting helpers for Streamlit and Markdown output."""

from __future__ import annotations

from numbers import Real


def _is_number(value: object) -> bool:
    """Return True when the value is a real number that should be formatted."""

    return isinstance(value, Real) and not isinstance(value, bool)


def format_count(value: object) -> str:
    """Format a count as an integer string."""

    if not _is_number(value):
        return "N/A"
    return str(int(round(float(value))))


def format_confidence(value: object) -> str:
    """Format a confidence score with up to six decimal places."""

    if not _is_number(value):
        return "N/A"
    formatted = f"{float(value):.6f}"
    return formatted.rstrip("0").rstrip(".") if "." in formatted else formatted


def format_metric(value: object) -> str:
    """Format an evaluation metric with four decimal places."""

    if not _is_number(value):
        return "N/A"
    return f"{float(value):.4f}"


def format_cv_score(value: object) -> str:
    """Format a cross-validation score with four decimal places."""

    return format_metric(value)


def format_percentage(value: object) -> str:
    """Format a percentage value with two decimal places."""

    if not _is_number(value):
        return "N/A"
    return f"{float(value):.2f}%"


def format_seconds(value: object) -> str:
    """Format seconds with three decimal places."""

    if not _is_number(value):
        return "N/A"
    return f"{float(value):.3f} s"