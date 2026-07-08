"""Tests for shared display formatting helpers."""

from __future__ import annotations

from src.display_formatting import format_percentage


def test_format_percentage_formats_numeric_values() -> None:
    """Numeric percentage values should render with two decimals."""

    assert format_percentage(20.5678) == "20.57%"


def test_format_percentage_preserves_percentage_strings() -> None:
    """Already formatted percentage strings should remain readable."""

    assert format_percentage("20.57%") == "20.57%"


def test_format_percentage_converts_numeric_strings() -> None:
    """Numeric strings should also render as percentages."""

    assert format_percentage("0.24") == "0.24%"