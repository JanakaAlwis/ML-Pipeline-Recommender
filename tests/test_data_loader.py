"""Tests for the Dataset Loader module."""

from __future__ import annotations

from dataclasses import fields
import io
from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import (
	CSVLoadResult,
	load_csv,
	load_csv_from_path,
	load_csv_from_uploaded_file,
)


class FakeUploadedFile:
	"""Simple file-like stand-in for a Streamlit uploaded file."""

	def __init__(self, content: bytes, name: str) -> None:
		self._buffer = io.BytesIO(content)
		self.name = name

	def read(self, size: int = -1) -> bytes:
		return self._buffer.read(size)


def test_loader_result_has_expected_fields() -> None:
	"""The result dataclass should expose the agreed Phase 1 fields."""

	field_names = [field.name for field in fields(CSVLoadResult)]
	assert field_names == [
		"success",
		"dataframe",
		"messages",
		"source_name",
		"file_size_bytes",
		"rows",
		"columns",
		"encoding_used",
		"delimiter_used",
		"warnings",
		"errors",
	]


def test_load_csv_from_path_success(tmp_path: Path) -> None:
	"""A valid CSV file path should load successfully."""

	csv_path = tmp_path / "sample.csv"
	csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

	result = load_csv_from_path(csv_path)

	assert result.success is True
	assert isinstance(result.dataframe, pd.DataFrame)
	assert result.rows == 2
	assert result.columns == 2
	assert result.encoding_used == "utf-8"
	assert result.errors == []


def test_load_csv_from_uploaded_file_success() -> None:
	"""A valid uploaded-file-like object should load successfully."""

	uploaded = FakeUploadedFile(b"a,b\n1,2\n3,4\n", "sample.csv")

	result = load_csv_from_uploaded_file(uploaded)

	assert result.success is True
	assert isinstance(result.dataframe, pd.DataFrame)
	assert result.rows == 2
	assert result.columns == 2
	assert result.source_name == "sample.csv"


def test_load_csv_rejects_unsupported_extension(tmp_path: Path) -> None:
	"""Files with a non-CSV extension should be rejected."""

	file_path = tmp_path / "sample.txt"
	file_path.write_text("a,b\n1,2\n", encoding="utf-8")

	result = load_csv(file_path)

	assert result.success is False
	assert any("Only CSV files are supported" in error for error in result.errors)


def test_load_csv_rejects_missing_file(tmp_path: Path) -> None:
	"""Missing files should return a clear failure result."""

	file_path = tmp_path / "missing.csv"

	result = load_csv(file_path)

	assert result.success is False
	assert any("File does not exist" in error for error in result.errors)


def test_load_csv_rejects_empty_file(tmp_path: Path) -> None:
	"""Empty files should return a clear failure result."""

	file_path = tmp_path / "empty.csv"
	file_path.write_text("", encoding="utf-8")

	result = load_csv(file_path)

	assert result.success is False
	assert any("File is empty" in error for error in result.errors)


def test_load_csv_rejects_zero_rows(tmp_path: Path) -> None:
	"""CSV files with only headers should be rejected."""

	file_path = tmp_path / "headers_only.csv"
	file_path.write_text("a,b\n", encoding="utf-8")

	result = load_csv(file_path)

	assert result.success is False
	assert any("zero rows" in error for error in result.errors)


def test_load_csv_uses_latin1_fallback(tmp_path: Path) -> None:
	"""Files that are not valid UTF-8 should fall back to latin1."""

	file_path = tmp_path / "latin1.csv"
	file_path.write_bytes("name\nCaf\xe9\n".encode("latin1"))

	result = load_csv(file_path)

	assert result.success is True
	assert result.encoding_used == "latin1"
	assert result.dataframe is not None
	assert result.dataframe.iloc[0, 0] == "Café"


def test_load_csv_rejects_malformed_csv(tmp_path: Path) -> None:
	"""Malformed CSV content should fail with a parse error."""

	file_path = tmp_path / "malformed.csv"
	file_path.write_text('a,b\n1,"2\n', encoding="utf-8")

	result = load_csv(file_path)

	assert result.success is False
	assert any("Could not parse CSV" in error or "CSV is empty or malformed" in error for error in result.errors)


def test_load_csv_returns_failure_result_for_invalid_type() -> None:
	"""Unsupported source types should return a failure result."""

	result = load_csv(123)  # type: ignore[arg-type]

	assert result.success is False
	assert result.dataframe is None
	assert result.errors
