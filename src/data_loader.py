"""Dataset Loader module for Phase 1.

This module provides a small and explainable CSV loading boundary for the
thesis project. It supports local file paths and Streamlit-style uploaded
file objects, validates the input, and returns a structured result instead of
raising raw parsing errors to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
import io
import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CSVLoadResult:
	"""Container for the outcome of a CSV load attempt.

	Attributes:
		success: Whether the load completed successfully.
		dataframe: Loaded DataFrame, or None if loading failed.
		messages: Combined user-facing messages.
		source_name: File name or path label for the source.
		file_size_bytes: Size of the source in bytes, when known.
		rows: Number of loaded rows, when successful.
		columns: Number of loaded columns, when successful.
		encoding_used: Encoding that succeeded, when applicable.
		delimiter_used: CSV delimiter used for parsing.
		warnings: Warning messages collected during loading.
		errors: Error messages collected during loading.
	"""

	success: bool
	dataframe: pd.DataFrame | None
	messages: list[str]
	source_name: str
	file_size_bytes: int | None
	rows: int | None
	columns: int | None
	encoding_used: str | None
	delimiter_used: str | None
	warnings: list[str]
	errors: list[str]


class DatasetLoaderError(Exception):
	"""Base class for dataset loader errors."""


class UnsupportedFileTypeError(DatasetLoaderError):
	"""Raised when a file is not a CSV source."""


class FileNotFoundLoaderError(DatasetLoaderError):
	"""Raised when a local file path does not exist."""


class EmptyFileError(DatasetLoaderError):
	"""Raised when a source has no readable content."""


class FileReadError(DatasetLoaderError):
	"""Raised when the source cannot be read."""


class EncodingDetectionError(DatasetLoaderError):
	"""Raised when none of the supported encodings can decode the file."""


class CSVParseError(DatasetLoaderError):
	"""Raised when pandas cannot parse the CSV content."""


@runtime_checkable
class SupportsRead(Protocol):
	"""Protocol for file-like objects that expose a read method."""

	name: str | None

	def read(self, size: int = -1) -> bytes | str:
		"""Read content from the file-like object."""


def load_csv(source: str | Path | SupportsRead) -> CSVLoadResult:
	"""Load a CSV from a path or file-like object.

	Args:
		source: Local CSV file path or Streamlit-style uploaded file.

	Returns:
		A structured CSVLoadResult instance.
	"""

	if isinstance(source, (str, Path)):
		return load_csv_from_path(source)
	if hasattr(source, "read"):
		return load_csv_from_uploaded_file(source)

	return build_loader_result(
		success=False,
		dataframe=None,
		source_name=_resolve_source_name(source),
		file_size_bytes=None,
		encoding_used=None,
		delimiter_used=None,
		warnings=[],
		errors=[f"Unsupported source type: {_resolve_source_name(source)}"],
	)


def load_csv_from_path(source: str | Path) -> CSVLoadResult:
	"""Load a CSV from a local filesystem path.

	Args:
		source: Path to a CSV file.

	Returns:
		A structured CSVLoadResult instance.
	"""

	source_name = str(Path(source))
	try:
		logger.info("Loading CSV source: %s", source_name)
		_validate_path_source(Path(source), source_name)
		file_bytes = Path(source).read_bytes()
		if not file_bytes:
			logger.error("Source is empty: %s", source_name)
			raise EmptyFileError(f"File is empty: {source_name}")
		dataframe, encoding_used = read_csv_with_fallback(file_bytes, source_name)
		logger.info(
			"CSV loaded successfully: rows=%s, columns=%s",
			dataframe.shape[0],
			dataframe.shape[1],
		)
		return build_loader_result(
			success=True,
			dataframe=dataframe,
			source_name=source_name,
			file_size_bytes=len(file_bytes),
			encoding_used=encoding_used,
			delimiter_used=",",
			warnings=[],
			errors=[],
		)
	except DatasetLoaderError as exc:
		logger.error("CSV loading failed: %s", exc)
		return build_loader_result(
			success=False,
			dataframe=None,
			source_name=source_name,
			file_size_bytes=None,
			encoding_used=None,
			delimiter_used=None,
			warnings=[],
			errors=[str(exc)],
		)


def load_csv_from_uploaded_file(source: SupportsRead) -> CSVLoadResult:
	"""Load a CSV from a file-like uploaded object.

	Args:
		source: Streamlit-style uploaded file or similar file-like object.

	Returns:
		A structured CSVLoadResult instance.
	"""

	source_name = _resolve_source_name(source)
	try:
		logger.info("Loading CSV source: %s", source_name)
		_validate_uploaded_source(source, source_name)
		file_bytes = _read_source_bytes(source, source_name)
		if not file_bytes:
			logger.error("Source is empty: %s", source_name)
			raise EmptyFileError(f"File is empty: {source_name}")
		dataframe, encoding_used = read_csv_with_fallback(file_bytes, source_name)
		logger.info(
			"CSV loaded successfully: rows=%s, columns=%s",
			dataframe.shape[0],
			dataframe.shape[1],
		)
		return build_loader_result(
			success=True,
			dataframe=dataframe,
			source_name=source_name,
			file_size_bytes=len(file_bytes),
			encoding_used=encoding_used,
			delimiter_used=",",
			warnings=[],
			errors=[],
		)
	except DatasetLoaderError as exc:
		logger.error("CSV loading failed: %s", exc)
		return build_loader_result(
			success=False,
			dataframe=None,
			source_name=source_name,
			file_size_bytes=None,
			encoding_used=None,
			delimiter_used=None,
			warnings=[],
			errors=[str(exc)],
		)


def validate_csv_source(source: str | Path | SupportsRead, source_name: str | None = None) -> None:
	"""Validate that the source is a CSV and is readable.

	Args:
		source: Input source to validate.
		source_name: Optional display name for the source.

	Raises:
		UnsupportedFileTypeError: If a named source is not a CSV.
		FileNotFoundLoaderError: If a local path does not exist.
		EmptyFileError: If the file is empty.
		FileReadError: If the source cannot be read.
	"""

	display_name = source_name or _resolve_source_name(source)

	if isinstance(source, (str, Path)):
		_validate_path_source(Path(source), display_name)
		return

	_validate_uploaded_source(source, display_name)


def detect_encoding(file_bytes: bytes, source_name: str) -> tuple[str, str]:
	"""Decode bytes using a simple fallback encoding strategy.

	Args:
		file_bytes: Raw file content.
		source_name: Display name for logging and errors.

	Returns:
		A tuple of decoded text and the encoding that succeeded.

	Raises:
		EncodingDetectionError: If decoding fails for all supported encodings.
	"""

	encodings = ["utf-8", "utf-8-sig", "latin1"]
	for encoding in encodings:
		try:
			logger.info("Attempting %s decode for: %s", encoding, source_name)
			return file_bytes.decode(encoding), encoding
		except UnicodeDecodeError:
			logger.warning("Decode failed for %s using %s", source_name, encoding)

	raise EncodingDetectionError(f"Could not decode file using supported encodings: {source_name}")


def read_csv_with_fallback(file_bytes: bytes, source_name: str) -> tuple[pd.DataFrame, str]:
	"""Read CSV content with a simple encoding fallback strategy.

	Args:
		file_bytes: Raw file content.
		source_name: Display name for logging and errors.

	Returns:
		A tuple containing the DataFrame and the encoding used.

	Raises:
		EmptyFileError: If the file is empty.
		EncodingDetectionError: If decoding fails.
		CSVParseError: If pandas cannot parse the CSV.
	"""

	if not file_bytes:
		raise EmptyFileError(f"File is empty: {source_name}")

	decoded_text, encoding_used = detect_encoding(file_bytes, source_name)

	try:
		dataframe = pd.read_csv(io.StringIO(decoded_text))
	except pd.errors.EmptyDataError as exc:
		logger.error("CSV parse failed for source: %s", source_name)
		raise CSVParseError(f"CSV is empty or malformed: {source_name}") from exc
	except pd.errors.ParserError as exc:
		logger.error("CSV parse failed for source: %s", source_name)
		raise CSVParseError(f"Could not parse CSV: {source_name}") from exc
	except ValueError as exc:
		logger.error("CSV parse failed for source: %s", source_name)
		raise CSVParseError(f"Could not parse CSV: {source_name}") from exc

	if dataframe.shape[1] == 0:
		logger.error("No columns found after parsing: %s", source_name)
		raise CSVParseError(f"CSV has zero columns: {source_name}")
	if dataframe.shape[0] == 0:
		logger.error("No rows found after parsing: %s", source_name)
		raise CSVParseError(f"CSV has zero rows: {source_name}")

	return dataframe, encoding_used


def build_loader_result(
	*,
	success: bool,
	dataframe: pd.DataFrame | None,
	source_name: str,
	file_size_bytes: int | None,
	encoding_used: str | None,
	delimiter_used: str | None,
	warnings: list[str],
	errors: list[str],
) -> CSVLoadResult:
	"""Build a standardized CSV load result.

	Args:
		success: Whether loading succeeded.
		dataframe: Loaded DataFrame, if any.
		source_name: File name or label for the source.
		file_size_bytes: File size in bytes, when known.
		encoding_used: Encoding that succeeded, when known.
		delimiter_used: Delimiter used for parsing, when known.
		warnings: Warning messages.
		errors: Error messages.

	Returns:
		A CSVLoadResult instance.
	"""

	messages = [*warnings, *errors]
	rows = int(dataframe.shape[0]) if dataframe is not None else None
	columns = int(dataframe.shape[1]) if dataframe is not None else None
	return CSVLoadResult(
		success=success,
		dataframe=dataframe,
		messages=messages,
		source_name=source_name,
		file_size_bytes=file_size_bytes,
		rows=rows,
		columns=columns,
		encoding_used=encoding_used,
		delimiter_used=delimiter_used,
		warnings=warnings,
		errors=errors,
	)


def _validate_path_source(path: Path, source_name: str) -> None:
	"""Validate a local filesystem path.

	Args:
		path: Local CSV path.
		source_name: Display name for logging and errors.

	Raises:
		UnsupportedFileTypeError: If the extension is not .csv.
		FileNotFoundLoaderError: If the file does not exist.
		FileReadError: If the path is not a readable file.
	"""

	if path.suffix.lower() != ".csv":
		logger.error("Unsupported file type rejected: %s", source_name)
		raise UnsupportedFileTypeError(f"Only CSV files are supported: {source_name}")
	if not path.exists():
		logger.error("Missing file rejected: %s", source_name)
		raise FileNotFoundLoaderError(f"File does not exist: {source_name}")
	if not path.is_file():
		raise FileReadError(f"Path is not a file: {source_name}")

	try:
		if path.stat().st_size == 0:
			logger.error("Source is empty: %s", source_name)
			raise EmptyFileError(f"File is empty: {source_name}")
	except OSError as exc:
		raise FileReadError(f"Could not inspect file: {source_name}") from exc


def _validate_uploaded_source(source: SupportsRead, source_name: str) -> None:
	"""Validate a file-like uploaded source.

	Args:
		source: Uploaded file-like object.
		source_name: Display name for logging and errors.

	Raises:
		UnsupportedFileTypeError: If the provided file name is not a CSV.
		FileReadError: If the source cannot be read.
	"""

	if hasattr(source, "name") and getattr(source, "name"):
		name = str(getattr(source, "name"))
		if Path(name).suffix.lower() != ".csv":
			logger.error("Unsupported file type rejected: %s", source_name)
			raise UnsupportedFileTypeError(f"Only CSV files are supported: {source_name}")

	if not hasattr(source, "read"):
		raise FileReadError(f"Unsupported source type: {source_name}")


def _resolve_source_name(source: str | Path | SupportsRead) -> str:
	"""Resolve a human-readable name for the source."""

	if isinstance(source, (str, Path)):
		return str(source)

	name = getattr(source, "name", None)
	if name:
		return str(name)
	return "uploaded_file"


def _read_source_bytes(source: SupportsRead, source_name: str) -> bytes:
	"""Read bytes from a file-like source.

	Args:
		source: File-like object.
		source_name: Display name for error messages.

	Returns:
		Raw file bytes.

	Raises:
		FileReadError: If the source cannot be read.
	"""

	try:
		content = source.read()
	except OSError as exc:
		raise FileReadError(f"Could not read file: {source_name}") from exc

	if isinstance(content, str):
		return content.encode("utf-8")
	if isinstance(content, (bytes, bytearray)):
		return bytes(content)

	raise FileReadError(f"Unsupported content type for source: {source_name}")
