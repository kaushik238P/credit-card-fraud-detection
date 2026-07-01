"""
Custom exception hierarchy for the data ingestion layer.

Design:
    All exceptions inherit from IngestionError so callers can choose between:

    1. Broad handling — catch all ingestion failures with a single clause:
           except IngestionError as exc:
               handle(exc)

    2. Fine-grained handling — catch specific failure types:
           except DatasetNotFoundError:
               ...
           except UnsupportedFileFormatError:
               ...

    Every exception carries a structured context dict that downstream
    consumers (FastAPI error handlers, monitoring systems, MLflow loggers)
    can inspect programmatically without parsing the message string.

Hierarchy:
    Exception
    └── IngestionError
        ├── DatasetNotFoundError
        ├── UnsupportedFileFormatError
        ├── EmptyDatasetError
        └── DatasetReadError
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class IngestionError(Exception):
    """
    Base exception for all data ingestion failures.

    Never raise this class directly. Use one of the concrete subclasses
    so callers can distinguish between failure types.

    Attributes:
        message: Human-readable description of the failure.
        context: Key-value pairs with diagnostic information. This dict is
            intentionally kept serializable (no pandas objects, no DataFrames)
            so it can be logged as structured JSON or attached to MLflow runs.
    """

    def __init__(self, message: str, context: dict | None = None) -> None:
        self.message = message
        self.context: dict = context or {}
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Produces the string passed to Exception.__init__."""
        if not self.context:
            return self.message
        ctx_str = " | ".join(f"{k}={v!r}" for k, v in self.context.items())
        return f"{self.message} [Context: {ctx_str}]"

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"message={self.message!r}, "
            f"context={self.context!r})"
        )


# ---------------------------------------------------------------------------
# Concrete exceptions
# ---------------------------------------------------------------------------


class DatasetNotFoundError(IngestionError):
    """
    Raised when the specified dataset file cannot be located on the filesystem.

    Raised by: DataLoader._validate_path()
    Validation order: FIRST — cheapest check (single stat syscall).

    Conditions:
        - The path does not exist on the filesystem.
        - The path exists but points to a directory, not a file.

    Attributes:
        path: The resolved absolute path that was checked.

    Example:
        >>> raise DatasetNotFoundError(
        ...     path="/data/raw/creditcard.csv",
        ...     detail="Path is a directory, not a file.",
        ... )
    """

    def __init__(self, path: str, detail: str = "") -> None:
        self.path = path
        context: dict = {"path": path}
        if detail:
            context["detail"] = detail
        message = f"Dataset not found at path '{path}'."
        if detail:
            message = f"{message} {detail}"
        super().__init__(message=message, context=context)


class UnsupportedFileFormatError(IngestionError):
    """
    Raised when the file's extension is absent from the supported extensions registry.

    Raised by: DataLoader._validate_extension()
    Validation order: SECOND — dictionary lookup, no I/O.

    Conditions:
        - File exists but its suffix (e.g., '.json') is not registered.
        - File has no extension at all (empty suffix '').

    Attributes:
        extension: The unsupported extension string found on the file.
        supported: The list of currently registered extensions.
        path: The file path that triggered the error.

    Example:
        >>> raise UnsupportedFileFormatError(
        ...     extension=".json",
        ...     supported=[".csv"],
        ...     path="/data/raw/creditcard.json",
        ... )
    """

    def __init__(
        self,
        extension: str,
        supported: list[str],
        path: str = "",
    ) -> None:
        self.extension = extension
        self.supported = supported
        self.path = path
        super().__init__(
            message=(
                f"Unsupported file format '{extension}'. "
                f"Currently supported formats: {supported}. "
                "To add support for this format, register a new reader in "
                "DataLoader._READER_DISPATCH."
            ),
            context={
                "extension": extension,
                "supported": supported,
                "path": path,
            },
        )


class EmptyDatasetError(IngestionError):
    """
    Raised when the file is readable but the resulting DataFrame contains no data.

    Raised by: DataLoader._validate_non_empty() and DataLoader._read_csv()
    Validation order: FOURTH (last) — occurs after the full I/O read.

    This exception signals a data quality failure at the ingestion boundary.
    The pipeline must halt rather than propagate an empty DataFrame downstream,
    which would cause silent failures in preprocessing and model training.

    Conditions:
        - DataFrame.empty is True (zero rows OR zero columns).
        - pandas raises EmptyDataError for zero-byte or header-only CSV files.

    Attributes:
        path: The file that was read successfully but yielded no data.
        num_rows: Row count of the empty DataFrame.
        num_columns: Column count of the empty DataFrame.

    Example:
        >>> raise EmptyDatasetError(
        ...     path="/data/raw/creditcard.csv",
        ...     num_rows=0,
        ...     num_columns=31,
        ... )
    """

    def __init__(self, path: str, num_rows: int, num_columns: int) -> None:
        self.path = path
        self.num_rows = num_rows
        self.num_columns = num_columns
        super().__init__(
            message=(
                f"Dataset is empty after loading '{path}'. "
                f"Shape: ({num_rows} rows x {num_columns} columns). "
                "Verify that the source file contains data."
            ),
            context={
                "path": path,
                "num_rows": num_rows,
                "num_columns": num_columns,
            },
        )


class DatasetReadError(IngestionError):
    """
    Raised when an I/O or parsing failure occurs while reading the dataset file.

    Raised by: DataLoader._read_csv() (and future format-specific readers).
    Validation order: THIRD — occurs during actual file I/O.

    Wraps low-level library exceptions (pandas ParserError, UnicodeDecodeError,
    OSError, PermissionError) into a domain exception, decoupling all callers
    from pandas internals. If the I/O engine is ever swapped, callers still
    catch DatasetReadError.

    The original exception is always chained via `raise ... from original_exc`
    so the full traceback is preserved for debugging.

    Attributes:
        path: The file that could not be read.
        original_error: The underlying exception that caused the read failure.

    Example:
        >>> try:
        ...     df = pd.read_csv(path)
        ... except Exception as exc:
        ...     raise DatasetReadError(path=str(path), original_error=exc) from exc
    """

    def __init__(self, path: str, original_error: Exception) -> None:
        self.path = path
        self.original_error = original_error
        super().__init__(
            message=(
                f"Failed to read dataset '{path}'. "
                f"Cause: {type(original_error).__name__}: {original_error}"
            ),
            context={
                "path": path,
                "error_type": type(original_error).__name__,
                "error_detail": str(original_error),
            },
        )
