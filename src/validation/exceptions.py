"""
Custom exception hierarchy for the Validation Layer.

Design:
    All exceptions inherit from ValidationError so callers can choose between:

    1. Broad handling — catch all validation failures:
           except ValidationError as exc:
               handle(exc)

    2. Fine-grained handling:
           except SchemaValidationError:
               ...
           except TargetValidationError:
               ...

    Only STRUCTURAL failures that make further validation meaningless are raised
    as exceptions.  Every other problem is captured in a ValidationResult object
    and included in the ValidationReport, so the caller receives a full picture.

Hierarchy:
    Exception
    └── ValidationError
        ├── SchemaValidationError
        ├── TargetValidationError
        └── ValidationFailureError
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.validation.models import ValidationReport


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class ValidationError(Exception):
    """
    Base exception for all validation-layer failures.

    Never raise this class directly. Use one of the concrete subclasses
    so callers can distinguish between failure types.

    Attributes:
        message: Human-readable description of the failure.
        context: Serialisable key-value pairs with diagnostic information.
            This dict is intentionally kept JSON-serialisable (no DataFrames,
            no pandas objects) so it can be logged as structured JSON or
            attached to MLflow run tags.
    """

    def __init__(self, message: str, context: dict | None = None) -> None:
        self.message = message
        self.context: dict = context or {}
        super().__init__(self._format())

    def _format(self) -> str:
        """Produce the string passed to Exception.__init__."""
        if not self.context:
            return self.message
        ctx = " | ".join(f"{k}={v!r}" for k, v in self.context.items())
        return f"{self.message} [Context: {ctx}]"

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"message={self.message!r}, context={self.context!r})"
        )


# ---------------------------------------------------------------------------
# Concrete exceptions
# ---------------------------------------------------------------------------


class SchemaValidationError(ValidationError):
    """
    Raised when the dataset's column structure is fundamentally incompatible
    with the DataSchema contract.

    Raised by: DatasetValidator.validate() during Stage 0 (pre-flight) and
    Stage 1 (schema) when:
        - The DataFrame is empty (zero rows or zero columns).
        - One or more required columns are absent from df.columns.

    These conditions make all subsequent checks meaningless — there is no
    point running quality or target checks on an empty or structurally
    broken dataset.

    Attributes:
        missing_columns: Required columns that were not found in the DataFrame.
            Empty list when the failure is an empty DataFrame.
        schema_name: Name of the DataSchema that was used.
    """

    def __init__(
        self,
        missing_columns: list[str],
        schema_name: str,
        detail: str = "",
    ) -> None:
        self.missing_columns = missing_columns
        self.schema_name = schema_name

        if missing_columns:
            msg = (
                f"Schema '{schema_name}' validation failed: "
                f"{len(missing_columns)} required column(s) missing: {missing_columns}."
            )
        else:
            msg = f"Schema '{schema_name}' validation failed."

        if detail:
            msg = f"{msg} {detail}"

        super().__init__(
            message=msg,
            context={
                "schema_name": schema_name,
                "missing_columns": missing_columns,
                "detail": detail,
            },
        )


class TargetValidationError(ValidationError):
    """
    Raised when the target/label column declared in DataSchema.target_column
    is absent from the DataFrame.

    Raised by: DatasetValidator.validate() during Stage 4 (target) when:
        - schema.target_column is not None AND
        - schema.target_column is not in df.columns.

    A missing target column makes supervised learning impossible.  Raising
    an exception here (rather than returning a ValidationResult) forces
    the pipeline to halt before any model sees the data.

    Attributes:
        target_column: The expected target column name from the schema.
        available_columns: Column names that DO exist in the DataFrame.
    """

    def __init__(
        self,
        target_column: str,
        available_columns: list[str],
    ) -> None:
        self.target_column = target_column
        self.available_columns = available_columns

        super().__init__(
            message=(
                f"Target column '{target_column}' is not present in the DataFrame. "
                f"Available columns: {available_columns}."
            ),
            context={
                "target_column": target_column,
                "available_columns": available_columns,
            },
        )


class ValidationFailureError(ValidationError):
    """
    Raised when the complete validation run finishes with one or more
    ERROR-severity results AND the DatasetValidator was constructed with
    ``raise_on_failure=True``.

    This exception is intentionally raised AFTER all checks have run so the
    caller receives a complete ValidationReport (not just the first error).

    This is the only validation exception that carries a full ValidationReport,
    making it suitable for:
        - FastAPI 422 handlers that need to serialise the report.
        - CI/CD pipelines that want to log the full report before halting.
        - MLflow runs that attach the report as an artifact before failing.

    Attributes:
        report: The complete ValidationReport from the failed validation run.
        error_count: Number of ERROR-severity findings.
    """

    def __init__(self, report: ValidationReport, error_count: int) -> None:
        self.report = report
        self.error_count = error_count

        super().__init__(
            message=(
                f"Validation failed for dataset '{report.dataset_name}' "
                f"with {error_count} error(s). "
                f"Inspect 'exc.report' for the full ValidationReport."
            ),
            context={
                "dataset_name": report.dataset_name,
                "error_count": error_count,
                "report_id": report.report_id,
            },
        )
