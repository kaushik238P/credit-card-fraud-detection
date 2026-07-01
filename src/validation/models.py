"""
Immutable data models for the Validation Layer.

All models are frozen dataclasses — pure data containers with no business
logic, no I/O, and no pandas operations.  Every model exposes a to_dict()
method that returns a JSON-serialisable dictionary so results can be logged
as structured data, attached to MLflow runs, or served via FastAPI.

Model hierarchy:
    ValidationResult      — output of a single check function
    ValidationSummary     — aggregated totals across all check results
    ValidationReport      — complete audit record of one validation run
    ValidatedDataset      — pipeline exchange object returned by DatasetValidator

Architectural decisions:
    - ValidationIssue from the LLD is REMOVED per implementation requirements.
      Each ValidationResult carries a details dict directly, which is simpler
      and avoids an unnecessary level of nesting.
    - dict fields (details, metadata_snapshot) use field(hash=False,
      compare=False) so frozen dataclasses remain hashable.  Callers must
      treat these dicts as read-only; the frozen constraint prevents field
      reassignment but not in-place dict mutation.
    - pd.DataFrame fields use repr=False, compare=False, hash=False to avoid
      pandas equality-semantics issues in frozen dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from src.ingestion.models import DatasetMetadata


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationResult:
    """
    The standardised output of every check function in checks.py.

    Design rationale for a uniform result type:
        - The validator aggregates results with a single loop — no special-
          casing per check.
        - New checks can be added without modifying the aggregation logic.
        - Unit tests for checks are trivial: assert on ValidationResult fields.
        - Downstream consumers (FastAPI, monitoring, report generator) always
          receive the same structure regardless of which checks ran.

    Attributes:
        check_name: Identifier of the function that produced this result.
            Matches the Python function name (e.g., ``"check_missing_values"``).
            Used for filtering and reporting.

        category: Which validation category this check belongs to.
            One of ``"STRUCTURAL"``, ``"QUALITY"``, ``"TARGET"``.
            Enables category-level aggregation and selective execution.

        status: Outcome of the check.
            ``"PASSED"`` — no issues found.
            ``"WARNING"`` — potential concern that does not block the pipeline.
            ``"FAILED"`` — definitive problem; blocks pipeline when errors > 0.

        severity: Seriousness of the finding.
            ``"INFO"`` — informational only (status=PASSED).
            ``"WARNING"`` — concern worth investigating (status=WARNING).
            ``"ERROR"`` — definitive problem (status=FAILED).
            Status and severity are always correlated:
            PASSED↔INFO, WARNING↔WARNING, FAILED↔ERROR.

        message: Human-readable one-line summary of the check outcome.
            Appears directly in the report and in log output.

        details: Machine-readable key-value context for this result.
            Always JSON-serialisable.  Examples:
            ``{"missing_pct": 18.2, "column": "merchant"}``
            ``{"missing_required": ["cc_num"], "unexpected_columns": []}``
            Empty dict when the check passes cleanly.

        execution_time_ms: Wall-clock time to execute this check in milliseconds.
            Measured inside the check function.  Used for performance monitoring
            and bottleneck detection in CI/CD.
    """

    check_name: str
    category: str
    status: str
    severity: str
    message: str
    details: dict = field(default_factory=dict, hash=False, compare=False)
    execution_time_ms: float = 0.0

    def to_dict(self) -> dict:
        """
        Returns a JSON-serialisable dictionary representation.

        Returns:
            dict: All fields as JSON primitives.
        """
        return {
            "check_name": self.check_name,
            "category": self.category,
            "status": self.status,
            "severity": self.severity,
            "message": self.message,
            "details": self.details,
            "execution_time_ms": self.execution_time_ms,
        }

    def __str__(self) -> str:
        return (
            f"ValidationResult("
            f"check={self.check_name!r}, "
            f"status={self.status}, "
            f"severity={self.severity}, "
            f"msg={self.message!r})"
        )


# ---------------------------------------------------------------------------
# ValidationSummary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationSummary:
    """
    Aggregated totals across all ValidationResult objects for one run.

    Provides a quick-read signal without requiring iteration over all results.
    The five fields match the simplified requirements exactly — no extra counters.

    Attributes:
        total_checks: Total number of checks executed in this validation run.

        passed: Count of checks that returned status ``"PASSED"``.

        errors: Count of checks that returned severity ``"ERROR"``
            (= count with status ``"FAILED"``).
            The pipeline blocks when ``errors > 0``.

        warnings: Count of checks that returned severity ``"WARNING"``
            (= count with status ``"WARNING"``).

        duration_ms: Total wall-clock time for the complete validation run
            in milliseconds.  Includes check execution time and validator
            orchestration overhead.

        overall_passed: ``True`` if and only if ``errors == 0``.
            This is the single boolean gate that all downstream consumers check.
            Mirrors ``ValidationReport.passed`` for convenience.
    """

    total_checks: int
    passed: int
    errors: int
    warnings: int
    duration_ms: float
    overall_passed: bool

    def to_dict(self) -> dict:
        """
        Returns a JSON-serialisable dictionary representation.

        Returns:
            dict: All fields as JSON primitives.
        """
        return {
            "total_checks": self.total_checks,
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
            "duration_ms": round(self.duration_ms, 2),
            "overall_passed": self.overall_passed,
        }

    def __str__(self) -> str:
        status = "PASSED" if self.overall_passed else "FAILED"
        return (
            f"ValidationSummary("
            f"status={status}, "
            f"total={self.total_checks}, "
            f"passed={self.passed}, "
            f"warnings={self.warnings}, "
            f"errors={self.errors}, "
            f"duration={self.duration_ms:.1f}ms)"
        )


# ---------------------------------------------------------------------------
# ValidationReport
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationReport:
    """
    Complete, immutable audit record of a single validation run.

    This is the permanent output of ``DatasetValidator.validate()``.
    It is designed to be:
        - Serialised to JSON and stored as an MLflow artifact.
        - Formatted to Markdown and saved to the reports/ directory.
        - Returned in FastAPI responses as a structured 422 body.
        - Attached to ``ValidationFailureError`` for CI/CD pipeline inspection.

    Attributes:
        report_id: UUID identifying this report instance uniquely.
            Enables deduplication and cross-pipeline traceability.

        dataset_name: ``file_name`` from DatasetMetadata (e.g., ``"archive.zip"``).
            Human-readable identifier for the dataset that was validated.

        dataset_hash: SHA-256 content hash from DatasetMetadata.
            Content-addressable link between ingestion and validation records.
            ``None`` if hash computation was disabled during ingestion.

        schema_name: Name of the DataSchema used (e.g., ``"fraud_transaction_v1"``).
            Enables schema-version tracking in audit logs.

        schema_version: Version string of the DataSchema used.

        results: All individual check results in execution order.
            The complete, ordered audit trail of the validation run.

        summary: Aggregated totals and overall pass/fail signal.

        generated_at: UTC-aware datetime when the report was generated.

        passed: ``True`` if ``summary.errors == 0``.
            Top-level convenience field — mirrors ``summary.overall_passed``.

        validation_duration_ms: Total wall-clock time for the run in ms.

        metadata_snapshot: Serialised ``DatasetMetadata.to_dict()`` snapshot
            at validation time.  Preserves complete ingestion provenance inside
            the report so the report is self-contained.
    """

    report_id: str
    dataset_name: str
    dataset_hash: str | None
    schema_name: str
    schema_version: str
    results: tuple[ValidationResult, ...]
    summary: ValidationSummary
    generated_at: datetime
    passed: bool
    validation_duration_ms: float
    metadata_snapshot: dict = field(hash=False, compare=False)

    def to_dict(self) -> dict:
        """
        Returns a fully JSON-serialisable dictionary.

        All nested objects are recursively serialised via their own to_dict()
        methods.  datetime values are converted to ISO 8601 strings.

        Returns:
            dict: Complete, JSON-serialisable representation of the report.
        """
        return {
            "report_id": self.report_id,
            "dataset_name": self.dataset_name,
            "dataset_hash": self.dataset_hash,
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "results": [r.to_dict() for r in self.results],
            "summary": self.summary.to_dict(),
            "generated_at": self.generated_at.isoformat(),
            "passed": self.passed,
            "validation_duration_ms": round(self.validation_duration_ms, 2),
            "metadata_snapshot": self.metadata_snapshot,
        }

    def errors(self) -> list[ValidationResult]:
        """Returns only the ERROR-severity results."""
        return [r for r in self.results if r.severity == "ERROR"]

    def warnings(self) -> list[ValidationResult]:
        """Returns only the WARNING-severity results."""
        return [r for r in self.results if r.severity == "WARNING"]

    def __str__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"ValidationReport("
            f"id={self.report_id[:8]}..., "
            f"dataset={self.dataset_name!r}, "
            f"status={status}, "
            f"checks={len(self.results)}, "
            f"errors={self.summary.errors}, "
            f"warnings={self.summary.warnings})"
        )


# ---------------------------------------------------------------------------
# ValidatedDataset
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidatedDataset:
    """
    Pipeline exchange object returned by DatasetValidator.validate().

    Bundles the original, unmodified DataFrame with its ingestion metadata and
    the complete ValidationReport.  Every downstream stage (preprocessing,
    feature engineering, training, FastAPI, Streamlit) receives this single
    object.

    Design notes:
        - ``data`` carries ``compare=False, hash=False`` because pandas
          DataFrames do not support reliable equality semantics.
        - ``frozen=True`` prevents accidental field reassignment.  It does NOT
          prevent in-place mutation of the DataFrame itself — downstream
          consumers must work on explicit copies.
        - The validation layer NEVER modifies ``data``.  The DataFrame here is
          byte-for-byte identical to ``LoadedDataset.data``.

    Attributes:
        data: The original, unmodified pandas DataFrame.

        metadata: Ingestion-layer metadata.  Preserved for full provenance.

        report: Complete ValidationReport produced by DatasetValidator.

        passed: ``True`` if ``report.passed`` is ``True`` (no ERROR-severity
            findings).  Top-level convenience field so callers avoid
            ``result.report.summary.overall_passed``.
    """

    data: pd.DataFrame = field(repr=False, compare=False, hash=False)
    metadata: DatasetMetadata
    report: ValidationReport
    passed: bool

    def to_dict(self) -> dict:
        """
        Returns a JSON-serialisable dictionary excluding the DataFrame.

        The DataFrame is excluded because it is neither JSON-serialisable
        nor a suitable artifact for logging.  Use ``data`` directly for
        downstream processing.

        Returns:
            dict: metadata + report serialised, plus the passed flag.
        """
        return {
            "passed": self.passed,
            "metadata": self.metadata.to_dict(),
            "report": self.report.to_dict(),
        }

    def __repr__(self) -> str:
        return (
            f"ValidatedDataset("
            f"dataset={self.metadata.file_name!r}, "
            f"shape=({self.metadata.num_rows}x{self.metadata.num_columns}), "
            f"passed={self.passed}, "
            f"errors={self.report.summary.errors}, "
            f"warnings={self.report.summary.warnings})"
        )

    def __str__(self) -> str:
        return repr(self)
