"""
DatasetValidator: the single public entry point of the Validation Layer.

Responsibilities:
    - Orchestrate the ordered validation pipeline.
    - Execute checks via _run_check() with exception safety.
    - Aggregate results into ValidationSummary via _aggregate_results().
    - Build the immutable ValidationReport via _build_report().
    - Apply fail-fast logic for structural impossibilities.
    - Emit structured log events throughout.
    - Return a ValidatedDataset to the caller.

What the validator intentionally does NOT do:
    - Format reports (that is report.py's job).
    - Perform any data cleaning or transformation.
    - Access files directly (it only reads from the LoadedDataset it receives).

Validation pipeline (in execution order):
    Stage 0  Pre-flight     check_empty_dataframe   ALWAYS + fail-fast
    Stage 1  Schema         check_schema            ALWAYS + fail-fast
    Stage 2  Structural     check_data_types        filtered by category
                            check_empty_columns
    Stage 3  Quality        check_missing_values    filtered by category
                            check_duplicate_rows
    Stage 4  Target         check_target_column     if target set + fail-fast
                            check_class_distribution

Fail-fast conditions (raise exception, not return ValidationResult):
    - DataFrame is None                -> ValidationError
    - DataFrame is empty               -> SchemaValidationError
    - Required columns missing         -> SchemaValidationError
    - Target column absent             -> TargetValidationError
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Final

import pandas as pd

from config.logging import get_logger
from src.ingestion.models import LoadedDataset
from src.validation.checks import (
    check_class_distribution,
    check_data_types,
    check_duplicate_rows,
    check_empty_columns,
    check_empty_dataframe,
    check_missing_values,
    check_schema,
    check_target_column,
)
from src.validation.exceptions import (
    SchemaValidationError,
    TargetValidationError,
    ValidationError,
    ValidationFailureError,
)
from src.validation.models import (
    ValidatedDataset,
    ValidationReport,
    ValidationResult,
    ValidationSummary,
)
from src.validation.schema import DataSchema

logger = get_logger(__name__)

# All valid category strings.
_VALID_CATEGORIES: Final[frozenset[str]] = frozenset({"STRUCTURAL", "QUALITY", "TARGET"})

# Optional checks (not fail-fast) mapped to their categories.
# Stage 2 — STRUCTURAL (after schema passes)
# Stage 3 — QUALITY
_OPTIONAL_CHECKS: Final[list[tuple[Callable, str]]] = [
    (check_data_types,    "STRUCTURAL"),
    (check_empty_columns, "STRUCTURAL"),
    (check_missing_values,  "QUALITY"),
    (check_duplicate_rows,  "QUALITY"),
]


class DatasetValidator:
    """
    Orchestrates the full validation workflow for a LoadedDataset.

    This is the single public entry point of the validation module. No caller
    should ever interact directly with checks.py or models.py.

    Configuration is read at construction time, making this class Docker-
    friendly and fully testable: instantiate with different parameters for
    different pipeline contexts (training, inference, CI, monitoring).

    Design patterns applied:
        Chain of Responsibility  - checks are executed in a fixed ordered list.
        Strategy                 - thresholds dict customises check sensitivity.
        Composite                - categories filter which check groups run.

    Example:
        >>> from src.validation import DatasetValidator, FRAUD_TRANSACTION_SCHEMA
        >>> from src.ingestion import DataLoader
        >>>
        >>> loaded = DataLoader().load("data/raw/archive.zip")
        >>> validator = DatasetValidator(schema=FRAUD_TRANSACTION_SCHEMA)
        >>> result = validator.validate(loaded)
        >>> result.passed
        True

    Args:
        schema: DataSchema contract to validate against. Required.
        categories: Set of category strings to execute. When None (default),
            all categories run. Pass {"STRUCTURAL"} to run only structure
            checks (useful for CI speed optimisation).
        raise_on_failure: When True, raises ValidationFailureError if the
            validation run completes with error_count > 0. When False
            (default), returns a ValidatedDataset with passed=False and lets
            the caller decide whether to proceed.
        thresholds: Override default severity thresholds for checks that use
            them (missing_values, duplicate_rows, class_distribution).
            Example: {"missing_values_error_pct": 10.0}.
    """

    def __init__(
        self,
        schema: DataSchema,
        categories: set[str] | None = None,
        raise_on_failure: bool = False,
        thresholds: dict | None = None,
    ) -> None:
        if not isinstance(schema, DataSchema):
            raise TypeError(
                f"schema must be a DataSchema instance, got {type(schema).__name__}."
            )

        invalid_cats = (categories or set()) - _VALID_CATEGORIES
        if invalid_cats:
            raise ValueError(
                f"Invalid categories: {invalid_cats}. "
                f"Valid values: {sorted(_VALID_CATEGORIES)}."
            )

        self._schema: DataSchema = schema
        self._active_categories: frozenset[str] = (
            frozenset(categories) if categories else frozenset(_VALID_CATEGORIES)
        )
        self._raise_on_failure: bool = raise_on_failure
        self._thresholds: dict = thresholds or {}

        logger.debug(
            "DatasetValidator initialised | schema=%s | categories=%s | raise_on_failure=%s",
            schema.name,
            sorted(self._active_categories),
            raise_on_failure,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, loaded_dataset: LoadedDataset) -> ValidatedDataset:
        """
        Executes the full validation pipeline against a LoadedDataset.

        This is the only public method. The signature is intentionally stable:
        adding new checks or categories never changes it.

        Fail-fast conditions (exception raised before report is built):
            - loaded_dataset.data is None    -> ValidationError
            - DataFrame.empty is True        -> SchemaValidationError
            - Required columns missing       -> SchemaValidationError
            - Target column absent           -> TargetValidationError

        Everything else is collected as ValidationResult objects and included
        in the final ValidationReport so the caller sees all issues at once.

        Args:
            loaded_dataset: The LoadedDataset returned by DataLoader.load().
                Must contain a non-None DataFrame.

        Returns:
            ValidatedDataset: Contains the original data, ingestion metadata,
                and the complete ValidationReport. Check .passed for the
                overall pass/fail signal.

        Raises:
            ValidationError: DataFrame is None.
            SchemaValidationError: DataFrame is empty or required columns missing.
            TargetValidationError: Target column is absent from the DataFrame.
            ValidationFailureError: Overall validation failed and
                raise_on_failure=True was set on this validator.
        """
        _global_start = time.perf_counter()
        all_results: list[ValidationResult] = []

        dataset_name = (
            loaded_dataset.metadata.file_name
            if loaded_dataset.metadata
            else "unknown"
        )

        logger.info(
            "Validation started | dataset='%s' | schema='%s %s' | "
            "shape=(%s rows x %s cols) | categories=%s",
            dataset_name,
            self._schema.name,
            self._schema.version,
            f"{loaded_dataset.metadata.num_rows:,}" if loaded_dataset.metadata else "?",
            loaded_dataset.metadata.num_columns if loaded_dataset.metadata else "?",
            sorted(self._active_categories),
        )

        # ── Guard: DataFrame must not be None ──────────────────────────
        df = loaded_dataset.data
        if df is None:
            raise ValidationError(
                message=(
                    f"Cannot validate dataset '{dataset_name}': "
                    "LoadedDataset.data is None."
                ),
                context={"dataset_name": dataset_name},
            )

        # ── Stage 0: Pre-flight — always run, fail-fast ─────────────────
        logger.debug("Stage 0: Pre-flight (check_empty_dataframe).")
        pre_result = self._run_check(check_empty_dataframe, df)
        all_results.append(pre_result)

        if pre_result.status == "FAILED":
            logger.error(
                "Pre-flight FAILED for '%s': %s", dataset_name, pre_result.message
            )
            raise SchemaValidationError(
                missing_columns=[],
                schema_name=self._schema.name,
                detail=pre_result.message,
            )

        logger.debug("Stage 0: Pre-flight passed.")

        # ── Stage 1: Schema — always run, fail-fast ─────────────────────
        logger.debug("Stage 1: Schema validation (check_schema).")
        schema_result = self._run_check(check_schema, df)
        all_results.append(schema_result)

        if schema_result.status == "FAILED":
            missing = schema_result.details.get("missing_required", [])
            logger.error(
                "Schema validation FAILED for '%s': missing=%s",
                dataset_name,
                missing,
            )
            raise SchemaValidationError(
                missing_columns=missing,
                schema_name=self._schema.name,
            )

        logger.debug("Stage 1: Schema passed (status=%s).", schema_result.status)

        # ── Stages 2-3: Optional checks filtered by category ─────────────
        optional_checks = self._resolve_checks()

        structural_checks = [fn for fn, cat in optional_checks if cat == "STRUCTURAL"]
        quality_checks = [fn for fn, cat in optional_checks if cat == "QUALITY"]

        if structural_checks:
            logger.debug(
                "Stage 2: Structural checks (%d check(s)).", len(structural_checks)
            )
            all_results.extend(self._run_stage(structural_checks, df))

        if quality_checks:
            logger.debug(
                "Stage 3: Quality checks (%d check(s)).", len(quality_checks)
            )
            all_results.extend(self._run_stage(quality_checks, df))

        # ── Stage 4: Target checks — fail-fast if target column missing ──
        if "TARGET" in self._active_categories and self._schema.target_column:
            logger.debug(
                "Stage 4: Target checks (target_column='%s').",
                self._schema.target_column,
            )

            target_result = self._run_check(check_target_column, df)
            all_results.append(target_result)

            if target_result.status == "FAILED":
                logger.error(
                    "Target validation FAILED for '%s': %s",
                    dataset_name,
                    target_result.message,
                )
                raise TargetValidationError(
                    target_column=self._schema.target_column,
                    available_columns=list(df.columns),
                )

            logger.debug("check_target_column passed.")

            dist_result = self._run_check(check_class_distribution, df)
            all_results.append(dist_result)

        # ── Aggregate + build report ─────────────────────────────────────
        duration_ms = (time.perf_counter() - _global_start) * 1_000
        summary = self._aggregate_results(all_results, duration_ms)
        report = self._build_report(all_results, summary, loaded_dataset)

        _status_str = "PASSED" if report.passed else "FAILED"
        logger.info(
            "Validation complete | dataset='%s' | status=%s | "
            "checks=%d | errors=%d | warnings=%d | duration=%.2fms",
            dataset_name,
            _status_str,
            summary.total_checks,
            summary.errors,
            summary.warnings,
            summary.duration_ms,
        )

        if self._raise_on_failure and not report.passed:
            raise ValidationFailureError(report=report, error_count=summary.errors)

        return ValidatedDataset(
            data=df,
            metadata=loaded_dataset.metadata,
            report=report,
            passed=report.passed,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_check(
        self,
        check_fn: Callable,
        df: pd.DataFrame,
    ) -> ValidationResult:
        """
        Calls a single check function with full exception safety.

        Any unexpected exception raised inside the check is caught here and
        returned as a FAILED ValidationResult so the validator never crashes
        mid-run. This is the safety net for buggy custom checks.

        Args:
            check_fn: A check function with signature
                (df, schema, thresholds) -> ValidationResult.
            df: The DataFrame to pass to the check.

        Returns:
            ValidationResult: The result from the check, or a synthesised
                FAILED result if the check raised an unexpected exception.
        """
        name = getattr(check_fn, "__name__", "unknown_check")
        logger.debug("Running check: '%s'.", name)

        try:
            result: ValidationResult = check_fn(df, self._schema, self._thresholds)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Check '%s' raised an unexpected error: %s: %s",
                name,
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            return ValidationResult(
                check_name=name,
                category="STRUCTURAL",
                status="FAILED",
                severity="ERROR",
                message=f"Check raised unexpected error: {type(exc).__name__}: {exc}",
                details={
                    "error_type": type(exc).__name__,
                    "error_detail": str(exc),
                },
                execution_time_ms=0.0,
            )

        # Log each check's outcome at the appropriate level.
        if result.severity == "ERROR":
            logger.error(
                "Check '%s' FAILED | %s", result.check_name, result.message
            )
        elif result.severity == "WARNING":
            logger.warning(
                "Check '%s' WARNING | %s", result.check_name, result.message
            )
        else:
            logger.debug(
                "Check '%s' PASSED | %s", result.check_name, result.message
            )

        return result

    def _run_stage(
        self,
        checks: list[Callable],
        df: pd.DataFrame,
    ) -> list[ValidationResult]:
        """
        Executes a list of check functions and collects all results.

        Unlike Stage 0 and Stage 1, stages run here are non-fail-fast:
        all checks execute even if earlier ones return FAILED. This ensures
        the ValidationReport contains a complete picture of all issues.

        Args:
            checks: Ordered list of check functions to execute.
            df: The DataFrame to pass to each check.

        Returns:
            list[ValidationResult]: One result per check, in execution order.
        """
        return [self._run_check(fn, df) for fn in checks]

    def _resolve_checks(self) -> list[tuple[Callable, str]]:
        """
        Returns the list of optional checks filtered by active categories.

        This is the single location that changes when new optional checks are
        added. The validator's validate() method does not change.

        Returns:
            list[tuple[Callable, str]]: (check_fn, category) pairs, in
                execution order, filtered to active categories only.
        """
        return [
            (fn, cat)
            for fn, cat in _OPTIONAL_CHECKS
            if cat in self._active_categories
        ]

    def _aggregate_results(
        self,
        results: list[ValidationResult],
        duration_ms: float,
    ) -> ValidationSummary:
        """
        Aggregates a list of ValidationResult objects into a ValidationSummary.

        Args:
            results: All results from the completed validation run.
            duration_ms: Total wall-clock time for the run.

        Returns:
            ValidationSummary: Aggregated totals with overall pass/fail.
        """
        total = len(results)
        passed = sum(1 for r in results if r.status == "PASSED")
        errors = sum(1 for r in results if r.severity == "ERROR")
        warnings = sum(1 for r in results if r.severity == "WARNING")

        return ValidationSummary(
            total_checks=total,
            passed=passed,
            errors=errors,
            warnings=warnings,
            duration_ms=round(duration_ms, 2),
            overall_passed=errors == 0,
        )

    def _build_report(
        self,
        results: list[ValidationResult],
        summary: ValidationSummary,
        loaded_dataset: LoadedDataset,
    ) -> ValidationReport:
        """
        Assembles an immutable ValidationReport from aggregated run data.

        Args:
            results: All ValidationResult objects in execution order.
            summary: Aggregated summary for this run.
            loaded_dataset: The original LoadedDataset for provenance data.

        Returns:
            ValidationReport: Complete, immutable report ready for export.
        """
        return ValidationReport(
            report_id=str(uuid.uuid4()),
            dataset_name=loaded_dataset.metadata.file_name,
            dataset_hash=loaded_dataset.metadata.dataset_hash,
            schema_name=self._schema.name,
            schema_version=self._schema.version,
            results=tuple(results),
            summary=summary,
            generated_at=datetime.now(tz=timezone.utc),
            passed=summary.overall_passed,
            validation_duration_ms=summary.duration_ms,
            metadata_snapshot=loaded_dataset.metadata.to_dict(),
        )
