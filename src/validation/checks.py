"""
Pure, stateless validation check functions for the Validation Layer.

Design rules (enforced for ALL functions in this module):
    1. Single responsibility  - each function validates exactly one concern.
    2. No side effects        - no logging, no file I/O, no global mutation.
    3. Always return          - never raise exceptions; wrap errors in a
                                FAILED ValidationResult so the validator
                                never crashes mid-run.
    4. Self-timed             - execution_time_ms is measured inside each
                                function and stored in the returned result.
    5. Deterministic          - same input always produces identical output.
    6. Uniform signature      - every check accepts (df, schema, thresholds)
                                so the validator can dispatch without
                                special-casing any individual check.

Status / Severity mapping (always correlated):
    PASSED  <-> INFO
    WARNING <-> WARNING
    FAILED  <-> ERROR

Default thresholds (overridable via the thresholds dict):
    missing_values_warning_pct   = 5.0   (% of rows in a column)
    missing_values_error_pct     = 20.0
    duplicate_rows_warning_pct   = 1.0   (% of total rows)
    duplicate_rows_error_pct     = 10.0
    class_imbalance_warning_pct  = 5.0   (minority class % of total)
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from src.validation.models import ValidationResult
from src.validation.schema import DataSchema


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_THRESHOLDS: dict[str, float] = {
    "missing_values_warning_pct": 5.0,
    "missing_values_error_pct": 20.0,
    "duplicate_rows_warning_pct": 1.0,
    "duplicate_rows_error_pct": 10.0,
    "class_imbalance_warning_pct": 5.0,
}

# Maps a pandas dtype string prefix/exact value to our broad category.
_NUMERIC_KINDS: frozenset[str] = frozenset({"i", "u", "f"})  # int, uint, float
# pandas 3.x: default string dtype renders as "str" (StringDtype), not "object".
_CATEGORICAL_DTYPES: frozenset[str] = frozenset({"object", "string", "category", "str"})
_BOOLEAN_DTYPES: frozenset[str] = frozenset({"bool", "boolean"})
# Pandas nullable numeric dtype name prefixes (Int8, UInt32, Float64, etc.)
_NULLABLE_NUMERIC_PREFIXES: tuple[str, ...] = ("Int", "UInt", "Float")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _threshold(thresholds: dict | None, key: str) -> float:
    """Returns a threshold value, falling back to the module default.

    Args:
        thresholds: Caller-supplied overrides, or None.
        key: The threshold key to look up.

    Returns:
        float: The effective threshold value.
    """
    if thresholds and key in thresholds:
        return float(thresholds[key])
    return _DEFAULT_THRESHOLDS[key]


def _dtype_to_category(dtype: Any) -> str:
    """Maps a pandas dtype to one of our broad category strings.

    Handles:
        - numpy dtypes (int8, float64, object, etc.)
        - pandas ExtensionDtype (StringDtype -> "str", BooleanDtype, Int64, etc.)
        - pandas 3.x default string dtype which renders as "str" not "object".

    Args:
        dtype: A numpy dtype or pandas ExtensionDtype object.

    Returns:
        str: One of "numeric", "categorical", "datetime", "boolean", "unknown".
    """
    # 1. numpy dtype kind check (most common path for numeric columns)
    if hasattr(dtype, "kind") and dtype.kind in _NUMERIC_KINDS:
        return "numeric"

    dtype_str = str(dtype)

    # 2. Exact membership check for common string forms
    if dtype_str in _CATEGORICAL_DTYPES:
        return "categorical"
    if dtype_str in _BOOLEAN_DTYPES:
        return "boolean"

    # 3. Prefix-based checks for parameterised dtypes
    if dtype_str.startswith(("int", "uint", "float")):
        return "numeric"
    if dtype_str.startswith(_NULLABLE_NUMERIC_PREFIXES):
        return "numeric"
    if "datetime" in dtype_str:
        return "datetime"
    if dtype_str.startswith("string"):
        return "categorical"

    return "unknown"


def _make_result(
    check_name: str,
    category: str,
    status: str,
    severity: str,
    message: str,
    details: dict,
    start: float,
) -> ValidationResult:
    """Constructs a ValidationResult with elapsed time already computed.

    Args:
        check_name: Name of the check function.
        category: Validation category (STRUCTURAL / QUALITY / TARGET).
        status: PASSED / WARNING / FAILED.
        severity: INFO / WARNING / ERROR.
        message: One-line human-readable summary.
        details: JSON-serialisable context dict.
        start: perf_counter value recorded at check start.

    Returns:
        ValidationResult: Fully populated result object.
    """
    elapsed_ms = round((time.perf_counter() - start) * 1_000, 3)
    return ValidationResult(
        check_name=check_name,
        category=category,
        status=status,
        severity=severity,
        message=message,
        details=details,
        execution_time_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------


def check_empty_dataframe(
    df: pd.DataFrame,
    schema: DataSchema,
    thresholds: dict | None = None,
) -> ValidationResult:
    """
    Pre-flight guard: verifies the DataFrame is non-empty.

    This is always the first check executed. If it fails, the validator
    raises SchemaValidationError immediately and no further checks run.

    Args:
        df: The DataFrame to inspect.
        schema: The DataSchema contract (unused here; present for uniform API).
        thresholds: Not used by this check.

    Returns:
        ValidationResult: PASSED/INFO if non-empty, FAILED/ERROR if empty.
    """
    _start = time.perf_counter()
    n_rows = len(df)
    n_cols = len(df.columns)

    if df.empty:
        return _make_result(
            check_name="check_empty_dataframe",
            category="STRUCTURAL",
            status="FAILED",
            severity="ERROR",
            message=(
                f"DataFrame is empty "
                f"({n_rows} rows x {n_cols} columns). "
                "Cannot validate an empty dataset."
            ),
            details={"num_rows": n_rows, "num_columns": n_cols},
            start=_start,
        )

    return _make_result(
        check_name="check_empty_dataframe",
        category="STRUCTURAL",
        status="PASSED",
        severity="INFO",
        message=f"DataFrame is non-empty: {n_rows:,} rows x {n_cols} columns.",
        details={"num_rows": n_rows, "num_columns": n_cols},
        start=_start,
    )


def check_schema(
    df: pd.DataFrame,
    schema: DataSchema,
    thresholds: dict | None = None,
) -> ValidationResult:
    """
    Merged schema check: verifies column names, required presence, column
    count bounds, optional column presence, duplicate names, and unexpected
    extra columns — all in one ValidationResult.

    Replaces the LLD's separate check_column_names() + check_required_columns()
    per implementation requirement Change 3.

    Severity logic:
        ERROR  - any required column is missing OR duplicate column names exist.
        WARNING - optional columns absent, unexpected extra columns, or column
                  count outside schema bounds.
        INFO   - all declared columns present, no extras, count within bounds.

    Args:
        df: The DataFrame to inspect.
        schema: The DataSchema contract to validate against.
        thresholds: Not used by this check.

    Returns:
        ValidationResult: Single result capturing all schema findings.
    """
    _start = time.perf_counter()

    df_cols: list[str] = list(df.columns)
    col_count: int = len(df_cols)
    required_set = schema.required_column_set
    optional_set = schema.optional_column_set
    known_set = required_set | optional_set

    # Duplicate column names in the DataFrame
    seen: set[str] = set()
    duplicate_cols: list[str] = []
    for col in df_cols:
        if col in seen:
            duplicate_cols.append(col)
        seen.add(col)

    # Missing required columns
    df_col_set = set(df_cols)
    missing_required: list[str] = [
        col for col in schema.required_columns if col not in df_col_set
    ]

    # Missing optional columns
    missing_optional: list[str] = [
        col for col in schema.optional_columns if col not in df_col_set
    ]

    # Unexpected / undeclared columns
    unexpected_cols: list[str] = [col for col in df_cols if col not in known_set]

    # Column count bounds
    count_within_bounds = col_count >= schema.min_columns
    if schema.max_columns is not None:
        count_within_bounds = count_within_bounds and col_count <= schema.max_columns

    details: dict = {
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "unexpected_columns": unexpected_cols,
        "duplicate_columns": duplicate_cols,
        "column_count": col_count,
        "min_columns": schema.min_columns,
        "max_columns": schema.max_columns,
        "count_within_bounds": count_within_bounds,
    }

    # Determine outcome
    if missing_required or duplicate_cols:
        parts: list[str] = []
        if missing_required:
            parts.append(f"{len(missing_required)} required column(s) missing")
        if duplicate_cols:
            parts.append(f"{len(duplicate_cols)} duplicate column name(s) found")
        msg = f"Schema validation FAILED: {'; '.join(parts)}."
        return _make_result("check_schema", "STRUCTURAL", "FAILED", "ERROR", msg, details, _start)

    if missing_optional or unexpected_cols or not count_within_bounds:
        parts = []
        if missing_optional:
            parts.append(f"{len(missing_optional)} optional column(s) absent")
        if unexpected_cols:
            parts.append(f"{len(unexpected_cols)} unexpected column(s) found")
        if not count_within_bounds:
            parts.append(
                f"column count {col_count} outside bounds "
                f"[{schema.min_columns}, {schema.max_columns or 'unbounded'}]"
            )
        msg = f"Schema validation WARNING: {'; '.join(parts)}."
        return _make_result("check_schema", "STRUCTURAL", "WARNING", "WARNING", msg, details, _start)

    return _make_result(
        "check_schema",
        "STRUCTURAL",
        "PASSED",
        "INFO",
        f"Schema validation passed: all {len(schema.required_columns)} required "
        f"column(s) present, column count {col_count} within bounds.",
        details,
        _start,
    )


def check_data_types(
    df: pd.DataFrame,
    schema: DataSchema,
    thresholds: dict | None = None,
) -> ValidationResult:
    """
    Verifies that columns declared in schema.expected_types have compatible
    broad dtype categories in the DataFrame.

    Columns not in expected_types are skipped. Columns missing from the
    DataFrame (already flagged by check_schema) are also skipped.

    Data type mismatches are returned as ValidationResult(severity="ERROR")
    per implementation requirement Change 7 — no DataTypeValidationError raised.

    Args:
        df: The DataFrame to inspect.
        schema: Contains expected_types mapping column -> category.
        thresholds: Not used by this check.

    Returns:
        ValidationResult: ERROR if any mismatches, INFO if all types match.
    """
    _start = time.perf_counter()

    mismatches: list[dict] = []

    for col, expected_cat in schema.expected_types.items():
        if col not in df.columns:
            continue  # missing columns are already reported by check_schema
        actual_cat = _dtype_to_category(df[col].dtype)
        if actual_cat != expected_cat:
            mismatches.append(
                {
                    "column": col,
                    "expected_category": expected_cat,
                    "actual_dtype": str(df[col].dtype),
                    "actual_category": actual_cat,
                }
            )

    details: dict = {
        "columns_checked": len(schema.expected_types),
        "mismatches": mismatches,
        "mismatch_count": len(mismatches),
    }

    if mismatches:
        col_names = [m["column"] for m in mismatches]
        return _make_result(
            "check_data_types",
            "STRUCTURAL",
            "FAILED",
            "ERROR",
            f"Data type mismatch in {len(mismatches)} column(s): {col_names}.",
            details,
            _start,
        )

    return _make_result(
        "check_data_types",
        "STRUCTURAL",
        "PASSED",
        "INFO",
        f"All {len(schema.expected_types)} declared column type(s) match expected categories.",
        details,
        _start,
    )


def check_empty_columns(
    df: pd.DataFrame,
    schema: DataSchema,
    thresholds: dict | None = None,
) -> ValidationResult:
    """
    Detects columns where every single value is null (entirely empty columns).

    An entirely-null required column is an ERROR. An entirely-null column not
    in the schema (undeclared) triggers a WARNING.

    Args:
        df: The DataFrame to inspect.
        schema: Used to classify empty columns as required or other.
        thresholds: Not used by this check.

    Returns:
        ValidationResult: ERROR / WARNING / INFO based on findings.
    """
    _start = time.perf_counter()

    all_null_cols: list[str] = [
        col for col in df.columns if df[col].isna().all()
    ]
    required_empty: list[str] = [c for c in all_null_cols if c in schema.required_column_set]
    other_empty: list[str] = [c for c in all_null_cols if c not in schema.required_column_set]

    details: dict = {
        "entirely_empty_columns": all_null_cols,
        "required_empty": required_empty,
        "other_empty": other_empty,
        "total_empty_count": len(all_null_cols),
    }

    if required_empty:
        return _make_result(
            "check_empty_columns",
            "STRUCTURAL",
            "FAILED",
            "ERROR",
            f"{len(required_empty)} required column(s) are entirely null: {required_empty}.",
            details,
            _start,
        )

    if other_empty:
        return _make_result(
            "check_empty_columns",
            "STRUCTURAL",
            "WARNING",
            "WARNING",
            f"{len(other_empty)} non-required column(s) are entirely null: {other_empty}.",
            details,
            _start,
        )

    return _make_result(
        "check_empty_columns",
        "STRUCTURAL",
        "PASSED",
        "INFO",
        "No entirely-null columns detected.",
        details,
        _start,
    )


def check_missing_values(
    df: pd.DataFrame,
    schema: DataSchema,
    thresholds: dict | None = None,
) -> ValidationResult:
    """
    Computes per-column missing value counts and percentages. Returns a single
    ValidationResult whose severity reflects the worst-affected column.

    Severity logic (worst column drives the result):
        ERROR   - max missing % >= missing_values_error_pct (default 20%)
        WARNING - max missing % >= missing_values_warning_pct (default 5%)
        INFO    - either no missing values, or all columns below warning threshold

    Args:
        df: The DataFrame to inspect.
        schema: Not directly used for logic here; present for uniform API.
        thresholds: Optional overrides for warning/error percentage thresholds.

    Returns:
        ValidationResult: Severity driven by the worst-affected column.
    """
    _start = time.perf_counter()

    warn_pct = _threshold(thresholds, "missing_values_warning_pct")
    error_pct = _threshold(thresholds, "missing_values_error_pct")
    total_rows = len(df)

    columns_with_missing: dict[str, dict] = {}
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        if n_missing > 0:
            pct = round((n_missing / total_rows) * 100, 4) if total_rows > 0 else 0.0
            columns_with_missing[col] = {"count": n_missing, "pct": pct}

    total_missing_cells = sum(v["count"] for v in columns_with_missing.values())
    max_pct = (
        max(v["pct"] for v in columns_with_missing.values())
        if columns_with_missing
        else 0.0
    )
    most_affected = (
        max(columns_with_missing, key=lambda c: columns_with_missing[c]["pct"])
        if columns_with_missing
        else None
    )

    details: dict = {
        "columns_with_missing": columns_with_missing,
        "total_missing_cells": total_missing_cells,
        "columns_affected": len(columns_with_missing),
        "max_missing_pct": max_pct,
        "most_affected_column": most_affected,
        "warning_threshold_pct": warn_pct,
        "error_threshold_pct": error_pct,
    }

    if not columns_with_missing:
        return _make_result(
            "check_missing_values",
            "QUALITY",
            "PASSED",
            "INFO",
            "No missing values detected across all columns.",
            details,
            _start,
        )

    if max_pct >= error_pct:
        return _make_result(
            "check_missing_values",
            "QUALITY",
            "FAILED",
            "ERROR",
            (
                f"Missing values exceed error threshold ({error_pct}%) "
                f"in {len(columns_with_missing)} column(s). "
                f"Worst: '{most_affected}' at {max_pct:.2f}%."
            ),
            details,
            _start,
        )

    if max_pct >= warn_pct:
        return _make_result(
            "check_missing_values",
            "QUALITY",
            "WARNING",
            "WARNING",
            (
                f"Missing values detected in {len(columns_with_missing)} column(s). "
                f"Worst: '{most_affected}' at {max_pct:.2f}% "
                f"(warning threshold: {warn_pct}%)."
            ),
            details,
            _start,
        )

    # Missing values present but below warning threshold
    return _make_result(
        "check_missing_values",
        "QUALITY",
        "PASSED",
        "INFO",
        (
            f"Minor missing values in {len(columns_with_missing)} column(s) "
            f"(max {max_pct:.2f}%, below warning threshold {warn_pct}%)."
        ),
        details,
        _start,
    )


def check_duplicate_rows(
    df: pd.DataFrame,
    schema: DataSchema,
    thresholds: dict | None = None,
) -> ValidationResult:
    """
    Counts duplicate rows across the full DataFrame.

    Severity logic:
        ERROR   - duplicate % >= duplicate_rows_error_pct (default 10%)
        WARNING - duplicate % >= duplicate_rows_warning_pct (default 1%)
        INFO    - zero duplicates, or below warning threshold

    Args:
        df: The DataFrame to inspect.
        schema: Not directly used; present for uniform API.
        thresholds: Optional overrides for warning/error percentage thresholds.

    Returns:
        ValidationResult: Severity driven by the duplicate percentage.
    """
    _start = time.perf_counter()

    warn_pct = _threshold(thresholds, "duplicate_rows_warning_pct")
    error_pct = _threshold(thresholds, "duplicate_rows_error_pct")
    total_rows = len(df)
    n_dups = int(df.duplicated().sum())
    dup_pct = round((n_dups / total_rows) * 100, 4) if total_rows > 0 else 0.0

    details: dict = {
        "duplicate_count": n_dups,
        "duplicate_pct": dup_pct,
        "total_rows": total_rows,
        "warning_threshold_pct": warn_pct,
        "error_threshold_pct": error_pct,
    }

    if n_dups == 0:
        return _make_result(
            "check_duplicate_rows",
            "QUALITY",
            "PASSED",
            "INFO",
            f"No duplicate rows detected in {total_rows:,} rows.",
            details,
            _start,
        )

    if dup_pct >= error_pct:
        return _make_result(
            "check_duplicate_rows",
            "QUALITY",
            "FAILED",
            "ERROR",
            (
                f"Duplicate rows exceed error threshold ({error_pct}%): "
                f"{n_dups:,} duplicates ({dup_pct:.2f}% of {total_rows:,} rows)."
            ),
            details,
            _start,
        )

    if dup_pct >= warn_pct:
        return _make_result(
            "check_duplicate_rows",
            "QUALITY",
            "WARNING",
            "WARNING",
            (
                f"Duplicate rows detected: {n_dups:,} ({dup_pct:.2f}%) "
                f"of {total_rows:,} rows."
            ),
            details,
            _start,
        )

    # Duplicates present but below warning threshold
    return _make_result(
        "check_duplicate_rows",
        "QUALITY",
        "PASSED",
        "INFO",
        (
            f"Minor duplicate rows: {n_dups:,} ({dup_pct:.2f}%), "
            f"below warning threshold ({warn_pct}%)."
        ),
        details,
        _start,
    )


def check_target_column(
    df: pd.DataFrame,
    schema: DataSchema,
    thresholds: dict | None = None,
) -> ValidationResult:
    """
    Validates the target/label column declared in schema.target_column.

    Checks:
        1. target_column is set in the schema (skip if None).
        2. target_column is present in df.columns.
        3. target_column is not entirely null.
        4. target_column has at least 2 unique non-null values.

    If the column is missing, status=FAILED so the validator can raise
    TargetValidationError immediately (fail-fast for target column).

    Args:
        df: The DataFrame to inspect.
        schema: Provides target_column name.
        thresholds: Not used by this check.

    Returns:
        ValidationResult: FAILED/ERROR if target is absent or all-null,
            INFO otherwise.
    """
    _start = time.perf_counter()

    if schema.target_column is None:
        return _make_result(
            "check_target_column",
            "TARGET",
            "PASSED",
            "INFO",
            "No target column configured in schema. Skipping target check.",
            {"target_column": None},
            _start,
        )

    target = schema.target_column

    if target not in df.columns:
        return _make_result(
            "check_target_column",
            "TARGET",
            "FAILED",
            "ERROR",
            f"Target column '{target}' is not present in the DataFrame.",
            {
                "target_column": target,
                "available_columns": list(df.columns),
            },
            _start,
        )

    null_count = int(df[target].isna().sum())
    total_rows = len(df)
    unique_values = df[target].dropna().unique().tolist()
    n_unique = len(unique_values)

    if null_count == total_rows:
        return _make_result(
            "check_target_column",
            "TARGET",
            "FAILED",
            "ERROR",
            f"Target column '{target}' is entirely null.",
            {
                "target_column": target,
                "null_count": null_count,
                "total_rows": total_rows,
                "unique_values": [],
            },
            _start,
        )

    if n_unique < 2:
        return _make_result(
            "check_target_column",
            "TARGET",
            "FAILED",
            "ERROR",
            (
                f"Target column '{target}' has only {n_unique} unique value(s): "
                f"{unique_values}. Classification requires at least 2 classes."
            ),
            {
                "target_column": target,
                "null_count": null_count,
                "total_rows": total_rows,
                "unique_values": [str(v) for v in unique_values],
                "unique_count": n_unique,
            },
            _start,
        )

    return _make_result(
        "check_target_column",
        "TARGET",
        "PASSED",
        "INFO",
        (
            f"Target column '{target}' is valid: "
            f"{null_count} null(s), {n_unique} unique class(es)."
        ),
        {
            "target_column": target,
            "null_count": null_count,
            "total_rows": total_rows,
            "unique_values": [str(v) for v in sorted(unique_values)],
            "unique_count": n_unique,
        },
        _start,
    )


def check_class_distribution(
    df: pd.DataFrame,
    schema: DataSchema,
    thresholds: dict | None = None,
) -> ValidationResult:
    """
    Computes value counts for the target column and detects class imbalance.

    A WARNING is issued when the minority class percentage falls below
    class_imbalance_warning_pct (default 5.0%). The check reports the
    imbalance — it never fixes it. Oversampling / undersampling belongs to
    the preprocessing layer.

    This check is skipped (PASSED/INFO) when:
        - schema.target_column is None.
        - The target column is not in df.columns (already caught by
          check_target_column; this check only runs after target passes).

    Args:
        df: The DataFrame to inspect.
        schema: Provides target_column name.
        thresholds: Optional override for class_imbalance_warning_pct.

    Returns:
        ValidationResult: WARNING if severely imbalanced, INFO otherwise.
    """
    _start = time.perf_counter()

    imbalance_warn_pct = _threshold(thresholds, "class_imbalance_warning_pct")

    if schema.target_column is None or schema.target_column not in df.columns:
        return _make_result(
            "check_class_distribution",
            "TARGET",
            "PASSED",
            "INFO",
            "Skipping class distribution check (target column unavailable).",
            {"target_column": schema.target_column},
            _start,
        )

    target = schema.target_column
    counts = df[target].value_counts(dropna=True)
    total = int(counts.sum())

    class_counts: dict[str, int] = {str(k): int(v) for k, v in counts.items()}
    class_pcts: dict[str, float] = {
        str(k): round((int(v) / total) * 100, 4) for k, v in counts.items()
    }

    minority_class = str(counts.idxmin())
    minority_count = int(counts.min())
    minority_pct = round((minority_count / total) * 100, 4)
    majority_class = str(counts.idxmax())
    majority_count = int(counts.max())
    imbalance_ratio = round(majority_count / minority_count, 2) if minority_count > 0 else float("inf")

    details: dict = {
        "target_column": target,
        "class_counts": class_counts,
        "class_percentages": class_pcts,
        "total_labelled_rows": total,
        "minority_class": minority_class,
        "minority_count": minority_count,
        "minority_pct": minority_pct,
        "majority_class": majority_class,
        "majority_count": majority_count,
        "imbalance_ratio": imbalance_ratio,
        "warning_threshold_pct": imbalance_warn_pct,
    }

    if minority_pct < imbalance_warn_pct:
        return _make_result(
            "check_class_distribution",
            "TARGET",
            "WARNING",
            "WARNING",
            (
                f"Class imbalance detected in '{target}': "
                f"minority class '{minority_class}' = {minority_pct:.4f}% "
                f"({minority_count:,} samples). "
                f"Imbalance ratio: {imbalance_ratio:.1f}:1. "
                f"Consider SMOTE, class weights, or threshold tuning."
            ),
            details,
            _start,
        )

    return _make_result(
        "check_class_distribution",
        "TARGET",
        "PASSED",
        "INFO",
        (
            f"Class distribution for '{target}' is within acceptable bounds "
            f"(minority class: {minority_pct:.4f}%)."
        ),
        details,
        _start,
    )
