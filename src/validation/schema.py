"""
Schema contract for the Validation Layer.

DataSchema is a pure declarative contract - it declares what a valid dataset
looks like but contains zero validation logic. All validation logic lives
exclusively in checks.py and is orchestrated by validator.py.

Design decisions:
    - ColumnSchema from the LLD is REMOVED per implementation requirements.
      Fields required_columns, optional_columns, and expected_types replace it.
    - expected_types maps column names to broad dtype categories
      ("numeric", "categorical", "datetime", "boolean") rather than specific
      pandas dtypes. This makes the schema engine-agnostic.
    - DataSchema is a frozen dataclass instantiated once and reused.
    - FRAUD_TRANSACTION_SCHEMA is a pre-built schema for fraudTrain.csv.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class DataSchema:
    """
    Immutable, declarative contract describing a valid dataset.

    This object is passed to DatasetValidator at construction time and is
    read (never written) by every check function in checks.py. Adding
    schema support for a new dataset means creating a new DataSchema instance
    - existing schemas and the validator are never modified (OCP).

    Attributes:
        name: Human-readable schema identifier used in reports and logs.
            Example: "fraud_transaction_v1".
        version: Semantic version string for schema evolution tracking.
            Example: "1.0.0".
        required_columns: Ordered tuple of column names that MUST be present
            in the DataFrame. Absence of any required column triggers a hard
            SchemaValidationError (fail-fast).
        optional_columns: Ordered tuple of column names that MAY be present.
            Absence of an optional column produces a WARNING, not a failure.
        expected_types: Mapping from column name to its broad dtype category.
            Supported values: "numeric", "categorical", "datetime", "boolean".
            Only columns listed here are type-checked. Columns absent from
            this dict are not type-checked.
        target_column: Name of the label/target column. None for unlabelled
            datasets. When set, DatasetValidator runs target-specific checks.
        min_columns: Minimum acceptable total column count.
        max_columns: Maximum acceptable total column count. None means unbounded.
        min_rows: Minimum acceptable row count.
        description: Human-readable description of this schema.
        created_at: UTC-aware datetime when this schema version was authored.
    """

    name: str
    version: str
    required_columns: tuple[str, ...]
    optional_columns: tuple[str, ...]
    expected_types: dict[str, str] = field(hash=False, compare=False)
    target_column: str | None
    min_columns: int
    max_columns: int | None
    min_rows: int
    description: str
    created_at: datetime

    @property
    def all_known_columns(self) -> tuple[str, ...]:
        """Union of required and optional columns in declaration order."""
        return self.required_columns + self.optional_columns

    @property
    def required_column_set(self) -> frozenset[str]:
        """frozenset of required column names for O(1) membership tests."""
        return frozenset(self.required_columns)

    @property
    def optional_column_set(self) -> frozenset[str]:
        """frozenset of optional column names for O(1) membership tests."""
        return frozenset(self.optional_columns)

    @property
    def numeric_columns(self) -> tuple[str, ...]:
        """Names of columns with expected_type == 'numeric'."""
        return tuple(
            col for col, cat in self.expected_types.items() if cat == "numeric"
        )

    @property
    def categorical_columns(self) -> tuple[str, ...]:
        """Names of columns with expected_type == 'categorical'."""
        return tuple(
            col for col, cat in self.expected_types.items() if cat == "categorical"
        )

    def to_dict(self) -> dict:
        """
        Returns a JSON-serialisable representation of the schema.

        Returns:
            dict: All fields serialised to JSON primitives.
        """
        return {
            "name": self.name,
            "version": self.version,
            "required_columns": list(self.required_columns),
            "optional_columns": list(self.optional_columns),
            "expected_types": dict(self.expected_types),
            "target_column": self.target_column,
            "min_columns": self.min_columns,
            "max_columns": self.max_columns,
            "min_rows": self.min_rows,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
        }

    def __str__(self) -> str:
        return (
            f"DataSchema("
            f"name={self.name!r}, "
            f"version={self.version!r}, "
            f"required={len(self.required_columns)}, "
            f"optional={len(self.optional_columns)}, "
            f"target={self.target_column!r})"
        )


# ---------------------------------------------------------------------------
# FraudTransactionSchema
# ---------------------------------------------------------------------------

# Column definitions for fraudTrain.csv / fraudTest.csv (23 columns).
# Confirmed from live ingestion: loader.load("data/raw/archive.zip").
_FRAUD_REQUIRED_COLUMNS: tuple[str, ...] = (
    "Unnamed: 0",
    "trans_date_trans_time",
    "cc_num",
    "merchant",
    "category",
    "amt",
    "first",
    "last",
    "gender",
    "street",
    "city",
    "state",
    "zip",
    "lat",
    "long",
    "city_pop",
    "job",
    "dob",
    "trans_num",
    "unix_time",
    "merch_lat",
    "merch_long",
    "is_fraud",
)

_FRAUD_EXPECTED_TYPES: dict[str, str] = {
    "Unnamed: 0":            "numeric",
    "trans_date_trans_time": "categorical",
    "cc_num":                "numeric",
    "merchant":              "categorical",
    "category":              "categorical",
    "amt":                   "numeric",
    "first":                 "categorical",
    "last":                  "categorical",
    "gender":                "categorical",
    "street":                "categorical",
    "city":                  "categorical",
    "state":                 "categorical",
    "zip":                   "numeric",
    "lat":                   "numeric",
    "long":                  "numeric",
    "city_pop":              "numeric",
    "job":                   "categorical",
    "dob":                   "categorical",
    "trans_num":             "categorical",
    "unix_time":             "numeric",
    "merch_lat":             "numeric",
    "merch_long":            "numeric",
    "is_fraud":              "numeric",
}

FRAUD_TRANSACTION_SCHEMA: DataSchema = DataSchema(
    name="fraud_transaction_v1",
    version="1.0.0",
    required_columns=_FRAUD_REQUIRED_COLUMNS,
    optional_columns=(),
    expected_types=_FRAUD_EXPECTED_TYPES,
    target_column="is_fraud",
    min_columns=23,
    max_columns=23,
    min_rows=1,
    description=(
        "Schema for the IEEE-CIS / Kaggle Credit Card Fraud Detection dataset. "
        "Covers fraudTrain.csv (1,296,675 rows x 23 columns). "
        "Target column: is_fraud (0=legitimate, 1=fraud). "
        "For inference without labels, use a separate schema with "
        "is_fraud in optional_columns and target_column=None."
    ),
    created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
)
