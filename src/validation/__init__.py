"""
Validation Layer public API for the Credit Card Fraud Detection System.

Public surface area:
    DatasetValidator        -- sole public entry point; call .validate(loaded_dataset)
    ValidatedDataset        -- pipeline exchange object returned by validate()
    ValidationReport        -- complete audit record of one validation run
    ValidationSummary       -- aggregated totals (total_checks, passed, errors, warnings)
    ValidationResult        -- output of a single check function
    DataSchema              -- declarative schema contract
    FRAUD_TRANSACTION_SCHEMA -- pre-built schema for fraudTrain.csv / fraudTest.csv
    ReportExporter          -- formats ValidationReport to JSON or Markdown
    ValidationError         -- base exception for all validation failures
    SchemaValidationError   -- structural impossibility (empty DF / missing required cols)
    TargetValidationError   -- target column absent from DataFrame
    ValidationFailureError  -- validation completed with errors + raise_on_failure=True

Example:
    >>> from src.validation import DatasetValidator, FRAUD_TRANSACTION_SCHEMA
    >>> from src.ingestion import DataLoader
    >>>
    >>> loaded = DataLoader().load("data/raw/archive.zip")
    >>> validator = DatasetValidator(schema=FRAUD_TRANSACTION_SCHEMA)
    >>> result = validator.validate(loaded)
    >>> result.passed
    True
    >>> from src.validation import ReportExporter
    >>> ReportExporter(result.report).to_markdown("reports/validation/run.md")
"""

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
from src.validation.report import ReportExporter
from src.validation.schema import FRAUD_TRANSACTION_SCHEMA, DataSchema
from src.validation.validator import DatasetValidator

__all__ = [
    # Primary entry point
    "DatasetValidator",
    # Pipeline exchange objects
    "ValidatedDataset",
    "ValidationReport",
    "ValidationSummary",
    "ValidationResult",
    # Schema
    "DataSchema",
    "FRAUD_TRANSACTION_SCHEMA",
    # Reporting
    "ReportExporter",
    # Exception hierarchy
    "ValidationError",
    "SchemaValidationError",
    "TargetValidationError",
    "ValidationFailureError",
]
