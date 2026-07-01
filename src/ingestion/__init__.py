"""
Ingestion layer package for the Credit Card Fraud Detection System.

Public surface area:
    DataLoader      — primary entry point; call DataLoader().load(path).
    LoadedDataset   — pipeline exchange object returned by DataLoader.
    DatasetMetadata — immutable metadata record bundled with every load.
    IngestionError  — base exception; catch this to handle all ingestion failures.

All other symbols (private readers, helper functions) are implementation details
and are NOT part of the public API.

Example:
    >>> from src.ingestion import DataLoader
    >>> result = DataLoader().load("data/raw/creditcard.csv")
    >>> print(result.metadata.num_rows)
    284807
"""

from src.ingestion.exceptions import (
    DatasetNotFoundError,
    DatasetReadError,
    EmptyDatasetError,
    IngestionError,
    UnsupportedFileFormatError,
)
from src.ingestion.loader import DataLoader
from src.ingestion.models import DatasetMetadata, LoadedDataset

__all__ = [
    # Primary entry point
    "DataLoader",
    # Pipeline exchange objects
    "LoadedDataset",
    "DatasetMetadata",
    # Exception hierarchy
    "IngestionError",
    "DatasetNotFoundError",
    "UnsupportedFileFormatError",
    "EmptyDatasetError",
    "DatasetReadError",
]
