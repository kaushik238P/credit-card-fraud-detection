"""
Data models for the ingestion layer.

DatasetMetadata:
    Immutable record of factual information about a loaded dataset.
    Pure data container — no behavior, no I/O, no pandas operations.
    All fields use primitive or standard-library types to ensure
    JSON-serializability and compatibility with MLflow artifact tagging.

LoadedDataset:
    Standard pipeline exchange object. Bundles the raw DataFrame with its
    immutable metadata. This is the single type returned by DataLoader.load()
    and consumed by all downstream pipeline stages (preprocessing, training,
    monitoring, API endpoints).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd


# ---------------------------------------------------------------------------
# DatasetMetadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetMetadata:
    """
    Immutable record of factual information about a loaded dataset.

    This object is created exactly once, immediately after loading, and never
    mutated. It is safe to pass across threads, serialize to JSON, log as
    structured data, or attach to MLflow runs.

    Design notes:
        - ``frozen=True`` enforces immutability at the Python level; any
          attempt to assign to a field after construction raises FrozenInstanceError.
        - ``column_names`` is stored as ``tuple[str, ...]`` (not list) because
          a frozen dataclass that participates in hashing must contain only
          hashable types. Lists are mutable and unhashable.
        - ``file_path`` is stored as ``str`` (not Path) because pathlib.Path
          is not JSON-serializable by default, which would complicate MLflow
          tagging, REST API responses, and log aggregation.
        - ``load_timestamp`` is always UTC-aware. Timezone-naive datetimes
          are rejected by design to avoid ambiguity in distributed deployments.

    Attributes:
        file_name: Base filename including extension (e.g., 'creditcard.csv').
            Used in logs, dashboards, and API responses for human identification.

        file_path: Resolved absolute path as a string.
            Stored for reproducibility and audit traceability. Using the
            resolved absolute form avoids ambiguity from relative paths or
            symlinks.

        file_extension: Lowercase file extension (e.g., '.csv').
            Indicates which reader was dispatched. Useful for debugging and
            pipeline auditing when multiple formats are supported.

        file_size_bytes: On-disk file size in bytes.
            Enables capacity planning, monitoring alerts for unexpectedly
            small or large files, and data drift detection at the size level.

        num_rows: Row count of the loaded DataFrame.
            Used to validate expected dataset cardinality. Surfaced in
            pipeline dashboards and MLflow run metrics.

        num_columns: Column count of the loaded DataFrame.
            Detects schema drift at the shape level without inspecting
            column contents. A column count change between runs is an early
            warning of upstream schema changes.

        column_names: Ordered tuple of all column names.
            Allows downstream consumers to verify expected columns are present
            without re-reading the file. Stored as tuple for immutability.

        memory_usage_bytes: In-memory DataFrame footprint in bytes (deep copy).
            Drives decisions about chunked loading, streaming, or memory
            budget allocation in containerized environments.

        target_column: Optional name of the label/target column.
            Passed through from the pipeline configuration — this field is
            never inferred from the data. ``None`` when the dataset is
            unlabeled or the target is not yet specified.

        load_timestamp: UTC-aware datetime when loading completed.
            Enables reproducibility tracking, dataset versioning, and
            audit logs. Always UTC to avoid timezone ambiguity.

        dataset_hash: Optional SHA-256 hex digest of the raw file bytes.
            Provides content-addressable identity for the dataset:
            - Detects silent data corruption.
            - Enables MLflow dataset versioning.
            - Supports cache invalidation in batch pipelines.
            Defaults to None if hash computation is disabled.

        dataset_version: Optional semantic version tag (e.g., 'v1.3.0').
            Set by the pipeline orchestrator or data engineering team.
            Enables downstream consumers to assert they are processing
            the expected dataset version. Defaults to None.

        loaded_by: Optional identifier of the process or user that triggered
            the load (e.g., 'fraud-pipeline-v2', 'john.doe', 'ci-runner-42').
            Provides an audit trail for multi-user and multi-service
            deployments. Defaults to the FRAUD_INGESTION_LOADED_BY env var
            value, or None if unset.
    """

    # -----------------------------------------------------------------------
    # Core identity fields
    # -----------------------------------------------------------------------
    file_name: str
    file_path: str
    file_extension: str

    # -----------------------------------------------------------------------
    # Size and shape fields
    # -----------------------------------------------------------------------
    file_size_bytes: int
    num_rows: int
    num_columns: int
    column_names: tuple[str, ...]   # tuple: hashable + immutable
    memory_usage_bytes: int

    # -----------------------------------------------------------------------
    # Pipeline context fields
    # -----------------------------------------------------------------------
    target_column: str | None
    load_timestamp: datetime        # always UTC-aware

    # -----------------------------------------------------------------------
    # Optional enrichment fields (MLflow / versioning / audit trail)
    # -----------------------------------------------------------------------
    dataset_hash: str | None = None
    dataset_version: str | None = None
    loaded_by: str | None = None

    # -----------------------------------------------------------------------
    # Serialization helpers
    # -----------------------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Returns a JSON-serializable dictionary representation of the metadata.

        All types are reduced to JSON primitives (str, int, float, list, None)
        so the result can be passed directly to:
            - json.dumps()
            - mlflow.log_params() / mlflow.set_tags()
            - FastAPI JSONResponse
            - Structured log records

        Returns:
            dict: Key-value mapping of all metadata fields.
        """
        return {
            "file_name": self.file_name,
            "file_path": self.file_path,
            "file_extension": self.file_extension,
            "file_size_bytes": self.file_size_bytes,
            "num_rows": self.num_rows,
            "num_columns": self.num_columns,
            "column_names": list(self.column_names),
            "memory_usage_bytes": self.memory_usage_bytes,
            "target_column": self.target_column,
            "load_timestamp": self.load_timestamp.isoformat(),
            "dataset_hash": self.dataset_hash,
            "dataset_version": self.dataset_version,
            "loaded_by": self.loaded_by,
        }

    def __str__(self) -> str:
        return (
            f"DatasetMetadata("
            f"file='{self.file_name}', "
            f"shape=({self.num_rows}x{self.num_columns}), "
            f"size={self.file_size_bytes:,} bytes, "
            f"loaded_at={self.load_timestamp.isoformat()}"
            f")"
        )


# ---------------------------------------------------------------------------
# LoadedDataset
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadedDataset:
    """
    Standard pipeline exchange object returned by DataLoader.load().

    Bundles the raw, unprocessed DataFrame with its immutable metadata.
    Every downstream consumer (preprocessing, feature engineering, training,
    API endpoints, Streamlit UI) receives this single object and can access
    both the data and its provenance without additional lookups.

    Design notes:
        - ``data`` is excluded from equality comparisons and hashing because
          pandas DataFrames do not support reliable equality semantics:
          ``df1 == df2`` returns a DataFrame of booleans, not a single bool.
          Identity of a LoadedDataset is determined entirely by its metadata.
        - ``frozen=True`` prevents accidental reassignment of either field
          after construction, but does NOT prevent in-place mutation of the
          DataFrame itself. Downstream consumers must treat ``data`` as
          read-only and work on explicit copies when transformation is needed.
        - ``repr=False`` on ``data`` prevents accidental printing of large
          DataFrames when the object appears in logs or debugger output.

    Attributes:
        data: The loaded pandas DataFrame. Raw and unprocessed.
            Contains exactly what the source file contains, with no
            cleaning, encoding, or transformation applied.
        metadata: Immutable metadata record for this dataset load.
            Provides full provenance, shape, size, and hash information.

    Example:
        >>> loader = DataLoader()
        >>> result = loader.load("data/raw/creditcard.csv")
        >>> result.data.head()
        >>> result.metadata.num_rows
        284807
        >>> result.metadata.dataset_hash
        'a3f8c2...'
    """

    data: pd.DataFrame = field(repr=False, compare=False, hash=False)
    metadata: DatasetMetadata

    def __repr__(self) -> str:
        return (
            f"LoadedDataset("
            f"file='{self.metadata.file_name}', "
            f"shape=({self.metadata.num_rows}x{self.metadata.num_columns}), "
            f"memory={self.metadata.memory_usage_bytes:,} bytes, "
            f"hash={self.metadata.dataset_hash or 'N/A'!r}"
            f")"
        )

    def __str__(self) -> str:
        return repr(self)
