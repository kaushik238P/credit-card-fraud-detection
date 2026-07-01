"""
DataLoader: Primary entry point for the data ingestion layer.

Responsibilities (and ONLY these):
    1. Resolve and normalize the file path.
    2. Assert the file exists on disk.
    3. Assert the file extension is supported.
    4. Delegate reading to the correct private reader method.
    5. Assert the resulting DataFrame is non-empty.
    6. Compute the SHA-256 content hash of the raw file.
    7. Collect and return a DatasetMetadata record.
    8. Emit structured log events throughout.
    9. Return a LoadedDataset containing (data, metadata).

What DataLoader intentionally does NOT do:
    - Data cleaning or imputation
    - Missing value handling
    - Outlier removal
    - Feature engineering or scaling
    - Encoding
    - Schema validation
    - EDA or profiling
    - Model training

Extensibility:
    Adding support for a new file format (Parquet, Feather, Excel, SQL,
    cloud storage) requires only two steps:
        1. Implement a private _read_<format>() method.
        2. Register it in _READER_DISPATCH with the corresponding extension key.

    The public DataLoader.load() interface is never changed.
"""

from __future__ import annotations

import hashlib
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from config.logging import get_logger
from config.settings import settings
from src.ingestion.exceptions import (
    DatasetNotFoundError,
    DatasetReadError,
    EmptyDatasetError,
    IngestionError,
    UnsupportedFileFormatError,
)
from src.ingestion.models import DatasetMetadata, LoadedDataset

# Module-level named logger — integrates with the centralized logging config.
# Name follows Python convention: 'src.ingestion.loader'
logger = get_logger(__name__)


class DataLoader:
    """
    Orchestrates the full dataset loading workflow.

    This is the single public entry point for the ingestion layer. All other
    classes, methods, and modules in this package are implementation details.

    Configuration is read from ``config.settings`` (not from the constructor)
    so the class is Docker-friendly and requires no changes to work in
    different deployment environments — configuration is injected via
    environment variables.

    Extension pattern:
        To add a new file format, add a private ``_read_<format>()`` method
        and register it in ``_READER_DISPATCH``. No other change is needed.

    Example:
        >>> loader = DataLoader()
        >>> result = loader.load("data/raw/creditcard.csv")
        >>> print(result.metadata.num_rows)
        284807

    Attributes:
        _READER_DISPATCH: Class-level registry mapping lowercase file
            extensions to the names of private reader methods. This is the
            single location that changes when new formats are added.
    """

    # -----------------------------------------------------------------------
    # Reader dispatch registry
    # -----------------------------------------------------------------------
    # Maps a lowercase file extension string to the name of the private method
    # that handles reading that format. getattr(self, method_name) is called
    # at runtime to dispatch to the correct reader without any if/elif chains.
    #
    # To add Parquet support:
    #   1. Implement _read_parquet(self, path: Path) -> pd.DataFrame
    #   2. Add ".parquet": "_read_parquet" here.
    #   That's it. DataLoader.load() does not change.
    _READER_DISPATCH: dict[str, str] = {
        ".csv": "_read_csv",
        ".zip": "_read_zip",
        # ".parquet":  "_read_parquet",   # future
        # ".feather":  "_read_feather",   # future
        # ".xlsx":     "_read_excel",     # future
        # ".xls":      "_read_excel",     # future
    }

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def load(self, path: str | Path) -> LoadedDataset:
        """
        Loads a dataset from the given path and returns a LoadedDataset.

        This method is the single public interface of the ingestion layer.
        It will never change signature regardless of how many new file
        formats are added internally.

        Validation order (fail-fast, cheapest first):
            1. File existence  -> DatasetNotFoundError
            2. Extension check -> UnsupportedFileFormatError
            3. File I/O read   -> DatasetReadError
            4. Empty check     -> EmptyDatasetError

        Args:
            path: Path to the dataset file. Accepts a string or a
                ``pathlib.Path`` object. Relative paths are resolved to
                absolute paths before any filesystem operation.

        Returns:
            LoadedDataset: A frozen object containing:
                - ``data``: The raw, unprocessed ``pd.DataFrame``.
                - ``metadata``: An immutable ``DatasetMetadata`` record.

        Raises:
            DatasetNotFoundError: The path does not exist or is not a file.
            UnsupportedFileFormatError: The extension is not in the registry.
            DatasetReadError: An I/O or parsing failure occurred.
            EmptyDatasetError: The file parsed successfully but is empty.
            IngestionError: Any other unexpected ingestion-level failure.

        Example:
            >>> loader = DataLoader()
            >>> result = loader.load("data/raw/creditcard.csv")
            >>> df = result.data
            >>> meta = result.metadata
            >>> meta.num_rows, meta.num_columns
            (284807, 31)
        """
        start_time: float = time.perf_counter()

        # Step 0 — Normalize path to an absolute pathlib.Path.
        resolved_path: Path = Path(path).resolve()

        logger.debug(
            "Ingestion requested.",
            extra={"raw_path": str(path), "resolved_path": str(resolved_path)},
        )

        try:
            # Step 1 — Validate path exists and is a file.
            self._validate_path(resolved_path)

            # Step 2 — Validate the extension is supported.
            extension: str = self._validate_extension(resolved_path)

            # Emit INFO now that we know the file is reachable and supported.
            file_size_bytes: int = resolved_path.stat().st_size
            logger.info(
                "Loading dataset: '%s' (%s bytes).",
                resolved_path.name,
                f"{file_size_bytes:,}",
                extra={
                    "file_name": resolved_path.name,
                    "file_path": str(resolved_path),
                    "file_extension": extension,
                    "file_size_bytes": file_size_bytes,
                },
            )

            # Step 3 — Dispatch to the correct private reader and read data.
            reader: Callable[[Path], pd.DataFrame] = self._get_reader(extension)
            dataframe: pd.DataFrame = reader(resolved_path)

            # Step 4 — Validate the loaded DataFrame is non-empty.
            self._validate_non_empty(dataframe, resolved_path)

            logger.info(
                "Dataset read: %d rows x %d columns.",
                len(dataframe),
                len(dataframe.columns),
                extra={
                    "num_rows": len(dataframe),
                    "num_columns": len(dataframe.columns),
                },
            )

            # Step 5 — Compute content hash for MLflow / versioning.
            dataset_hash: str = self._compute_dataset_hash(resolved_path)
            logger.debug(
                "Dataset hash computed.",
                extra={"sha256_prefix": dataset_hash[:16] + "..."},
            )

            # Step 6 — Build the immutable metadata record.
            metadata: DatasetMetadata = self._build_metadata(
                path=resolved_path,
                df=dataframe,
                dataset_hash=dataset_hash,
            )

            # Step 7 — Log success with timing.
            elapsed_ms: float = (time.perf_counter() - start_time) * 1_000
            logger.info(
                "Ingestion complete: '%s' loaded in %.2f ms | "
                "Shape: (%dx%d) | Memory: %s bytes.",
                resolved_path.name,
                elapsed_ms,
                metadata.num_rows,
                metadata.num_columns,
                f"{metadata.memory_usage_bytes:,}",
                extra={
                    "file_name": resolved_path.name,
                    "num_rows": metadata.num_rows,
                    "num_columns": metadata.num_columns,
                    "memory_usage_bytes": metadata.memory_usage_bytes,
                    "elapsed_ms": round(elapsed_ms, 2),
                    "dataset_hash": dataset_hash,
                    "status": "success",
                },
            )

            return LoadedDataset(data=dataframe, metadata=metadata)

        except IngestionError:
            # Log and re-raise domain exceptions without wrapping.
            elapsed_ms = (time.perf_counter() - start_time) * 1_000
            logger.error(
                "Ingestion failed for '%s' after %.2f ms.",
                resolved_path.name,
                elapsed_ms,
                extra={
                    "file_path": str(resolved_path),
                    "elapsed_ms": round(elapsed_ms, 2),
                    "status": "failure",
                },
                exc_info=True,
            )
            raise

        except Exception as unexpected:
            # Wrap truly unexpected exceptions in IngestionError so callers
            # are never exposed to internal implementation details.
            elapsed_ms = (time.perf_counter() - start_time) * 1_000
            logger.critical(
                "Unexpected error during ingestion of '%s': %s",
                resolved_path.name,
                unexpected,
                extra={
                    "file_path": str(resolved_path),
                    "elapsed_ms": round(elapsed_ms, 2),
                    "status": "failure",
                    "error_type": type(unexpected).__name__,
                },
                exc_info=True,
            )
            raise IngestionError(
                message=(
                    f"Unexpected error during ingestion of '{resolved_path.name}': "
                    f"{type(unexpected).__name__}: {unexpected}"
                ),
                context={"error_type": type(unexpected).__name__},
            ) from unexpected

    # -----------------------------------------------------------------------
    # Private validation methods
    # -----------------------------------------------------------------------

    def _validate_path(self, path: Path) -> None:
        """
        Asserts the given path points to an existing, readable file.

        Validation order: FIRST (cheapest — single stat syscall).

        Args:
            path: Resolved absolute path to validate.

        Raises:
            DatasetNotFoundError: If the path does not exist or is a directory.
        """
        logger.debug("Validating path: '%s'.", path)

        if not path.exists():
            raise DatasetNotFoundError(
                path=str(path),
                detail="No such file or directory.",
            )

        if not path.is_file():
            raise DatasetNotFoundError(
                path=str(path),
                detail="Path exists but points to a directory, not a file.",
            )

        logger.debug("Path validated: '%s' exists and is a file.", path.name)

    def _validate_extension(self, path: Path) -> str:
        """
        Asserts the file extension is registered in the reader dispatch table.

        Validation order: SECOND (dictionary lookup — no I/O).

        Args:
            path: Resolved absolute path whose extension is validated.

        Returns:
            str: The lowercase, dot-prefixed extension (e.g., '.csv').

        Raises:
            UnsupportedFileFormatError: If the extension is absent from the registry.
        """
        extension: str = path.suffix.lower()

        logger.debug(
            "Validating extension: '%s' for file '%s'.",
            extension,
            path.name,
        )

        supported: list[str] = list(self._READER_DISPATCH.keys())

        if extension not in self._READER_DISPATCH:
            raise UnsupportedFileFormatError(
                extension=extension,
                supported=supported,
                path=str(path),
            )

        logger.debug(
            "Extension '%s' is supported. Reader: '%s'.",
            extension,
            self._READER_DISPATCH[extension],
        )
        return extension

    def _get_reader(self, extension: str) -> Callable[[Path], pd.DataFrame]:
        """
        Resolves the registered reader method for the given extension.

        This is the Strategy dispatch point. It retrieves the method name from
        ``_READER_DISPATCH`` and returns the bound method via ``getattr``,
        keeping all dispatch logic in one place.

        Args:
            extension: Lowercase, dot-prefixed file extension (e.g., '.csv').
                Must already be validated by ``_validate_extension``.

        Returns:
            Callable: A bound method of this DataLoader instance that accepts
                a ``Path`` and returns a ``pd.DataFrame``.
        """
        method_name: str = self._READER_DISPATCH[extension]
        return getattr(self, method_name)  # type: ignore[return-value]

    def _validate_non_empty(self, df: pd.DataFrame, path: Path) -> None:
        """
        Asserts the loaded DataFrame contains at least one row and one column.

        Validation order: FOURTH (last — occurs after full I/O read).

        Args:
            df: The DataFrame returned by the reader.
            path: The source path; included in the exception for context.

        Raises:
            EmptyDatasetError: If ``df.empty`` is True.
        """
        if df.empty:
            raise EmptyDatasetError(
                path=str(path),
                num_rows=len(df),
                num_columns=len(df.columns),
            )

        logger.debug(
            "Non-empty validation passed: %d rows x %d columns.",
            len(df),
            len(df.columns),
        )

    # -----------------------------------------------------------------------
    # Private reader methods
    # -----------------------------------------------------------------------

    def _read_csv(self, path: Path) -> pd.DataFrame:
        """
        Reads a CSV file into a pandas DataFrame using centralized settings.

        All pandas-specific and OS-level exceptions are caught here and
        re-raised as domain exceptions to decouple callers from pandas
        internals. If the I/O engine is ever replaced, callers still see
        the same domain exception types.

        Configuration:
            encoding   — from settings.ingestion.encoding
            low_memory — from settings.ingestion.low_memory

        Special case:
            pandas.errors.EmptyDataError is raised for zero-byte or
            header-only CSV files before _validate_non_empty can run.
            This is caught and re-raised as EmptyDatasetError immediately
            so callers always see the semantically correct exception type.

        Args:
            path: Resolved absolute path to the CSV file.

        Returns:
            pd.DataFrame: The raw DataFrame exactly as pandas read it.
                No transformations are applied.

        Raises:
            EmptyDatasetError: If the CSV file is empty or has no columns.
            DatasetReadError: Wrapping any other exception raised by pandas or the OS.
        """
        logger.debug(
            "Reading CSV: '%s' | encoding=%s | low_memory=%s.",
            path.name,
            settings.ingestion.encoding,
            settings.ingestion.low_memory,
        )

        try:
            dataframe: pd.DataFrame = pd.read_csv(
                path,
                encoding=settings.ingestion.encoding,
                low_memory=settings.ingestion.low_memory,
            )
        except pd.errors.EmptyDataError as exc:
            # pandas raises EmptyDataError for zero-byte or header-only files
            # before we reach _validate_non_empty. Surface it as the correct
            # domain exception immediately so callers see EmptyDatasetError
            # regardless of where emptiness is detected.
            raise EmptyDatasetError(
                path=str(path),
                num_rows=0,
                num_columns=0,
            ) from exc
        except Exception as exc:
            raise DatasetReadError(path=str(path), original_error=exc) from exc

        return dataframe

    # -----------------------------------------------------------------------
    # Future reader stubs (not implemented — register when needed)
    # -----------------------------------------------------------------------
    #
    # def _read_parquet(self, path: Path) -> pd.DataFrame:
    #     """Reads a Parquet file. Register in _READER_DISPATCH to activate."""
    #     try:
    #         return pd.read_parquet(path)
    #     except Exception as exc:
    #         raise DatasetReadError(path=str(path), original_error=exc) from exc
    #
    # def _read_feather(self, path: Path) -> pd.DataFrame:
    #     """Reads a Feather/Arrow IPC file. Register in _READER_DISPATCH to activate."""
    #     try:
    #         return pd.read_feather(path)
    #     except Exception as exc:
    #         raise DatasetReadError(path=str(path), original_error=exc) from exc
    #
    # def _read_excel(self, path: Path) -> pd.DataFrame:
    #     """Reads an Excel file (.xlsx / .xls). Register in _READER_DISPATCH to activate."""
    #     try:
    #         return pd.read_excel(path)
    #     except Exception as exc:
    #         raise DatasetReadError(path=str(path), original_error=exc) from exc

    def _read_zip(self, path: Path) -> pd.DataFrame:
        """
        Reads a CSV dataset contained inside a ZIP archive.

        The method inspects the ZIP manifest to locate CSV files without
        extracting the archive to disk. The CSV is streamed directly from
        the ZIP into pandas, keeping memory usage equivalent to reading a
        plain CSV file.

        Selection logic (in priority order):
            1. If ``settings.ingestion.zip_csv_target`` is set, that exact
               inner filename is used. A clear error is raised if it is not
               found inside the archive.
            2. If the ZIP contains exactly one CSV file, it is selected
               automatically and its name is logged at INFO level.
            3. If the ZIP contains multiple CSV files and no target is
               pinned, a ``DatasetReadError`` is raised listing all
               candidates so the caller can set ``FRAUD_INGESTION_ZIP_CSV_TARGET``
               to resolve the ambiguity.
            4. If the ZIP contains no CSV files at all, an
               ``EmptyDatasetError`` is raised.

        Supported inner file encodings follow ``settings.ingestion.encoding``.

        Args:
            path: Resolved absolute path to the ``.zip`` archive.

        Returns:
            pd.DataFrame: The raw DataFrame read from the inner CSV file.
                No transformations are applied.

        Raises:
            EmptyDatasetError: No CSV files found inside the ZIP archive.
            DatasetReadError: The archive is corrupt, the pinned target file
                is not found, multiple CSVs exist with no target pinned, or
                any other I/O / parsing error occurs.
        """
        logger.debug("Opening ZIP archive: '%s'.", path.name)

        try:
            with zipfile.ZipFile(path, "r") as zf:

                # ── Step 1: Collect all CSV member names ──────────────────
                all_members: list[str] = zf.namelist()
                csv_members: list[str] = [
                    name for name in all_members
                    if name.lower().endswith(".csv")
                    and not name.startswith("__MACOSX")  # skip macOS metadata
                ]

                logger.debug(
                    "ZIP '%s' contains %d member(s), %d CSV file(s): %s",
                    path.name,
                    len(all_members),
                    len(csv_members),
                    csv_members,
                )

                # ── Step 2: No CSV found ───────────────────────────────────
                if not csv_members:
                    raise EmptyDatasetError(
                        path=str(path),
                        num_rows=0,
                        num_columns=0,
                    )

                # ── Step 3: Resolve the target inner filename ──────────────
                pinned: str | None = settings.ingestion.zip_csv_target

                if pinned is not None:
                    # Caller explicitly pinned a target — validate it exists.
                    if pinned not in csv_members:
                        raise DatasetReadError(
                            path=str(path),
                            original_error=FileNotFoundError(
                                f"Pinned target '{pinned}' not found in ZIP archive. "
                                f"Available CSV files: {csv_members}. "
                                f"Update FRAUD_INGESTION_ZIP_CSV_TARGET to one of these."
                            ),
                        )
                    target_csv: str = pinned
                    logger.info(
                        "ZIP target pinned via settings: '%s' -> '%s'.",
                        path.name,
                        target_csv,
                    )

                elif len(csv_members) == 1:
                    # Exactly one CSV — auto-select without ambiguity.
                    target_csv = csv_members[0]
                    logger.info(
                        "ZIP auto-selected single CSV: '%s' -> '%s'.",
                        path.name,
                        target_csv,
                    )

                else:
                    # Multiple CSVs and no target pinned — raise with guidance.
                    raise DatasetReadError(
                        path=str(path),
                        original_error=ValueError(
                            f"ZIP archive '{path.name}' contains {len(csv_members)} CSV files: "
                            f"{csv_members}. "
                            f"Set the environment variable FRAUD_INGESTION_ZIP_CSV_TARGET "
                            f"to the exact inner filename you want to load. "
                            f"Example: FRAUD_INGESTION_ZIP_CSV_TARGET=creditcard.csv"
                        ),
                    )

                # ── Step 4: Stream the CSV from the ZIP into pandas ────────
                logger.info(
                    "Reading inner CSV '%s' from ZIP '%s' | encoding=%s.",
                    target_csv,
                    path.name,
                    settings.ingestion.encoding,
                )

                with zf.open(target_csv) as inner_file:
                    try:
                        dataframe: pd.DataFrame = pd.read_csv(
                            inner_file,
                            encoding=settings.ingestion.encoding,
                            low_memory=settings.ingestion.low_memory,
                        )
                    except pd.errors.EmptyDataError as exc:
                        raise EmptyDatasetError(
                            path=str(path),
                            num_rows=0,
                            num_columns=0,
                        ) from exc
                    except Exception as exc:
                        raise DatasetReadError(
                            path=str(path),
                            original_error=exc,
                        ) from exc

        except (EmptyDatasetError, DatasetReadError):
            # Re-raise domain exceptions unchanged — do not double-wrap.
            raise
        except zipfile.BadZipFile as exc:
            raise DatasetReadError(
                path=str(path),
                original_error=exc,
            ) from exc
        except Exception as exc:
            raise DatasetReadError(
                path=str(path),
                original_error=exc,
            ) from exc

        return dataframe



    # -----------------------------------------------------------------------
    # Private helper methods
    # -----------------------------------------------------------------------

    def _compute_dataset_hash(self, path: Path) -> str:
        """
        Computes the SHA-256 hex digest of the raw file bytes.

        The hash is computed by reading the file in fixed-size chunks
        (configured via settings.ingestion.hash_chunk_size_bytes) to avoid
        loading the entire file into memory twice — once for the DataFrame
        and once for hashing. This is safe for large files (multi-GB) in
        containerized environments with limited RAM.

        The resulting hash uniquely identifies the file's content:
            - Two files with different content will always have different hashes.
            - The same file loaded twice will always produce the same hash.

        Intended uses:
            - MLflow dataset versioning: mlflow.set_tag("dataset_hash", ...).
            - Cache invalidation: skip re-processing if hash is unchanged.
            - Data drift detection: compare hash between pipeline runs.
            - Audit trail: immutable content fingerprint in DatasetMetadata.

        Args:
            path: Resolved absolute path to the file to hash.

        Returns:
            str: Lowercase SHA-256 hex digest (64 hex characters).
                 Example: 'a3f8c2d1e4b5f6a7...'

        Raises:
            DatasetReadError: If the file cannot be read during hashing.
        """
        chunk_size: int = settings.ingestion.hash_chunk_size_bytes
        sha256 = hashlib.sha256()

        logger.debug(
            "Computing SHA-256 hash for '%s' (chunk size: %s bytes).",
            path.name,
            f"{chunk_size:,}",
        )

        try:
            with path.open("rb") as file_handle:
                while True:
                    chunk: bytes = file_handle.read(chunk_size)
                    if not chunk:
                        break
                    sha256.update(chunk)
        except Exception as exc:
            raise DatasetReadError(path=str(path), original_error=exc) from exc

        return sha256.hexdigest()

    def _build_metadata(
        self,
        path: Path,
        df: pd.DataFrame,
        dataset_hash: str | None = None,
    ) -> DatasetMetadata:
        """
        Constructs an immutable DatasetMetadata record from file stats and the
        loaded DataFrame.

        This method only reads shape-level information from the DataFrame
        (row count, column count, column names, memory usage). It never reads
        or logs cell values to protect data privacy.

        Args:
            path: Resolved absolute path to the source file.
            df: The fully loaded, validated, non-empty DataFrame.
            dataset_hash: Pre-computed SHA-256 hex digest of the raw file bytes.
                Pass None to omit the hash from metadata.

        Returns:
            DatasetMetadata: An immutable, JSON-serializable metadata record.
        """
        logger.debug("Building dataset metadata for '%s'.", path.name)

        # deep=True computes actual memory including object dtype overhead.
        memory_usage_bytes: int = int(df.memory_usage(deep=True).sum())

        metadata = DatasetMetadata(
            # Core identity
            file_name=path.name,
            file_path=str(path),
            file_extension=path.suffix.lower(),
            # Size and shape
            file_size_bytes=path.stat().st_size,
            num_rows=len(df),
            num_columns=len(df.columns),
            column_names=tuple(df.columns.tolist()),
            memory_usage_bytes=memory_usage_bytes,
            # Pipeline context
            target_column=settings.ingestion.target_column,
            load_timestamp=datetime.now(tz=timezone.utc),
            # Optional enrichment
            dataset_hash=dataset_hash,
            dataset_version=None,           # Set by orchestrator if needed.
            loaded_by=settings.ingestion.loaded_by,
        )

        logger.debug(
            "Metadata built: %s",
            metadata,
            extra={
                "file_name": metadata.file_name,
                "file_size_bytes": metadata.file_size_bytes,
                "memory_usage_bytes": metadata.memory_usage_bytes,
                "load_timestamp": metadata.load_timestamp.isoformat(),
                "dataset_hash": metadata.dataset_hash,
                "loaded_by": metadata.loaded_by,
            },
        )

        return metadata
