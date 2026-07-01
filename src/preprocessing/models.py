"""
Immutable data models for the Preprocessing Layer.

All models are frozen dataclasses. Every model exposes to_dict() returning
a fully JSON-serialisable dict (no pandas, no numpy objects).

Model hierarchy:
    DatasetSplit             — one time-based fold (X, y, metadata)
    TransformationMetadata   — per-transformer lineage record
    TransformationSummary    — aggregated stats across all transformers
    PreprocessingReport      — complete audit record for one run
    PreprocessedDataset      — primary pipeline exchange object
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.ingestion.models import DatasetMetadata


# ---------------------------------------------------------------------------
# DatasetSplit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSplit:
    """
    Holds one complete time-based fold after splitting.

    Attributes:
        fold_name: Identifier — "train", "validation", or "test".
        X: Feature matrix for this fold.
        y: Target vector for this fold.
        start_date: ISO 8601 date of the earliest transaction in the fold.
        end_date: ISO 8601 date of the latest transaction in the fold.
        num_rows: Row count.
        num_features: Column count of X.
        fraud_count: Number of positive (fraud=1) rows.
        fraud_pct: Percentage of fraud rows (0–100).
    """

    fold_name: str
    X: pd.DataFrame = field(repr=False, compare=False, hash=False)
    y: pd.Series = field(repr=False, compare=False, hash=False)
    start_date: str
    end_date: str
    num_rows: int
    num_features: int
    fraud_count: int
    fraud_pct: float

    def to_dict(self) -> dict:
        """Returns a JSON-serialisable dictionary excluding DataFrames."""
        return {
            "fold_name": self.fold_name,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "num_rows": self.num_rows,
            "num_features": self.num_features,
            "fraud_count": self.fraud_count,
            "fraud_pct": round(self.fraud_pct, 4),
        }

    def __str__(self) -> str:
        return (
            f"DatasetSplit(fold={self.fold_name!r}, "
            f"rows={self.num_rows:,}, "
            f"fraud_pct={self.fraud_pct:.4f}%, "
            f"dates={self.start_date} → {self.end_date})"
        )


# ---------------------------------------------------------------------------
# TransformationMetadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransformationMetadata:
    """
    Lineage record for a single transformation step.

    Attributes:
        transformer_name: Class name of the transformer used.
        step_name: Registry step name (e.g. "FREQUENCY_ENCODING").
        transformer_type: One of "ENCODER", "SCALER", "IDENTITY".
        columns_input: Columns consumed by this transformer.
        columns_output: Columns produced (may differ from input for OHE).
        input_shape: (rows, cols) of X before transformation.
        output_shape: (rows, cols) of X after transformation.
        fit_on_fold: Always "train" — enforced by the pipeline.
        parameters: Serialisable transformer configuration parameters.
        duration_ms: Wall-clock time to fit + transform in milliseconds.
    """

    transformer_name: str
    step_name: str
    transformer_type: str
    columns_input: tuple[str, ...]
    columns_output: tuple[str, ...]
    input_shape: tuple[int, int]
    output_shape: tuple[int, int]
    fit_on_fold: str
    parameters: dict = field(hash=False, compare=False)
    duration_ms: float

    def to_dict(self) -> dict:
        """Returns a JSON-serialisable dictionary."""
        return {
            "transformer_name": self.transformer_name,
            "step_name": self.step_name,
            "transformer_type": self.transformer_type,
            "columns_input": list(self.columns_input),
            "columns_output": list(self.columns_output),
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "fit_on_fold": self.fit_on_fold,
            "parameters": self.parameters,
            "duration_ms": round(self.duration_ms, 2),
        }


# ---------------------------------------------------------------------------
# TransformationSummary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransformationSummary:
    """
    Aggregated summary of all transformations applied in one run.

    Attributes:
        total_columns_input: Column count entering the transformation pipeline.
        total_columns_output: Column count after all transformations (OHE
            can expand column count).
        columns_encoded_frequency: Columns encoded by FrequencyEncoder.
        columns_encoded_ohe: Columns encoded by OneHotEncoder.
        columns_scaled: Columns scaled (or passed through IdentityTransformer).
        columns_identity: Columns passed through without change.
        columns_dropped: Columns dropped before transformations.
        ohe_expansion_map: Mapping of original column → list of OHE output
            column names.
        transformation_order: Ordered list of step names executed.
    """

    total_columns_input: int
    total_columns_output: int
    columns_encoded_frequency: tuple[str, ...]
    columns_encoded_ohe: tuple[str, ...]
    columns_scaled: tuple[str, ...]
    columns_identity: tuple[str, ...]
    columns_dropped: tuple[str, ...]
    ohe_expansion_map: dict[str, list[str]] = field(hash=False, compare=False)
    transformation_order: tuple[str, ...]

    def to_dict(self) -> dict:
        """Returns a JSON-serialisable dictionary."""
        return {
            "total_columns_input": self.total_columns_input,
            "total_columns_output": self.total_columns_output,
            "columns_encoded_frequency": list(self.columns_encoded_frequency),
            "columns_encoded_ohe": list(self.columns_encoded_ohe),
            "columns_scaled": list(self.columns_scaled),
            "columns_identity": list(self.columns_identity),
            "columns_dropped": list(self.columns_dropped),
            "ohe_expansion_map": self.ohe_expansion_map,
            "transformation_order": list(self.transformation_order),
        }


# ---------------------------------------------------------------------------
# PreprocessingReport
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreprocessingReport:
    """
    Complete, immutable audit record for one preprocessing run.

    Attributes:
        report_id: UUID4 string uniquely identifying this run.
        dataset_name: Filename of the source dataset.
        dataset_hash: SHA-256 hash propagated from EngineeredDataset.
        split_strategy: Always "TIME_BASED".
        train_date_range: (start, end) ISO date strings for training fold.
        validation_date_range: (start, end) ISO date strings for validation.
        test_date_range: (start, end) ISO date strings for test fold.
        train_rows: Row count of the training fold.
        validation_rows: Row count of the validation fold.
        test_rows: Row count of the test fold.
        train_fraud_pct: Fraud percentage in the training fold.
        validation_fraud_pct: Fraud percentage in the validation fold.
        test_fraud_pct: Fraud percentage in the test fold.
        feature_names_input: Feature column names entering transformations.
        feature_names_output: Feature column names after all transformations.
        input_feature_count: len(feature_names_input).
        output_feature_count: len(feature_names_output).
        transformation_summary: Aggregated transformation statistics.
        transformation_metadata: Ordered per-transformer lineage records.
        transformation_timeline: Ordered list of step dicts with duration_ms.
        artifact_paths: Mapping of artifact name → absolute file path.
        parquet_paths: Mapping of dataset name → absolute file path.
        feature_schema_path: Absolute path to input_features.json / output_features.json.
        class_distribution_path: Absolute path to class_distribution.json.
        data_integrity_path: Absolute path to data_integrity.json.
        dataset_schema_path: Absolute path to dataset_schema.json.
        scaling_enabled: Whether RobustScaler was used (vs IdentityTransformer).
        warnings: Non-fatal warning messages accumulated during the run.
        generated_at: ISO 8601 UTC timestamp of report generation.
        duration_ms: Total wall-clock time for the full preprocessing run.
    """

    report_id: str
    dataset_name: str
    dataset_hash: str | None
    split_strategy: str
    train_date_range: tuple[str, str]
    validation_date_range: tuple[str, str]
    test_date_range: tuple[str, str]
    train_rows: int
    validation_rows: int
    test_rows: int
    train_fraud_pct: float
    validation_fraud_pct: float
    test_fraud_pct: float
    feature_names_input: tuple[str, ...]
    feature_names_output: tuple[str, ...]
    input_feature_count: int
    output_feature_count: int
    transformation_summary: TransformationSummary
    transformation_metadata: tuple[TransformationMetadata, ...]
    transformation_timeline: tuple[dict, ...] = field(hash=False, compare=False)
    artifact_paths: dict[str, str] = field(hash=False, compare=False)
    parquet_paths: dict[str, str] = field(hash=False, compare=False)
    feature_schema_path: str
    class_distribution_path: str
    data_integrity_path: str
    dataset_schema_path: str
    encoded_columns: tuple[str, ...]
    frequency_encoded_columns: tuple[str, ...]
    one_hot_encoded_columns: tuple[str, ...]
    dropped_columns: tuple[str, ...]
    scaling_enabled: bool
    warnings: tuple[str, ...]
    generated_at: str
    duration_ms: float

    def to_dict(self) -> dict:
        """Returns a fully JSON-serialisable dictionary."""
        return {
            "report_id": self.report_id,
            "dataset_name": self.dataset_name,
            "dataset_hash": self.dataset_hash,
            "split_strategy": self.split_strategy,
            "train_date_range": list(self.train_date_range),
            "validation_date_range": list(self.validation_date_range),
            "test_date_range": list(self.test_date_range),
            "train_rows": self.train_rows,
            "validation_rows": self.validation_rows,
            "test_rows": self.test_rows,
            "train_fraud_pct": round(self.train_fraud_pct, 4),
            "validation_fraud_pct": round(self.validation_fraud_pct, 4),
            "test_fraud_pct": round(self.test_fraud_pct, 4),
            "feature_names_input": list(self.feature_names_input),
            "feature_names_output": list(self.feature_names_output),
            "input_feature_count": self.input_feature_count,
            "output_feature_count": self.output_feature_count,
            "transformation_summary": self.transformation_summary.to_dict(),
            "transformation_metadata": [m.to_dict() for m in self.transformation_metadata],
            "transformation_timeline": list(self.transformation_timeline),
            "artifact_paths": self.artifact_paths,
            "parquet_paths": self.parquet_paths,
            "feature_schema_path": self.feature_schema_path,
            "class_distribution_path": self.class_distribution_path,
            "data_integrity_path": self.data_integrity_path,
            "dataset_schema_path": self.dataset_schema_path,
            "encoded_columns": list(self.encoded_columns),
            "frequency_encoded_columns": list(self.frequency_encoded_columns),
            "one_hot_encoded_columns": list(self.one_hot_encoded_columns),
            "dropped_columns": list(self.dropped_columns),
            "scaling_enabled": self.scaling_enabled,
            "warnings": list(self.warnings),
            "generated_at": self.generated_at,
            "duration_ms": round(self.duration_ms, 2),
        }

    def __str__(self) -> str:
        return (
            f"PreprocessingReport("
            f"id={self.report_id[:8]}..., "
            f"dataset={self.dataset_name!r}, "
            f"train={self.train_rows:,}, "
            f"val={self.validation_rows:,}, "
            f"test={self.test_rows:,}, "
            f"features={self.input_feature_count}→{self.output_feature_count}, "
            f"duration={self.duration_ms:.0f}ms)"
        )


# ---------------------------------------------------------------------------
# PreprocessedDataset
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreprocessedDataset:
    """
    Primary pipeline exchange object returned by PreprocessingPipeline.

    This is the only object passed from the Preprocessing Layer to the
    Training Layer. Contains all three folds, feature metadata, and the
    complete provenance report.

    Design notes:
        - All DataFrame/Series fields carry ``compare=False, hash=False,
          repr=False`` to keep frozen dataclasses hashable.
        - ``frozen=True`` prevents field reassignment but does NOT prevent
          in-place mutation — downstream consumers must work on explicit copies.
        - ``to_dict()`` excludes DataFrames for JSON serialisation.

    Attributes:
        X_train: Training feature matrix.
        X_validation: Validation feature matrix.
        X_test: Test feature matrix.
        y_train: Training target vector.
        y_validation: Validation target vector.
        y_test: Test target vector.
        feature_names: Output feature names after all transformations.
        target_column: Name of the target column.
        metadata: DatasetMetadata propagated from EngineeredDataset.
        report: Complete PreprocessingReport for this run.
    """

    X_train: pd.DataFrame = field(repr=False, compare=False, hash=False)
    X_validation: pd.DataFrame = field(repr=False, compare=False, hash=False)
    X_test: pd.DataFrame = field(repr=False, compare=False, hash=False)
    y_train: pd.Series = field(repr=False, compare=False, hash=False)
    y_validation: pd.Series = field(repr=False, compare=False, hash=False)
    y_test: pd.Series = field(repr=False, compare=False, hash=False)
    feature_names: tuple[str, ...]
    target_column: str
    metadata: DatasetMetadata
    report: PreprocessingReport

    def to_dict(self) -> dict:
        """
        Returns a JSON-serialisable dictionary excluding DataFrames.

        Use the DataFrame fields directly for downstream processing.
        """
        return {
            "feature_names": list(self.feature_names),
            "target_column": self.target_column,
            "metadata": self.metadata.to_dict(),
            "report": self.report.to_dict(),
        }

    def __repr__(self) -> str:
        return (
            f"PreprocessedDataset("
            f"dataset={self.metadata.file_name!r}, "
            f"X_train={self.X_train.shape}, "
            f"X_val={self.X_validation.shape}, "
            f"X_test={self.X_test.shape}, "
            f"features={len(self.feature_names)})"
        )

    def __str__(self) -> str:
        return repr(self)
