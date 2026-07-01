"""
Immutable data models for the Training Layer.

Model hierarchy:
    ModelType            — enum of supported estimator types
    ModelMetadata        — per-model configuration and lineage record
    TrainingReport       — complete audit record for one training run
    SerializedModel      — all artifact paths produced during serialization
    TrainingResult       — primary pipeline exchange object
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# ModelType
# ---------------------------------------------------------------------------


class ModelType(enum.Enum):
    """Enumeration of all supported model types."""

    LOGISTIC_REGRESSION = "LOGISTIC_REGRESSION"
    RANDOM_FOREST = "RANDOM_FOREST"
    XGBOOST = "XGBOOST"
    LIGHTGBM = "LIGHTGBM"
    CATBOOST = "CATBOOST"

    @classmethod
    def from_string(cls, value: str) -> "ModelType":
        """
        Resolves a ModelType from a case-insensitive string.

        Args:
            value: Model type name (e.g. "xgboost", "XGBOOST").

        Returns:
            ModelType: Matching enum member.

        Raises:
            ValueError: If no matching member exists.
        """
        try:
            return cls(value.upper())
        except (KeyError, ValueError):
            valid = [m.value for m in cls]
            raise ValueError(
                f"Unknown model type '{value}'. Valid options: {valid}"
            )


# ---------------------------------------------------------------------------
# ModelMetadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelMetadata:
    """
    Per-model configuration and lineage record.

    Captures everything needed to reproduce a training run exactly.

    Attributes:
        model_type: Enum identifier of the model.
        model_name: Human-readable label (e.g. "XGBoost Classifier").
        model_version: Run timestamp string "YYYYMMDD_HHMMSS".
        training_version: Training layer software version.
        hyperparameters: Serialisable dict of all params passed to estimator.
        random_seed: Global seed used — from config.settings.training.
        imbalance_strategy: One of CLASS_WEIGHT / SCALE_POS_WEIGHT /
            SAMPLE_WEIGHT / NONE.
        imbalance_ratio: Computed ratio of legitimate / fraud in train fold.
        imbalance_value_applied: Actual value passed to estimator
            (e.g. scale_pos_weight=171.8 or class_weight={0:1, 1:171.8}).
        feature_count: Number of input features.
        feature_names: Ordered feature names from PreprocessedDataset.
        feature_schema_hash: SHA-256 of sorted feature_names for drift detection.
        dataset_hash: SHA-256 propagated from PreprocessingReport.
        preprocessing_version: Version string from PreprocessingReport config.
        preprocessing_artifact_version: Run ID of the preprocessing artifacts
            (e.g. "20260630_163000") consumed to produce this model.
    """

    model_type: ModelType
    model_name: str
    model_version: str
    training_version: str
    hyperparameters: dict = field(hash=False, compare=False)
    random_seed: int
    imbalance_strategy: str
    imbalance_ratio: float
    imbalance_value_applied: Any = field(hash=False, compare=False)
    feature_count: int
    feature_names: tuple[str, ...]
    feature_schema_hash: str
    dataset_hash: str | None
    preprocessing_version: str
    preprocessing_artifact_version: str

    def to_dict(self) -> dict:
        """Returns a JSON-serialisable dictionary."""
        return {
            "model_type": self.model_type.value,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "training_version": self.training_version,
            "hyperparameters": self.hyperparameters,
            "random_seed": self.random_seed,
            "imbalance_strategy": self.imbalance_strategy,
            "imbalance_ratio": round(self.imbalance_ratio, 4),
            "imbalance_value_applied": str(self.imbalance_value_applied),
            "feature_count": self.feature_count,
            "feature_names": list(self.feature_names),
            "feature_schema_hash": self.feature_schema_hash,
            "dataset_hash": self.dataset_hash,
            "preprocessing_version": self.preprocessing_version,
            "preprocessing_artifact_version": self.preprocessing_artifact_version,
        }

    def __str__(self) -> str:
        return (
            f"ModelMetadata("
            f"type={self.model_type.value}, "
            f"version={self.model_version}, "
            f"features={self.feature_count}, "
            f"seed={self.random_seed})"
        )


# ---------------------------------------------------------------------------
# TrainingReport
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingReport:
    """
    Complete, immutable audit record for one training run.

    Contains no evaluation metrics — those belong to the Evaluation Layer.

    Attributes:
        report_id: UUID4 string uniquely identifying this run.
        model_metadata: Nested metadata record.
        training_rows: Row count of the training fold.
        training_fraud_count: Number of positive (fraud) rows in train fold.
        training_fraud_percentage: Fraud percentage in train fold (0–100).
        imbalance_ratio: Legitimate / fraud ratio computed from train fold.
        feature_count: Number of features passed to the estimator.
        hyperparameters: Serialisable dict (mirrors model_metadata).
        artifact_paths: Mapping of artifact name → absolute file path.
        training_timeline: Ordered list of step dicts with start/end/duration.
        warnings: Non-fatal warning messages accumulated during the run.
        generated_at: ISO 8601 UTC timestamp.
        duration_ms: Total wall-clock time for the full training run.
    """

    report_id: str
    model_metadata: ModelMetadata
    training_rows: int
    training_fraud_count: int
    training_fraud_percentage: float
    imbalance_ratio: float
    feature_count: int
    hyperparameters: dict = field(hash=False, compare=False)
    artifact_paths: dict[str, str] = field(hash=False, compare=False)
    training_timeline: tuple[dict, ...] = field(hash=False, compare=False)
    warnings: tuple[str, ...]
    generated_at: str
    duration_ms: float

    def to_dict(self) -> dict:
        """Returns a fully JSON-serialisable dictionary."""
        return {
            "report_id": self.report_id,
            "model_metadata": self.model_metadata.to_dict(),
            "training_rows": self.training_rows,
            "training_fraud_count": self.training_fraud_count,
            "training_fraud_percentage": round(self.training_fraud_percentage, 6),
            "imbalance_ratio": round(self.imbalance_ratio, 4),
            "feature_count": self.feature_count,
            "hyperparameters": self.hyperparameters,
            "artifact_paths": self.artifact_paths,
            "training_timeline": list(self.training_timeline),
            "warnings": list(self.warnings),
            "generated_at": self.generated_at,
            "duration_ms": round(self.duration_ms, 2),
        }

    def __str__(self) -> str:
        return (
            f"TrainingReport("
            f"id={self.report_id[:8]}..., "
            f"model={self.model_metadata.model_type.value}, "
            f"rows={self.training_rows:,}, "
            f"duration={self.duration_ms:.0f}ms)"
        )


# ---------------------------------------------------------------------------
# SerializedModel
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SerializedModel:
    """
    All artifact file paths produced during model serialization.

    Consumed by the Evaluation Layer and FastAPI inference service.

    Attributes:
        model_path: Absolute path to model.joblib.
        metadata_path: Absolute path to training_metadata.json.
        report_path: Absolute path to training_report.json.
        feature_names_path: Absolute path to feature_names.json.
        manifest_path: Absolute path to manifest.json.
        artifact_dir: Versioned run directory (YYYYMMDD_HHMMSS/).
        latest_dir: artifacts/models/latest/ path.
    """

    model_path: str
    metadata_path: str
    report_path: str
    feature_names_path: str
    manifest_path: str
    artifact_dir: str
    latest_dir: str

    def to_dict(self) -> dict:
        """Returns a JSON-serialisable dictionary."""
        return {
            "model_path": self.model_path,
            "metadata_path": self.metadata_path,
            "report_path": self.report_path,
            "feature_names_path": self.feature_names_path,
            "manifest_path": self.manifest_path,
            "artifact_dir": self.artifact_dir,
            "latest_dir": self.latest_dir,
        }


# ---------------------------------------------------------------------------
# TrainingResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingResult:
    """
    Primary pipeline exchange object returned by TrainingPipeline.

    Passed from the Training Layer to the Evaluation Layer.

    Design notes:
        - estimator, X/y fields carry compare=False, hash=False, repr=False
          to keep frozen dataclasses hashable.
        - frozen=True prevents field reassignment; downstream consumers must
          work on explicit copies of any mutable objects.
        - to_dict() excludes estimator and DataFrames for JSON serialisation.

    Attributes:
        estimator: Fitted sklearn-compatible estimator.
        validation_predictions: Cached predict() output on X_validation.
            Produced by the training pipeline — never the evaluation layer.
        validation_probabilities: Cached predict_proba()[:, 1] on X_validation.
        feature_names: Ordered feature names (mirrors X_train columns).
        target_column: Name of the binary target column.
        metadata: Complete ModelMetadata for this run.
        report: Complete TrainingReport for this run.
        serialized: All artifact file paths.
    """

    estimator: Any = field(repr=False, compare=False, hash=False)
    validation_predictions: np.ndarray = field(repr=False, compare=False, hash=False)
    validation_probabilities: np.ndarray = field(repr=False, compare=False, hash=False)
    feature_names: tuple[str, ...]
    target_column: str
    metadata: ModelMetadata
    report: TrainingReport
    serialized: SerializedModel

    def to_dict(self) -> dict:
        """Returns a JSON-serialisable dictionary excluding arrays."""
        return {
            "feature_names": list(self.feature_names),
            "target_column": self.target_column,
            "metadata": self.metadata.to_dict(),
            "report": self.report.to_dict(),
            "serialized": self.serialized.to_dict(),
        }

    def __repr__(self) -> str:
        return (
            f"TrainingResult("
            f"model={self.metadata.model_type.value}, "
            f"version={self.metadata.model_version}, "
            f"features={self.metadata.feature_count})"
        )

    def __str__(self) -> str:
        return repr(self)
