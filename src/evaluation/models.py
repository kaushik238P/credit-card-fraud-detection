"""
Domain models and schema definitions for the Evaluation Layer.
All objects are frozen, immutable dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MetricResult:
    """
    Immutable representation of a single statistical evaluation metric.
    """

    name: str
    value: float
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serialisable dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "description": self.description,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class BusinessMetric:
    """
    Operational/financial impact metric context.
    """

    name: str
    value: float
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serialisable dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "description": self.description,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ThresholdResult:
    """
    Payload containing threshold boundary optimization decisions.
    """

    optimization_strategy: str
    default_threshold: float
    optimized_threshold: float
    validation_precision: float
    validation_recall: float
    validation_f1: float

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serialisable dictionary."""
        return {
            "optimization_strategy": self.optimization_strategy,
            "default_threshold": self.default_threshold,
            "optimized_threshold": round(self.optimized_threshold, 6),
            "validation_precision": round(self.validation_precision, 6),
            "validation_recall": round(self.validation_recall, 6),
            "validation_f1": round(self.validation_f1, 6),
        }


@dataclass(frozen=True)
class PlotArtifact:
    """
    Descriptor for a generated diagnostic plot image.
    """

    plot_name: str
    file_name: str
    absolute_path: str

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serialisable dictionary."""
        return {
            "plot_name": self.plot_name,
            "file_name": self.file_name,
            "absolute_path": self.absolute_path,
        }


@dataclass(frozen=True)
class EvaluationReport:
    """
    Aggregated evaluation report containing all metric runs, hashes, and run metadata.
    """

    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    business_metrics: dict[str, Any]
    threshold_result: ThresholdResult
    feature_importance: dict[str, float]
    plots: tuple[PlotArtifact, ...]
    warnings: tuple[str, ...]
    dataset_hash: str | None
    feature_schema_hash: str
    preprocessing_version: str
    model_version: str
    evaluation_duration: float
    generated_at: str
    fold_statistics: dict[str, Any]
    threshold_comparison: dict[str, Any]
    calibration_note: str = (
        "Model probabilities are raw estimator outputs. Threshold optimization operates on "
        "uncalibrated probabilities. Probability calibration (Platt Scaling / Isotonic Regression) is future work."
    )

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serialisable dictionary."""
        return {
            "validation_metrics": self.validation_metrics,
            "test_metrics": self.test_metrics,
            "business_metrics": self.business_metrics,
            "threshold_result": self.threshold_result.to_dict(),
            "feature_importance": self.feature_importance,
            "plots": [p.to_dict() for p in self.plots],
            "warnings": list(self.warnings),
            "dataset_hash": self.dataset_hash,
            "feature_schema_hash": self.feature_schema_hash,
            "preprocessing_version": self.preprocessing_version,
            "model_version": self.model_version,
            "evaluation_duration": round(self.evaluation_duration, 2),
            "generated_at": self.generated_at,
            "fold_statistics": self.fold_statistics,
            "threshold_comparison": self.threshold_comparison,
            "calibration_note": self.calibration_note,
        }


@dataclass(frozen=True)
class EvaluationArtifacts:
    """
    Collection of physical files saved to disk by the ArtifactManager.
    """

    report_json_path: str
    report_md_path: str
    metrics_json_path: str
    business_metrics_json_path: str
    threshold_json_path: str
    feature_importance_csv_path: str
    feature_importance_json_path: str
    confusion_matrix_csv_path: str
    prediction_distribution_json_path: str
    prediction_summary_json_path: str
    manifest_path: str
    plot_paths: dict[str, str] = field(hash=False, compare=False)
    artifact_dir: str
    latest_dir: str

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serialisable dictionary."""
        return {
            "report_json_path": self.report_json_path,
            "report_md_path": self.report_md_path,
            "metrics_json_path": self.metrics_json_path,
            "business_metrics_json_path": self.business_metrics_json_path,
            "threshold_json_path": self.threshold_json_path,
            "feature_importance_csv_path": self.feature_importance_csv_path,
            "feature_importance_json_path": self.feature_importance_json_path,
            "confusion_matrix_csv_path": self.confusion_matrix_csv_path,
            "prediction_distribution_json_path": self.prediction_distribution_json_path,
            "prediction_summary_json_path": self.prediction_summary_json_path,
            "manifest_path": self.manifest_path,
            "plot_paths": self.plot_paths,
            "artifact_dir": self.artifact_dir,
            "latest_dir": self.latest_dir,
        }


@dataclass(frozen=True)
class EvaluationResult:
    """
    Final container payload of the Evaluation Layer.
    """

    report: EvaluationReport
    artifacts: EvaluationArtifacts

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serialisable dictionary."""
        return {
            "report": self.report.to_dict(),
            "artifacts": self.artifacts.to_dict(),
        }