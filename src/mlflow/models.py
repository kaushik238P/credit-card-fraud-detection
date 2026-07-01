"""
Domain models and schema definitions for the MLflow Layer.
All objects are frozen, immutable dataclasses supporting JSON serialisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ArtifactMetadata:
    """
    Metadata representation of a logged or uploaded artifact file.
    """

    filename: str
    size_bytes: int
    sha256: str
    created_at: str
    mime_type: str
    upload_status: str

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serialisable dictionary representation of the artifact metadata."""
        return {
            "filename": self.filename,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "created_at": self.created_at,
            "mime_type": self.mime_type,
            "upload_status": self.upload_status,
        }


@dataclass(frozen=True)
class RunMetadata:
    """
    Payload containing tracked run credentials and lineage metadata.
    """

    run_id: str
    experiment_id: str
    status: str
    artifact_uri: str
    start_time: str
    end_time: str
    duration_ms: float

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serialisable dictionary representation of the run metadata."""
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "status": self.status,
            "artifact_uri": self.artifact_uri,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": round(self.duration_ms, 2),
        }


@dataclass(frozen=True)
class RegisteredModel:
    """
    Metadata representing a registered model version in the Model Registry.
    """

    name: str
    version: str
    stage: str
    source_run_id: str
    model_uri: str

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serialisable dictionary representation of the registered model."""
        return {
            "name": self.name,
            "version": self.version,
            "stage": self.stage,
            "source_run_id": self.source_run_id,
            "model_uri": self.model_uri,
        }


@dataclass(frozen=True)
class ExperimentReport:
    """
    Detailed tracking metrics and configuration mappings logged in the current run.
    """

    logged_metrics: dict[str, float]
    logged_parameters: dict[str, Any]
    logged_tags: dict[str, str]
    uploaded_artifacts: tuple[ArtifactMetadata, ...]

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serialisable dictionary representation of the experiment report."""
        return {
            "logged_metrics": self.logged_metrics,
            "logged_parameters": self.logged_parameters,
            "logged_tags": self.logged_tags,
            "uploaded_artifacts": [a.to_dict() for a in self.uploaded_artifacts],
        }


@dataclass(frozen=True)
class ExperimentResult:
    """
    Final output container returned by the MLflow Layer.
    """

    report: ExperimentReport
    metadata: RunMetadata
    registered_model: RegisteredModel | None = None

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serialisable dictionary representation of the tracking results."""
        return {
            "report": self.report.to_dict(),
            "metadata": self.metadata.to_dict(),
            "registered_model": (
                self.registered_model.to_dict() if self.registered_model is not None else None
            ),
        }
