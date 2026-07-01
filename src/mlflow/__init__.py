"""
MLflow Experiment Tracking and Model Registry Layer.

Usage:
    from src.mlflow import MLflowPipeline, ExperimentResult

    pipeline = MLflowPipeline()
    result: ExperimentResult = pipeline.track(training_result, evaluation_result)
"""

from src.mlflow.exceptions import (
    ArtifactUploadError,
    MLflowError,
    PipelineError,
    RegistrationError,
    TrackingError,
)
from src.mlflow.models import (
    ArtifactMetadata,
    ExperimentReport,
    ExperimentResult,
    RegisteredModel,
    RunMetadata,
)
from src.mlflow.pipeline import MLflowPipeline
from src.mlflow.tracker import MLflowTracker

__all__ = [
    # Orchestrator & Wrapper
    "MLflowPipeline",
    "MLflowTracker",
    # Data Models
    "ExperimentResult",
    "RunMetadata",
    "RegisteredModel",
    "ExperimentReport",
    "ArtifactMetadata",
    # Exceptions
    "MLflowError",
    "TrackingError",
    "ArtifactUploadError",
    "RegistrationError",
    "PipelineError",
]
