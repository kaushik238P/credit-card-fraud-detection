"""
Training Layer — public API surface.

Usage:
    from src.training import TrainingPipeline, TrainingResult

    pipeline = TrainingPipeline()
    result: TrainingResult = pipeline.train(preprocessed_dataset)
"""

from src.training.exceptions import (
    ModelCreationError,
    PipelineError,
    SerializationError,
    TrainingError,
    TrainingFailureError,
)
from src.training.models import (
    ModelMetadata,
    ModelType,
    SerializedModel,
    TrainingReport,
    TrainingResult,
)
from src.training.pipeline import TrainingPipeline
from src.training.registry import MODEL_REGISTRY, ModelRegistryEntry

__all__ = [
    # Pipeline
    "TrainingPipeline",
    # Models
    "TrainingResult",
    "TrainingReport",
    "ModelMetadata",
    "SerializedModel",
    "ModelType",
    # Registry
    "MODEL_REGISTRY",
    "ModelRegistryEntry",
    # Exceptions
    "TrainingError",
    "ModelCreationError",
    "TrainingFailureError",
    "SerializationError",
    "PipelineError",
]
