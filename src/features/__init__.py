"""
Feature Engineering Layer public API for the Credit Card Fraud Detection System.

Public surface area:
    FeatureEngineeringPipeline  -- primary entry point
    EngineeredDataset           -- primary output object
    FeatureEngineeringReport    -- complete run summary
    FeatureDefinition           -- feature metadata/lineage dataclass
    FeatureGroup                -- grouped feature definitions
    EncodingRecommendation      -- encoding strategy recommendation
    FeatureEngineeringError     -- base exception
    TransformationError         -- single feature computation failure
    FeatureCreationError        -- bad module output
    PipelineError               -- unrecoverable pipeline failure

Usage:
    from src.features import FeatureEngineeringPipeline

    pipeline = FeatureEngineeringPipeline()
    engineered = pipeline.engineer(validated_dataset, eda_report)
"""

from src.features.exceptions import (
    FeatureCreationError,
    FeatureEngineeringError,
    PipelineError,
    TransformationError,
)
from src.features.models import (
    EncodingRecommendation,
    EngineeredDataset,
    FeatureDefinition,
    FeatureEngineeringReport,
    FeatureGroup,
)
from src.features.pipeline import FeatureEngineeringPipeline

__all__ = [
    # Orchestrator
    "FeatureEngineeringPipeline",
    # Output models
    "EngineeredDataset",
    "FeatureEngineeringReport",
    "FeatureDefinition",
    "FeatureGroup",
    "EncodingRecommendation",
    # Exception hierarchy
    "FeatureEngineeringError",
    "TransformationError",
    "FeatureCreationError",
    "PipelineError",
]
