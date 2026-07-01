"""
Preprocessing Layer public API for the Credit Card Fraud Detection System.

Public surface area:
    PreprocessingPipeline   -- primary entry point
    PreprocessedDataset     -- primary output object
    PreprocessingReport     -- complete run summary
    DatasetSplit            -- individual fold dataclass
    TransformationMetadata  -- per-transformer lineage record
    TransformationSummary   -- aggregated transformation statistics
    PreprocessingError      -- base exception
    SplitError              -- time-based split failure (always propagates)
    TransformationError     -- per-transformer failure (caught; continues)
    ArtifactError           -- artifact I/O failure (caught; continues)
    PipelineError           -- unrecoverable pipeline failure (always propagates)

Usage:
    from src.preprocessing import PreprocessingPipeline

    pipeline = PreprocessingPipeline()
    preprocessed = pipeline.preprocess(engineered_dataset)
"""

from src.preprocessing.exceptions import (
    ArtifactError,
    PipelineError,
    PreprocessingError,
    SplitError,
    TransformationError,
)
from src.preprocessing.models import (
    DatasetSplit,
    PreprocessedDataset,
    PreprocessingReport,
    TransformationMetadata,
    TransformationSummary,
)
from src.preprocessing.pipeline import PreprocessingPipeline

__all__ = [
    # Orchestrator
    "PreprocessingPipeline",
    # Output models
    "PreprocessedDataset",
    "PreprocessingReport",
    "DatasetSplit",
    "TransformationMetadata",
    "TransformationSummary",
    # Exception hierarchy
    "PreprocessingError",
    "SplitError",
    "TransformationError",
    "ArtifactError",
    "PipelineError",
]
