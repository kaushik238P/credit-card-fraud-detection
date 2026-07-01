"""
Evaluation Layer — public API surface.

Usage:
    from src.evaluation import EvaluationPipeline, EvaluationResult

    pipeline = EvaluationPipeline()
    result: EvaluationResult = pipeline.evaluate(training_result, preprocessed_dataset)
"""

from src.evaluation.exceptions import (
    EvaluationError,
    MetricError,
    PipelineError,
    PlotError,
    ReportError,
    ThresholdError,
)
from src.evaluation.models import (
    BusinessMetric,
    EvaluationArtifacts,
    EvaluationReport,
    EvaluationResult,
    MetricResult,
    PlotArtifact,
    ThresholdResult,
)
from src.evaluation.pipeline import EvaluationPipeline

__all__ = [
    # Pipeline
    "EvaluationPipeline",
    # Models
    "EvaluationResult",
    "EvaluationReport",
    "EvaluationArtifacts",
    "MetricResult",
    "BusinessMetric",
    "ThresholdResult",
    "PlotArtifact",
    # Exceptions
    "EvaluationError",
    "PipelineError",
    "MetricError",
    "ThresholdError",
    "PlotError",
    "ReportError",
]
