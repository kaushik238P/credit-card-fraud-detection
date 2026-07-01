"""
EDA Layer public API for the Credit Card Fraud Detection System.

Public surface area:
    EDAAnalyzer             -- primary entry point for running EDA
    EDAVisualizer           -- generates and saves figures
    InsightEngine           -- generates engineering recommendations
    ReportExporter          -- exports EDAReport to Markdown/JSON
    EDAReport               -- complete output of the EDA layer
    EDARecommendation       -- single actionable recommendation
    EDAError                -- base exception class
    AnalysisError           -- raised on module analysis failure
    VisualizationError      -- raised on figure rendering failure
    ReportGenerationError   -- raised on report export failure
"""

from src.eda.exceptions import (
    AnalysisError,
    EDAError,
    ReportGenerationError,
    VisualizationError,
)
from src.eda.models import (
    CategoricalAnalysis,
    CorrelationAnalysis,
    DatasetOverview,
    EDARecommendation,
    EDAReport,
    GeographicAnalysis,
    NumericalAnalysis,
    TargetAnalysis,
    TemporalAnalysis,
)
from src.eda.insights import InsightEngine
from src.eda.visualizer import EDAVisualizer
from src.eda.report import ReportExporter
from src.eda.analyzer import EDAAnalyzer

__all__ = [
    # Orchestrator
    "EDAAnalyzer",
    # Sub-engines
    "EDAVisualizer",
    "InsightEngine",
    # Export
    "ReportExporter",
    # Exception Hierarchy
    "EDAError",
    "AnalysisError",
    "VisualizationError",
    "ReportGenerationError",
    # Dataclasses
    "DatasetOverview",
    "TargetAnalysis",
    "NumericalAnalysis",
    "CategoricalAnalysis",
    "TemporalAnalysis",
    "GeographicAnalysis",
    "CorrelationAnalysis",
    "EDARecommendation",
    "EDAReport",
]
