"""
Custom exception hierarchy for the EDA Layer.

Hierarchy:
    Exception
    └── EDAError
        ├── AnalysisError
        ├── VisualizationError
        └── ReportGenerationError
"""

from __future__ import annotations


class EDAError(Exception):
    """
    Base exception for all EDA layer failures. Never raise directly.

    Attributes:
        message: Human-readable description of the failure.
        context: JSON-serialisable diagnostic key-value pairs.
    """

    def __init__(self, message: str, context: dict | None = None) -> None:
        self.message = message
        self.context: dict = context or {}
        super().__init__(self._format())

    def _format(self) -> str:
        if not self.context:
            return self.message
        ctx = " | ".join(f"{k}={v!r}" for k, v in self.context.items())
        return f"{self.message} [Context: {ctx}]"

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"message={self.message!r}, context={self.context!r})"
        )


class AnalysisError(EDAError):
    """
    Raised when an analysis module cannot produce a meaningful result.

    Raised by: individual _run_* methods in EDAAnalyzer when:
        - A required column for that analysis is absent.
        - A computation fails due to unexpected data conditions.

    Recovery: EDAAnalyzer catches this per-module, sets that analysis field
    to None in EDAReport, and continues with remaining modules.
    """


class VisualizationError(EDAError):
    """
    Raised when a figure cannot be rendered or saved to disk.

    Raised by: EDAVisualizer plot methods when:
        - matplotlib/seaborn raises an error during construction.
        - The output directory is not writable.
        - A required column for the plot is absent.

    Recovery: EDAVisualizer catches this per figure, skips that figure,
    and continues generating the rest.
    """


class ReportGenerationError(EDAError):
    """
    Raised when ReportExporter cannot serialise or write the EDAReport.

    Raised by: ReportExporter when:
        - The output directory is not writable.
        - An EDAReport field is not JSON-serialisable.
        - A file I/O error occurs during write.

    Recovery: Non-recoverable — re-raised to the caller.
    """
