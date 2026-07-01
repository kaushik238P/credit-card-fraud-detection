"""
Custom exception hierarchy for the Evaluation Layer.

Hierarchy:
    Exception
    └── EvaluationError
        ├── PipelineError     — guard or orchestrator failure (fatal)
        ├── MetricError       — metric calculation failure (fatal)
        ├── ThresholdError    — threshold optimization failure (fatal)
        ├── PlotError         — plot generation failure (non-fatal, yields warning)
        └── ReportError       — report generation failure (fatal)
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Base Exception
# ---------------------------------------------------------------------------


class EvaluationError(Exception):
    """
    Base exception class for all errors in the Evaluation Layer.

    Attributes:
        message: Human-readable explanation of the error.
        context: Optional diagnostic key-value pairs.
    """

    def __init__(self, message: str, context: dict | None = None) -> None:
        self.message = message
        self.context = context or {}
        super().__init__(self._format())

    def _format(self) -> str:
        if not self.context:
            return self.message
        ctx = " | ".join(f"{k}={v!r}" for k, v in self.context.items())
        return f"{self.message} [Context: {ctx}]"

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"message={self.message!r}, "
            f"context={self.context!r})"
        )


# ---------------------------------------------------------------------------
# Concrete Exceptions
# ---------------------------------------------------------------------------


class PipelineError(EvaluationError):
    """
    Raised when the evaluation pipeline configuration or input validation fails.

    Always propagates and terminates pipeline execution.
    """

    def __init__(self, stage: str, detail: str, context: dict | None = None) -> None:
        self.stage = stage
        ctx = {"stage": stage, **(context or {})}
        super().__init__(
            message=f"Evaluation pipeline failed at stage '{stage}': {detail}",
            context=ctx,
        )


class MetricError(EvaluationError):
    """
    Raised when a critical metric calculation fails.

    Always propagates and terminates pipeline execution.
    """

    def __init__(self, metric_name: str, detail: str, context: dict | None = None) -> None:
        self.metric_name = metric_name
        ctx = {"metric_name": metric_name, **(context or {})}
        super().__init__(
            message=f"Failed to compute metric '{metric_name}': {detail}",
            context=ctx,
        )


class ThresholdError(EvaluationError):
    """
    Raised when threshold optimization fails.

    Always propagates and terminates pipeline execution.
    """

    def __init__(self, strategy: str, detail: str, context: dict | None = None) -> None:
        self.strategy = strategy
        ctx = {"strategy": strategy, **(context or {})}
        super().__init__(
            message=f"Threshold optimization strategy '{strategy}' failed: {detail}",
            context=ctx,
        )


class PlotError(EvaluationError):
    """
    Raised when a diagnostic plot cannot be generated or saved.

    Non-fatal: caught by the pipeline, turned into a warning, and execution continues.
    """

    def __init__(self, plot_name: str, detail: str, context: dict | None = None) -> None:
        self.plot_name = plot_name
        ctx = {"plot_name": plot_name, **(context or {})}
        super().__init__(
            message=f"Failed to generate plot '{plot_name}': {detail}",
            context=ctx,
        )


class ReportError(EvaluationError):
    """
    Raised when report generation or compilation fails.

    Always propagates and terminates pipeline execution.
    """

    def __init__(self, report_type: str, detail: str, context: dict | None = None) -> None:
        self.report_type = report_type
        ctx = {"report_type": report_type, **(context or {})}
        super().__init__(
            message=f"Failed to generate '{report_type}' report: {detail}",
            context=ctx,
        )
