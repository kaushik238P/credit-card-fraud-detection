"""
Custom exception hierarchy for the Feature Engineering Layer.

Hierarchy:
    Exception
    └── FeatureEngineeringError
        ├── TransformationError
        ├── FeatureCreationError
        └── PipelineError
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class FeatureEngineeringError(Exception):
    """
    Base exception for all feature engineering layer failures.

    Never raise this class directly. Use one of the concrete subclasses
    so callers can distinguish between failure types.

    Attributes:
        message: Human-readable description of the failure.
        context: Serialisable key-value pairs with diagnostic information.
            Always JSON-serialisable (no DataFrames, no numpy objects).
    """

    def __init__(self, message: str, context: dict | None = None) -> None:
        self.message = message
        self.context: dict = context or {}
        super().__init__(self._format())

    def _format(self) -> str:
        """Produces the string passed to Exception.__init__."""
        if not self.context:
            return self.message
        ctx = " | ".join(f"{k}={v!r}" for k, v in self.context.items())
        return f"{self.message} [Context: {ctx}]"

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"message={self.message!r}, context={self.context!r})"
        )


# ---------------------------------------------------------------------------
# Concrete exceptions
# ---------------------------------------------------------------------------


class TransformationError(FeatureEngineeringError):
    """
    Raised when a single feature computation fails on bad or missing data.

    Raised by feature modules (temporal, amount, geographic, etc.) when:
        - A required source column is absent.
        - A vectorised operation raises an unexpected exception.
        - Parsed values (e.g., dates) are entirely NaN after coercion.

    The pipeline orchestrator catches this exception, logs a warning,
    appends it to the report, and continues to the next module.

    Attributes:
        feature_name: Name of the feature being computed when the error
            occurred.
        source_columns: Column names that were being consumed.
        original_exc: The underlying exception that was caught, if any.
    """

    def __init__(
        self,
        feature_name: str,
        source_columns: list[str],
        detail: str = "",
        original_exc: Exception | None = None,
    ) -> None:
        self.feature_name = feature_name
        self.source_columns = source_columns
        self.original_exc = original_exc

        msg = (
            f"Transformation failed for feature '{feature_name}' "
            f"using source columns {source_columns}."
        )
        if detail:
            msg = f"{msg} {detail}"

        super().__init__(
            message=msg,
            context={
                "feature_name": feature_name,
                "source_columns": source_columns,
                "detail": detail,
                "original_exc": str(original_exc) if original_exc else None,
            },
        )


class FeatureCreationError(FeatureEngineeringError):
    """
    Raised when a feature module produces incorrect output.

    Raised by the pipeline orchestrator when post-creation assertions fail:
        - Returned DataFrame has unexpected column names.
        - Returned DataFrame has a different row count from the input.
        - Column dtype does not match the declared FeatureDefinition.

    Attributes:
        module_name: Name of the feature module that produced the bad output.
        expected: Description of what was expected.
        actual: Description of what was actually received.
    """

    def __init__(
        self,
        module_name: str,
        expected: str,
        actual: str,
    ) -> None:
        self.module_name = module_name
        self.expected = expected
        self.actual = actual

        super().__init__(
            message=(
                f"Feature module '{module_name}' produced invalid output. "
                f"Expected: {expected}. Actual: {actual}."
            ),
            context={
                "module_name": module_name,
                "expected": expected,
                "actual": actual,
            },
        )


class PipelineError(FeatureEngineeringError):
    """
    Raised when the pipeline cannot proceed at all.

    Raised by FeatureEngineeringPipeline.engineer() when:
        - ValidatedDataset.passed is False.
        - EDAReport is None or does not contain feature_inventory.
        - The feature registry is empty or misconfigured.

    Unlike TransformationError and FeatureCreationError, a PipelineError
    is NOT caught internally — it propagates to the caller immediately.

    Attributes:
        stage: Name of the pipeline stage where the failure occurred.
    """

    def __init__(
        self,
        stage: str,
        detail: str,
        context: dict | None = None,
    ) -> None:
        self.stage = stage

        super().__init__(
            message=(
                f"Pipeline failed at stage '{stage}': {detail}"
            ),
            context={"stage": stage, **(context or {})},
        )
