"""
Custom exception hierarchy for the Preprocessing Layer.

Hierarchy:
    Exception
    └── PreprocessingError
        ├── SplitError          — terminates pipeline
        ├── TransformationError — caught; warning appended; continues
        ├── ArtifactError       — caught; warning appended; continues
        └── PipelineError       — terminates pipeline
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class PreprocessingError(Exception):
    """
    Base exception for all preprocessing layer failures.

    Never raise this directly. Use one of the concrete subclasses.

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
            f"message={self.message!r}, "
            f"context={self.context!r})"
        )


# ---------------------------------------------------------------------------
# Concrete exceptions
# ---------------------------------------------------------------------------


class SplitError(PreprocessingError):
    """
    Raised when time-based splitting fails. Always propagates.

    Raised when:
        - The datetime column is absent or entirely unparseable.
        - Any split fold (train, validation, or test) is empty after filtering.
        - Config date boundaries are logically invalid (end <= start).

    An empty fold makes training meaningless — the pipeline must not proceed.

    Attributes:
        fold_name: Name of the fold that triggered the error, if applicable.
        date_range: The configured date range that produced no rows.
    """

    def __init__(
        self,
        detail: str,
        fold_name: str = "",
        date_range: tuple[str, str] | None = None,
    ) -> None:
        self.fold_name = fold_name
        self.date_range = date_range

        super().__init__(
            message=f"Time-based split failed: {detail}",
            context={
                "fold_name": fold_name,
                "date_range": list(date_range) if date_range else None,
            },
        )


class TransformationError(PreprocessingError):
    """
    Raised when a transformer's fit() or transform() call fails.

    Caught by the pipeline per-transformer. A warning is appended and
    the pipeline continues with the next step.

    Raised when:
        - A required column is not found in the DataFrame.
        - A fitted transformer receives an incompatible column shape.
        - An unexpected internal error occurs during transformation.

    Attributes:
        transformer_name: Name of the transformer that failed.
        step_name: Registry step name where the failure occurred.
        original_exc: The underlying exception, if any.
    """

    def __init__(
        self,
        transformer_name: str,
        step_name: str,
        detail: str,
        original_exc: Exception | None = None,
    ) -> None:
        self.transformer_name = transformer_name
        self.step_name = step_name
        self.original_exc = original_exc

        super().__init__(
            message=(
                f"Transformer '{transformer_name}' failed at step '{step_name}': {detail}"
            ),
            context={
                "transformer_name": transformer_name,
                "step_name": step_name,
                "detail": detail,
                "original_exc": str(original_exc) if original_exc else None,
            },
        )


class ArtifactError(PreprocessingError):
    """
    Raised when saving or loading a preprocessing artifact fails.

    Caught by the pipeline per-artifact. A warning is appended and the
    pipeline continues. This allows training to proceed even if disk writes
    fail (e.g., in memory-only CI environments).

    Raised when:
        - The output directory is not writable.
        - Joblib serialisation fails for a transformer.
        - A JSON write or Parquet write fails.
        - A requested artifact file is not found during load.

    Attributes:
        artifact_name: Name of the artifact being saved or loaded.
        artifact_path: File path where the artifact was expected.
    """

    def __init__(
        self,
        artifact_name: str,
        artifact_path: str,
        detail: str,
        original_exc: Exception | None = None,
    ) -> None:
        self.artifact_name = artifact_name
        self.artifact_path = artifact_path
        self.original_exc = original_exc

        super().__init__(
            message=(
                f"Artifact operation failed for '{artifact_name}' "
                f"at path '{artifact_path}': {detail}"
            ),
            context={
                "artifact_name": artifact_name,
                "artifact_path": artifact_path,
                "detail": detail,
                "original_exc": str(original_exc) if original_exc else None,
            },
        )


class PipelineError(PreprocessingError):
    """
    Raised when the pipeline cannot proceed at all. Always propagates.

    Raised when:
        - The input EngineeredDataset is None or its report is missing.
        - The target column is absent from the engineered DataFrame.
        - The PREPROCESSING_REGISTRY is empty or invalid.
        - A guard condition fails before any processing begins.

    Attributes:
        stage: Pipeline stage name where the failure occurred.
    """

    def __init__(
        self,
        stage: str,
        detail: str,
        context: dict | None = None,
    ) -> None:
        self.stage = stage

        super().__init__(
            message=f"Pipeline failed at stage '{stage}': {detail}",
            context={"stage": stage, **(context or {})},
        )
