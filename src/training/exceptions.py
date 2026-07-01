"""
Custom exception hierarchy for the Training Layer.

Hierarchy:
    Exception
    └── TrainingError
        ├── ModelCreationError    — terminates pipeline
        ├── TrainingFailureError  — terminates pipeline
        ├── SerializationError    — caught; warning appended; continues
        └── PipelineError        — terminates pipeline

Propagation contract:
    PipelineError        → always re-raised
    ModelCreationError   → always re-raised
    TrainingFailureError → always re-raised
    SerializationError   → caught per-artifact; warning appended; pipeline continues
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class TrainingError(Exception):
    """
    Base exception for all Training Layer failures.

    Never raise directly. Use a concrete subclass.

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


class ModelCreationError(TrainingError):
    """
    Raised when ModelFactory fails to instantiate an estimator.

    Always propagates — training cannot proceed without a valid model.

    Raised when:
        - The requested ModelType is not in MODEL_REGISTRY.
        - A required library (xgboost, lightgbm, catboost) is not installed.
        - Invalid hyperparameters are passed to the estimator constructor.
        - The active_model config value maps to a disabled registry entry.

    Attributes:
        model_type: String name of the ModelType that failed.
        original_exc: The underlying exception from the estimator library.
    """

    def __init__(
        self,
        model_type: str,
        detail: str,
        original_exc: Exception | None = None,
    ) -> None:
        self.model_type = model_type
        self.original_exc = original_exc
        super().__init__(
            message=f"Failed to create model '{model_type}': {detail}",
            context={
                "model_type": model_type,
                "detail": detail,
                "original_exc": str(original_exc) if original_exc else None,
            },
        )


class TrainingFailureError(TrainingError):
    """
    Raised when estimator.fit() raises during training.

    Always propagates — a partially fitted model must not be serialised.

    Raised when:
        - estimator.fit() raises any exception.
        - The fitted estimator fails interface validation (missing predict()).
        - Memory is exhausted during training.

    Attributes:
        model_type: String name of the ModelType being trained.
        training_rows: Number of rows in the training set.
        original_exc: The underlying exception from fit().
    """

    def __init__(
        self,
        model_type: str,
        training_rows: int,
        detail: str,
        original_exc: Exception | None = None,
    ) -> None:
        self.model_type = model_type
        self.training_rows = training_rows
        self.original_exc = original_exc
        super().__init__(
            message=(
                f"Training failed for model '{model_type}' "
                f"on {training_rows:,} rows: {detail}"
            ),
            context={
                "model_type": model_type,
                "training_rows": training_rows,
                "detail": detail,
                "original_exc": str(original_exc) if original_exc else None,
            },
        )


class SerializationError(TrainingError):
    """
    Raised when a model artifact save or load operation fails.

    Caught per-artifact by the pipeline. A warning is appended and the
    pipeline continues returning the TrainingResult in memory.

    Raised when:
        - The artifact directory is not writable.
        - joblib.dump() fails for the fitted estimator.
        - A JSON write fails for metadata, report, or feature names.
        - The manifest file cannot be written.
        - latest/ sync fails.

    Attributes:
        artifact_name: Name of the artifact that failed (e.g. "model.joblib").
        artifact_path: File path where the write was attempted.
        original_exc: The underlying OS or joblib exception.
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
                f"Serialization failed for '{artifact_name}' "
                f"at '{artifact_path}': {detail}"
            ),
            context={
                "artifact_name": artifact_name,
                "artifact_path": artifact_path,
                "detail": detail,
                "original_exc": str(original_exc) if original_exc else None,
            },
        )


class PipelineError(TrainingError):
    """
    Raised when the pipeline cannot proceed at all. Always propagates.

    Raised when:
        - PreprocessedDataset is None or X_train is empty.
        - feature_names is empty.
        - active_model does not map to any enabled registry entry.
        - MODEL_REGISTRY is empty.
        - A structural invariant is violated before processing begins.

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
            message=f"Training pipeline failed at stage '{stage}': {detail}",
            context={"stage": stage, **(context or {})},
        )
