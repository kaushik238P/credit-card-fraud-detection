"""
Custom exceptions for the MLflow Layer.

Includes the base exception and specialized exceptions for tracking,
artifact upload, model registration, and orchestration pipeline faults.
"""

from __future__ import annotations

from typing import Any


class MLflowError(Exception):
    """
    Base exception class for all errors arising within the MLflow Layer.
    """

    def __init__(
        self,
        message: str,
        code: str = "MLFLOW_ERROR",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.context = context or {}


class TrackingError(MLflowError):
    """
    Raised when operations on the MLflow tracking server (e.g. logging metrics,
    starting/ending runs) fail.
    """

    def __init__(
        self,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code="TRACKING_ERROR", context=context)


class ArtifactUploadError(MLflowError):
    """
    Raised when upload of structural artifacts (such as estimator binaries,
    validation curves, or manifest sheets) encounters a network or filesystem fault.
    """

    def __init__(
        self,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code="ARTIFACT_UPLOAD_ERROR", context=context)


class RegistrationError(MLflowError):
    """
    Raised when model registration or version transition operations
    fail on the model registry database.
    """

    def __init__(
        self,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code="REGISTRATION_ERROR", context=context)


class PipelineError(MLflowError):
    """
    Raised when high-level orchestration pipeline guards or validation rules fail.
    """

    def __init__(
        self,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code="PIPELINE_ERROR", context=context)
