"""
Domain-specific exceptions for the FastAPI Inference Layer.
"""

from __future__ import annotations


class InferenceDomainError(Exception):
    """Base exception for all domain-specific inference errors."""

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or message


class ModelNotLoaded(InferenceDomainError):
    """Raised when an inference request is received but the model has not been loaded."""

    def __init__(self, message: str = "Model has not been loaded in memory.", detail: str | None = None) -> None:
        super().__init__(message, detail)


class ArtifactMismatch(InferenceDomainError):
    """Raised when loaded model artifact version, schema, or config hash fails startup validation."""

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(message, detail)


class TransformationError(InferenceDomainError):
    """Raised when feature engineering or preprocessing transformations fail during inference."""

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(message, detail)


class PredictionError(InferenceDomainError):
    """Raised when CatBoost estimator prediction raises an unexpected runtime exception."""

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(message, detail)


class MLflowUnavailable(InferenceDomainError):
    """Raised when tracking server registry lookup fails and offline fallback is disabled."""

    def __init__(self, message: str = "MLflow Tracking Server is unreachable.", detail: str | None = None) -> None:
        super().__init__(message, detail)


class ConfigurationError(InferenceDomainError):
    """Raised when settings or configurations contain errors preventing startup."""

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(message, detail)
