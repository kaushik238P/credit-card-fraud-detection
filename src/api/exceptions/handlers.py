"""
Global exception handlers mapping custom domain errors to RFC 7807 problem details responses.
"""

from config.logging import get_logger
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.api.exceptions.custom import (
    InferenceDomainError,
    ModelNotLoaded,
    ArtifactMismatch,
    TransformationError,
    PredictionError,
    MLflowUnavailable,
    ConfigurationError,
)

logger = get_logger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    """
    Registers global exception mapping handlers to the FastAPI app instance.
    
    Args:
        app: Target FastAPI application.
    """

    @app.exception_handler(ModelNotLoaded)
    async def model_not_loaded_handler(request: Request, exc: ModelNotLoaded) -> JSONResponse:
        return build_rfc7807_response(
            type_url="https://errors.fraud-api.local/model-not-loaded",
            title="Model Not Loaded",
            status=503,
            detail=exc.detail,
            request=request,
        )

    @app.exception_handler(ArtifactMismatch)
    async def artifact_mismatch_handler(request: Request, exc: ArtifactMismatch) -> JSONResponse:
        return build_rfc7807_response(
            type_url="https://errors.fraud-api.local/artifact-mismatch",
            title="Artifact Mismatch",
            status=500,
            detail=exc.detail,
            request=request,
        )

    @app.exception_handler(TransformationError)
    async def transformation_error_handler(request: Request, exc: TransformationError) -> JSONResponse:
        return build_rfc7807_response(
            type_url="https://errors.fraud-api.local/transformation-error",
            title="Transformation Error",
            status=422,
            detail=exc.detail,
            request=request,
        )

    @app.exception_handler(PredictionError)
    async def prediction_error_handler(request: Request, exc: PredictionError) -> JSONResponse:
        return build_rfc7807_response(
            type_url="https://errors.fraud-api.local/prediction-error",
            title="Prediction Error",
            status=500,
            detail=exc.detail,
            request=request,
        )

    @app.exception_handler(MLflowUnavailable)
    async def mlflow_unavailable_handler(request: Request, exc: MLflowUnavailable) -> JSONResponse:
        return build_rfc7807_response(
            type_url="https://errors.fraud-api.local/mlflow-unavailable",
            title="MLflow Server Unavailable",
            status=503,
            detail=exc.detail,
            request=request,
        )

    @app.exception_handler(ConfigurationError)
    async def configuration_error_handler(request: Request, exc: ConfigurationError) -> JSONResponse:
        return build_rfc7807_response(
            type_url="https://errors.fraud-api.local/configuration-error",
            title="Configuration Error",
            status=500,
            detail=exc.detail,
            request=request,
        )

    @app.exception_handler(InferenceDomainError)
    async def base_domain_error_handler(request: Request, exc: InferenceDomainError) -> JSONResponse:
        return build_rfc7807_response(
            type_url="https://errors.fraud-api.local/internal-domain-error",
            title="Internal Domain Error",
            status=500,
            detail=exc.detail,
            request=request,
        )


def build_rfc7807_response(
    type_url: str,
    title: str,
    status: int,
    detail: str,
    request: Request,
) -> JSONResponse:
    """
    Constructs a JSONResponse compliant with the RFC 7807 problem details specification.
    
    Args:
        type_url: Unique URI identifying the type of issue.
        title: Short description of the error category.
        status: HTTP status code.
        detail: Description of this occurrence.
        request: FastAPI request object.
        
    Returns:
        JSONResponse: Configured error response payload.
    """
    request_id = request.headers.get("X-Request-ID", "unknown")
    payload = {
        "type": type_url,
        "title": title,
        "status": status,
        "detail": detail,
        "instance": str(request.url),
        "request_id": request_id,
    }
    logger.error("RFC 7807 Error | status=%d | title=%s | detail=%s | request_id=%s", status, title, detail, request_id)
    return JSONResponse(
        status_code=status,
        content=payload,
        headers={"Content-Type": "application/problem+json"},
    )
