"""
FastAPI router for health endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, status

from src.api.dependencies.providers import get_model_provider
from src.api.schemas.health import LivenessResponse, ReadinessResponse
from src.api.services.model_provider import ModelProvider

router = APIRouter(prefix="/health", tags=["Health"])


@router.get(
    "/liveness",
    response_model=LivenessResponse,
    status_code=status.HTTP_200_OK,
    summary="Liveness check",
)
async def check_liveness() -> LivenessResponse:
    """
    Verifies that the API server process is active and running.
    
    Always returns 200 OK with a healthy status.
    """
    return LivenessResponse(status="healthy")


@router.get(
    "/readiness",
    response_model=ReadinessResponse,
    status_code=status.HTTP_200_OK,
    summary="Readiness check",
)
async def check_readiness(
    provider: ModelProvider = Depends(get_model_provider),
) -> ReadinessResponse:
    """
    Checks that the model and encoders are fully preloaded in RAM.
    
    Returns 200 OK if ready, 503 Service Unavailable otherwise.
    """
    try:
        context = provider.context
        model_loaded = context.model is not None
        pipeline_ready = context.preprocessing_pipeline is not None
        threshold_loaded = context.threshold > 0.0
        
        is_ready = model_loaded and pipeline_ready and threshold_loaded
        
        return ReadinessResponse(
            status="ready" if is_ready else "not_ready",
            model_loaded=model_loaded,
            pipeline_ready=pipeline_ready,
            threshold_loaded=threshold_loaded,
        )
    except Exception:
        return ReadinessResponse(
            status="not_ready",
            model_loaded=False,
            pipeline_ready=False,
            threshold_loaded=False,
        )
