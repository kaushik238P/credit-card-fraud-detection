"""
FastAPI dependency injection provider functions.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from src.api.config import inference_settings
from src.api.models.inference_context import InferenceContext
from src.api.services.model_provider import ModelProvider
from src.api.services.inference_pipeline import InferencePipeline

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_model_provider() -> ModelProvider:
    """Returns the singleton ModelProvider instance."""
    return ModelProvider()


def get_inference_context(
    provider: ModelProvider = Depends(get_model_provider),
) -> InferenceContext:
    """Returns the preloaded InferenceContext."""
    return provider.context


def get_inference_pipeline(
    context: InferenceContext = Depends(get_inference_context),
) -> InferencePipeline:
    """Returns a fresh InferencePipeline initialized with the active context."""
    return InferencePipeline(context=context)


async def verify_api_key(
    api_key: str | None = Security(API_KEY_HEADER),
) -> str | None:
    """
    Validates the incoming X-API-Key header against configuration settings.
    
    Args:
        api_key: Incoming API key header value.
        
    Returns:
        str | None: Verified key string.
        
    Raises:
        HTTPException: If key is invalid or missing.
    """
    expected_key = inference_settings.api_key
    if expected_key is None:
        # API Key auth is disabled
        return None
        
    if not api_key or api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API credential key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return api_key
