"""
Pydantic schemas for the health check endpoints.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class LivenessResponse(BaseModel):
    """Payload representing status of API server process execution."""
    status: str = Field(default="healthy", description="Status code indicating liveness.")


class ReadinessResponse(BaseModel):
    """Payload representing status of database and model memory preload connections."""
    status: str = Field(..., description="Overall readiness status.")
    model_loaded: bool = Field(..., description="Whether the CatBoost model is loaded in memory.")
    pipeline_ready: bool = Field(..., description="Whether the preprocessing pipeline is fully configured.")
    threshold_loaded: bool = Field(..., description="Whether the classification threshold is loaded.")
