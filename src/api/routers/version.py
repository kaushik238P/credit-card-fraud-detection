"""
FastAPI router for version information endpoint.
"""

from __future__ import annotations

import os
import subprocess
import mlflow
from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field

from src.api.dependencies.providers import get_inference_context
from src.api.models.inference_context import InferenceContext

router = APIRouter(tags=["System Info"])


class VersionResponse(BaseModel):
    """Payload details of the API system and pipeline lineage versions."""
    api_version: str = Field(..., description="FastAPI service code version.")
    pipeline_version: str = Field(..., description="Training pipeline code version.")
    model_version: str = Field(..., description="Timestamp version of the loaded model.")
    mlflow_version: str = Field(..., description="Installed MLflow package version.")
    git_commit: str | None = Field(..., description="Active git hash key.")
    dataset_hash: str = Field(..., description="SHA-256 hash of the training data.")


def get_git_commit() -> str | None:
    """Helper to fetch git commit hash of the workspace repository."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return res.stdout.strip()[:8]
    except Exception:
        # Fallback if git is not installed or repo is detached
        return os.environ.get("FRAUD_GIT_COMMIT") or None


@router.get(
    "/version",
    response_model=VersionResponse,
    status_code=status.HTTP_200_OK,
    summary="Get service and model versions",
)
async def get_version_info(
    context: InferenceContext = Depends(get_inference_context),
) -> VersionResponse:
    """
    Returns running service package versions, git commit keys, and dataset hash lineages.
    """
    import os
    git_hash = get_git_commit()
    
    return VersionResponse(
        api_version="1.0.0",
        pipeline_version=context.training_version,
        model_version=context.model_version,
        mlflow_version=mlflow.__version__,
        git_commit=git_hash,
        dataset_hash=context.dataset_hash,
    )
