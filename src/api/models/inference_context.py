"""
InferenceContext model class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class InferenceContext:
    """
    Immutable dataclass container holding all cached model and preprocessing assets.
    
    Loaded once at startup and passed down endpoints via dependency injection.
    """
    model: Any
    threshold: float
    feature_schema: dict[str, Any]
    preprocessing_pipeline: Any
    feature_engineering_pipeline: Any
    metadata: dict[str, Any]
    model_version: str
    dataset_hash: str
    training_version: str
    preprocessing_version: str
    feature_engineering_version: str
    mlflow_run_id: str
    mlflow_model_version: int | str
