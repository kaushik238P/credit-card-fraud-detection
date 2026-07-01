"""
Pydantic schemas for model information and metadata endpoints.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class ModelFeaturesResponse(BaseModel):
    """Payload details of required inference features and encoders mapping."""
    features: list[str] = Field(..., description="Expected ordered feature names.")
    count: int = Field(..., description="Number of expected features.")
    types: dict[str, str] = Field(..., description="Data types mapping of features.")
    encoded_features: dict[str, list[str]] = Field(..., description="Encoding details for columns.")


class ModelInfoResponse(BaseModel):
    """Payload representing model details and hyperparameter settings."""
    model_type: str = Field(..., description="Machine learning algorithm category (e.g. CATBOOST).")
    model_version: str = Field(..., description="Timestamp version descriptor of the active run.")
    training_version: str = Field(..., description="Version of the training pipeline code.")
    hyperparameters: dict[str, Any] = Field(..., description="Estimator hyperparameter configurations.")
    random_seed: int = Field(..., description="Global random seed.")
    imbalance_strategy: str = Field(..., description="Imbalance strategy applied.")
    imbalance_ratio: float = Field(..., description="Imbalance ratio in training data.")
    feature_count: int = Field(..., description="Number of features consumed by model.")


class MetadataResponse(BaseModel):
    """Payload representing full merged preprocessing, training, and registry lineage data."""
    dataset_hash: str = Field(..., description="SHA-256 hash of the preprocessed dataset.")
    feature_schema_hash: str = Field(..., description="SHA-256 hash of the feature schema.")
    preprocessing_version: str = Field(..., description="Preprocessing module code version.")
    training_version: str = Field(..., description="Training module code version.")
    preprocessing_artifact_version: str = Field(..., description="Version identifier of preprocessing run.")
    mlflow_run_id: str = Field(..., description="MLflow run ID associated with the model.")
    mlflow_model_version: int | str = Field(..., description="MLflow model registry version.")
    threshold: float = Field(..., description="Optimized decision threshold.")
    threshold_strategy: str = Field(..., description="Strategy used to optimize threshold (e.g. MAX_F1).")
