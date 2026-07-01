"""
Configuration module for the FastAPI Inference Layer.

Bridges config.settings values and environment overrides to API-specific options.
"""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path

from config.settings import settings


class InferenceSettings:
    """
    Exposes all configuration settings for the FastAPI inference service.
    
    Reads from environment variables and configures paths from config.settings dynamically.
    """

    @property
    def model_name(self) -> str:
        """Name of the registered model in MLflow."""
        return os.environ.get("FRAUD_MLFLOW_REGISTRY_NAME", "credit_card_fraud_detection")

    @property
    def model_stage(self) -> str:
        """Stage of the target model to load (e.g. Production)."""
        return os.environ.get("FRAUD_MLFLOW_REGISTRY_STAGE", "Production")

    @property
    def tracking_uri(self) -> str:
        """MLflow Tracking URI backend connection."""
        return (
            os.environ.get("FRAUD_MLFLOW_TRACKING_URI")
            or os.environ.get("MLFLOW_TRACKING_URI")
            or "sqlite:///mlflow.db"
        )

    @property
    def offline_mode(self) -> bool:
        """Whether to bypass MLflow and use local artifacts directly."""
        val = os.environ.get("FRAUD_INFERENCE_OFFLINE_MODE", "0")
        return val == "1" or val.lower() == "true"

    @property
    def local_model_dir(self) -> Path:
        """Path to local directory holding latest model assets."""
        return Path(settings.training.artifact_dir) / "latest"

    @property
    def local_preprocessing_dir(self) -> Path:
        """Path to local directory holding latest preprocessing assets."""
        return Path(settings.preprocessing.artifact_dir) / "latest"

    @property
    def evaluation_report_dir(self) -> Path:
        """Path to local directory holding evaluation outputs."""
        return Path(settings.evaluation.output_directory) / "latest"

    @property
    def host(self) -> str:
        """API server listen host address."""
        return os.environ.get("FRAUD_API_HOST", "0.0.0.0")

    @property
    def port(self) -> int:
        """API server listen port."""
        return int(os.environ.get("FRAUD_API_PORT", "8000"))

    @property
    def workers(self) -> int:
        """Number of Gunicorn/Uvicorn worker processes."""
        val = os.environ.get("FRAUD_API_WORKERS")
        if val:
            return int(val)
        return max(1, multiprocessing.cpu_count())

    @property
    def api_key(self) -> str | None:
        """Simple API Token key required for access validation."""
        return os.environ.get("FRAUD_API_KEY") or None

    @property
    def cors_origins(self) -> list[str]:
        """Allowed Origins list for CORS middleware configuration."""
        origins = os.environ.get("FRAUD_API_CORS_ORIGINS", "*")
        return [o.strip() for o in origins.split(",") if o.strip()]

    @property
    def trusted_hosts(self) -> list[str]:
        """Allowed host header patterns for TrustedHostMiddleware."""
        hosts = os.environ.get("FRAUD_API_TRUSTED_HOSTS", "*")
        return [h.strip() for h in hosts.split(",") if h.strip()]

    @property
    def max_content_size_bytes(self) -> int:
        """Maximum allowed request size limit in bytes."""
        return int(
            os.environ.get("FRAUD_API_MAX_PAYLOAD_SIZE_BYTES", str(10 * 1024 * 1024))
        )


inference_settings = InferenceSettings()
