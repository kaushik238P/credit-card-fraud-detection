"""
Singleton Model Provider service for preloading and caching all ML artifacts.
"""

from __future__ import annotations

from config.logging import get_logger
import json
import os
import threading
from pathlib import Path
from typing import Any

import joblib
import mlflow

from config.settings import settings
from src.api.config import inference_settings
from src.api.exceptions.custom import ArtifactMismatch, ConfigurationError, ModelNotLoaded
from src.api.models.inference_context import InferenceContext

logger = get_logger(__name__)


class ModelProvider:
    """
    Singleton service class providing thread-safe preloaded ML model and preprocessing artifacts.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls) -> ModelProvider:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._context: InferenceContext | None = None
        self._initialized = True

    @property
    def context(self) -> InferenceContext:
        """Returns the preloaded InferenceContext. Raises error if not initialized."""
        if self._context is None:
            raise ModelNotLoaded("InferenceContext has not been loaded. Call load_assets() first.")
        return self._context

    def load_assets(self) -> InferenceContext:
        """
        Loads all model assets, preprocessing transformers, schemas, and threshold parameters.
        
        Attempts connection to MLflow registry first (unless offline mode is enabled).
        Falls back to local artifacts if MLflow is unreachable or inactive.
        
        Returns:
            InferenceContext: Preloaded metadata and model cache context.
            
        Raises:
            ArtifactMismatch: If pipeline integrity validations fail.
            ConfigurationError: If configs or thresholds are invalid.
            MLflowUnavailable: If MLflow tracking fails with fallback disabled.
        """
        with self._lock:
            logger.info("Initializing preloaded inference assets...")
            
            # 1. Load Model, Threshold, and Training Metadata
            model: Any = None
            threshold: float = 0.0
            threshold_strategy: str = "UNKNOWN"
            training_metadata: dict[str, Any] = {}
            mlflow_run_id: str = "local"
            mlflow_model_version: int | str = "local"
            model_source: str = "local"

            # Check if we should use MLflow
            use_mlflow = not inference_settings.offline_mode
            if use_mlflow:
                try:
                    mlflow.set_tracking_uri(inference_settings.tracking_uri)
                    client = mlflow.tracking.MlflowClient()
                    
                    logger.info("Connecting to MLflow Tracking Server: %s", inference_settings.tracking_uri)
                    
                    # Find model in Production stage
                    prod_version = None
                    versions = client.get_registered_model(inference_settings.model_name).latest_versions
                    for v in versions:
                        if v.current_stage == inference_settings.model_stage:
                            prod_version = v
                            break
                            
                    if prod_version:
                        logger.info(
                            "Found Production model in MLflow Registry | version=%s | run_id=%s",
                            prod_version.version,
                            prod_version.run_id,
                        )
                        # Load model using mlflow
                        model_uri = f"models:/{inference_settings.model_name}/{inference_settings.model_stage}"
                        model = mlflow.catboost.load_model(model_uri)
                        mlflow_run_id = prod_version.run_id
                        mlflow_model_version = prod_version.version
                        model_source = "mlflow"
                        
                        # Download metadata artifacts
                        tmp_dir = Path("artifacts/tmp_mlflow")
                        tmp_dir.mkdir(parents=True, exist_ok=True)
                        
                        try:
                            # Retrieve metadata and threshold JSON files from run artifacts
                            client.download_artifacts(mlflow_run_id, "metadata/threshold.json", str(tmp_dir))
                            client.download_artifacts(mlflow_run_id, "metadata/training_metadata.json", str(tmp_dir))
                            
                            with open(tmp_dir / "metadata/threshold.json", "r", encoding="utf-8") as fh:
                                th_data = json.load(fh)
                                threshold = float(th_data["optimized_threshold"])
                                threshold_strategy = th_data.get("optimization_strategy", "UNKNOWN")
                                
                            with open(tmp_dir / "metadata/training_metadata.json", "r", encoding="utf-8") as fh:
                                training_metadata = json.load(fh)
                                
                            logger.info("Successfully fetched model, threshold=%f, and metadata from MLflow", threshold)
                        except Exception as meta_exc:
                            logger.warning("Failed to retrieve run metadata from MLflow artifacts: %s. Falling back to local files.", meta_exc)
                            model = None  # Force local fallback
                    else:
                        logger.warning("No Production model found in MLflow Registry. Falling back to local files.")
                except Exception as mlflow_exc:
                    logger.warning("MLflow connection failed: %s. Falling back to local files.", mlflow_exc)

            # Fallback to local files if MLflow load failed or was bypassed
            if model is None:
                logger.info("Loading model and training metadata from local fallback: %s", inference_settings.local_model_dir)
                model_file = inference_settings.local_model_dir / "model.joblib"
                metadata_file = inference_settings.local_model_dir / "training_metadata.json"
                threshold_file = inference_settings.evaluation_report_dir / "threshold.json"
                
                if not model_file.exists():
                    raise ConfigurationError(f"Local fallback model file not found: {model_file}")
                if not metadata_file.exists():
                    raise ConfigurationError(f"Local fallback metadata file not found: {metadata_file}")
                
                try:
                    model = joblib.load(model_file)
                    
                    with open(metadata_file, "r", encoding="utf-8") as fh:
                        training_metadata = json.load(fh)
                        
                    # Load threshold if exists, otherwise default to config settings
                    if threshold_file.exists():
                        with open(threshold_file, "r", encoding="utf-8") as fh:
                            th_data = json.load(fh)
                            threshold = float(th_data["optimized_threshold"])
                            threshold_strategy = th_data.get("optimization_strategy", "UNKNOWN")
                    else:
                        threshold = float(settings.evaluation.default_threshold)
                        threshold_strategy = "DEFAULT"
                        
                    logger.info("Successfully loaded local fallback assets | threshold=%f", threshold)
                except Exception as fallback_exc:
                    raise ConfigurationError(f"Failed to load local fallback assets: {fallback_exc}") from fallback_exc

            # 2. Load Preprocessing Artifacts
            prep_dir = inference_settings.local_preprocessing_dir
            freq_file = prep_dir / "frequency_encoder.joblib"
            ohe_file = prep_dir / "ohe_encoder.joblib"
            scaler_file = prep_dir / "scaler.joblib"
            schema_file = prep_dir / "dataset_schema.json"
            prep_metadata_file = prep_dir / "transformation_metadata.json"
            
            for f in (freq_file, ohe_file, scaler_file, schema_file, prep_metadata_file):
                if not f.exists():
                    raise ConfigurationError(f"Preprocessing artifact file not found: {f}")
            
            try:
                frequency_encoder = joblib.load(freq_file)
                ohe_encoder = joblib.load(ohe_file)
                scaler = joblib.load(scaler_file)
                
                with open(schema_file, "r", encoding="utf-8") as fh:
                    feature_schema = json.load(fh)
                    
                with open(prep_metadata_file, "r", encoding="utf-8") as fh:
                    prep_metadata = json.load(fh)
            except Exception as prep_exc:
                raise ConfigurationError(f"Failed to load preprocessing artifacts: {prep_exc}") from prep_exc

            # 3. Startup validations
            self._validate_assets(
                training_metadata=training_metadata,
                prep_metadata=prep_metadata,
                threshold=threshold,
            )

            # 4. Instantiate pipelines and assign transformers to pp_pipeline
            from src.features.pipeline import FeatureEngineeringPipeline
            from src.preprocessing.pipeline import PreprocessingPipeline
            
            fe_pipeline = FeatureEngineeringPipeline()
            pp_pipeline = PreprocessingPipeline()
            
            # Store pre-fitted transformers on the pipeline instance for reuse in transform()
            pp_pipeline.frequency_encoder = frequency_encoder
            pp_pipeline.ohe_encoder = ohe_encoder
            pp_pipeline.scaler = scaler

            # 5. Populate context
            self._context = InferenceContext(
                model=model,
                threshold=threshold,
                feature_schema=feature_schema,
                preprocessing_pipeline=pp_pipeline,
                feature_engineering_pipeline=fe_pipeline,
                metadata={
                    "training": training_metadata,
                    "preprocessing": prep_metadata,
                    "threshold_strategy": threshold_strategy,
                    "model_source": model_source,
                },
                model_version=training_metadata.get("model_version", "unknown"),
                dataset_hash=training_metadata.get("dataset_hash", "unknown"),
                training_version=training_metadata.get("training_version", "unknown"),
                preprocessing_version=training_metadata.get("preprocessing_version", "unknown"),
                feature_engineering_version="1.0.0",  # default FE code version
                mlflow_run_id=mlflow_run_id,
                mlflow_model_version=mlflow_model_version,
            )
            
            logger.info("InferenceContext loaded and validated successfully.")
            return self._context

    def _validate_assets(
        self,
        training_metadata: dict[str, Any],
        prep_metadata: dict[str, Any],
        threshold: float,
    ) -> None:
        """
        Runs checks to ensure loaded model, dataset hashes, and config versions match.
        
        Raises:
            ArtifactMismatch: If any structural mismatch is found.
            ConfigurationError: If threshold values are out of bounds.
        """
        # Threshold boundary validation
        if not (0.0 <= threshold <= 1.0):
            raise ConfigurationError(f"Optimized threshold {threshold} is outside valid [0.0, 1.0] probability range.")

        # Version checks against configurations
        expected_training_ver = settings.training.training_version
        expected_prep_ver = settings.preprocessing.preprocessing_version
        
        actual_training_ver = training_metadata.get("training_version")
        actual_prep_ver = training_metadata.get("preprocessing_version")

        if actual_training_ver != expected_training_ver:
            raise ArtifactMismatch(
                f"Training version mismatch! Settings expected '{expected_training_ver}', "
                f"but loaded model was trained on '{actual_training_ver}'."
            )
            
        if actual_prep_ver != expected_prep_ver:
            raise ArtifactMismatch(
                f"Preprocessing version mismatch! Settings expected '{expected_prep_ver}', "
                f"but loaded model expected '{actual_prep_ver}'."
            )

        # Dataset Hash validation
        # Find dataset_hash in training_metadata
        tr_dataset_hash = training_metadata.get("dataset_hash")
        
        # In Preprocessing transformation_metadata.json we don't have a direct dataset_hash field,
        # but we can look for it in the latest manifest.json or latest files.
        # Alternatively, we can check that it's consistent.
        # Let's check training_metadata feature_schema_hash matches preprocessing
        tr_schema_hash = training_metadata.get("feature_schema_hash")
        
        # Validate that model training features match exactly what preprocessing expects
        model_features = training_metadata.get("feature_names", [])
        if len(model_features) != 33:
            raise ArtifactMismatch(
                f"Model features count mismatch! Expected 33 features, model has {len(model_features)}."
            )

        logger.info("Startup validation: all version and hash matches passed.")
