"""
Thread-safe wrapper for direct interaction with the MLflow Tracking client.

All direct mlflow imports and calls are encapsulated within this module.
Other modules within this layer must interact with MLflow only through MLflowTracker.
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any

import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.tracking import MlflowClient

from pathlib import Path

from src.mlflow.exceptions import RegistrationError, TrackingError
from src.mlflow.models import RegisteredModel, RunMetadata

logger = logging.getLogger(__name__)


class MLflowTracker:
    """
    Thread-safe, direct wrapper around the raw mlflow package.

    Ensures synchronization of logging calls and isolates mlflow client APIs.
    """

    def __init__(self, tracking_uri: str | None = None) -> None:
        self._lock = threading.Lock()
        self._client: MlflowClient | None = None
        self._active_run: Any = None
        
        # Configure tracking backend connection
        uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
        if uri:
            mlflow.set_tracking_uri(uri)
            logger.info("MLflow tracking URI configured: %s", uri)
        
        self._client = MlflowClient()

    def start_run(
        self,
        experiment_name: str,
        run_name: str,
        nested: bool = False,
        parent_run_id: str | None = None,
    ) -> RunMetadata:
        """
        Starts a new active experiment run.

        Args:
            experiment_name: Name of target MLflow experiment.
            run_name: Custom name prefix for the current run instance.
            nested: Whether to start this as a child run inside an active parent run.
            parent_run_id: Optional parent run identifier to associate nested runs.

        Returns:
            RunMetadata: Structured configuration details of the started run.
        """
        with self._lock:
            try:
                # Resolve or create target experiment
                exp = mlflow.get_experiment_by_name(experiment_name)
                if exp is None:
                    exp_id = mlflow.create_experiment(experiment_name)
                    logger.info("Created new MLflow experiment '%s' (ID: %s)", experiment_name, exp_id)
                else:
                    exp_id = exp.experiment_id

                mlflow.set_experiment(experiment_name)

                # Set parent run ID if nested or parent_run_id provided
                tags = {}
                if parent_run_id:
                    tags["mlflow.parentRunId"] = parent_run_id

                # Trigger raw MLflow run
                run = mlflow.start_run(
                    run_name=run_name,
                    experiment_id=exp_id,
                    nested=nested,
                    tags=tags,
                )
                self._active_run = run
                
                start_iso = datetime.fromtimestamp(
                    run.info.start_time / 1000.0, tz=timezone.utc
                ).isoformat()

                logger.info(
                    "MLflow run started | run_id=%s | experiment=%s | run_name=%s",
                    run.info.run_id,
                    experiment_name,
                    run_name,
                )

                return RunMetadata(
                    run_id=run.info.run_id,
                    experiment_id=run.info.experiment_id,
                    status=run.info.status,
                    artifact_uri=run.info.artifact_uri,
                    start_time=start_iso,
                    end_time="",
                    duration_ms=0.0,
                )
            except Exception as exc:
                raise TrackingError(
                    f"Failed to start MLflow run: {exc}",
                    context={"experiment_name": experiment_name, "run_name": run_name},
                ) from exc

    def log_params(self, params: dict[str, Any]) -> None:
        """Logs a dictionary of parameters as key-value pairs."""
        with self._lock:
            if not self._active_run:
                raise TrackingError("Cannot log parameters: no active run.")
            try:
                # Stringify complex values to remain compatible
                cleaned_params = {}
                for k, v in params.items():
                    if isinstance(v, (dict, list, tuple)):
                        cleaned_params[k] = str(v)
                    else:
                        cleaned_params[k] = v

                mlflow.log_params(cleaned_params)
                logger.debug("Logged %d parameters to MLflow.", len(cleaned_params))
            except Exception as exc:
                raise TrackingError(f"Failed to log parameters: {exc}") from exc

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Logs a dictionary of metrics."""
        with self._lock:
            if not self._active_run:
                raise TrackingError("Cannot log metrics: no active run.")
            try:
                # Clean metric values, removing non-numeric dictionary components
                cleaned_metrics = {}
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        cleaned_metrics[k] = float(v)
                    elif isinstance(v, dict):
                        # Flatten or skip nested metrics
                        continue
                    else:
                        try:
                            cleaned_metrics[k] = float(v)
                        except (ValueError, TypeError):
                            continue

                mlflow.log_metrics(cleaned_metrics)
                logger.debug("Logged %d metrics to MLflow.", len(cleaned_metrics))
            except Exception as exc:
                raise TrackingError(f"Failed to log metrics: {exc}") from exc

    def log_tags(self, tags: dict[str, str]) -> None:
        """Logs a dictionary of tags."""
        with self._lock:
            if not self._active_run:
                raise TrackingError("Cannot log tags: no active run.")
            try:
                mlflow.set_tags(tags)
                logger.debug("Logged %d tags to MLflow.", len(tags))
            except Exception as exc:
                raise TrackingError(f"Failed to log tags: {exc}") from exc

    def log_artifacts(self, local_dir_or_file: str, artifact_path: str | None = None) -> None:
        """Uploads files or directories to the MLflow artifact store."""
        with self._lock:
            if not self._active_run:
                raise TrackingError("Cannot log artifacts: no active run.")
            try:
                p = Path(local_dir_or_file)
                if p.is_dir():
                    mlflow.log_artifacts(str(p.resolve()), artifact_path=artifact_path)
                else:
                    mlflow.log_artifact(str(p.resolve()), artifact_path=artifact_path)
                logger.debug("Logged artifact(s) from %s to MLflow.", local_dir_or_file)
            except Exception as exc:
                raise TrackingError(f"Failed to log artifacts from {local_dir_or_file}: {exc}") from exc

    def log_model(
        self,
        estimator: Any,
        artifact_path: str,
        signature: ModelSignature | None = None,
        input_example: Any | None = None,
    ) -> str:
        """
        Logs a trained estimator using the appropriate MLflow flavor framework model logger.

        Args:
            estimator: Trained model instance.
            artifact_path: Log destination sub-path.
            signature: Optional inferred schema signature description of model inputs/outputs.
            input_example: Preprocessed single sample data matrix.

        Returns:
            str: Model URI path descriptor.
        """
        with self._lock:
            if not self._active_run:
                raise TrackingError("Cannot log model: no active run.")
            try:
                name_lower = str(type(estimator).__name__).lower()

                # Centralized dispatch registry: maps class-name fragment to
                # (mlflow_module_dotpath, model_kwarg_name).  Add future model
                # flavours here without touching any other branch.
                _FLAVOR_REGISTRY: list[tuple[str, str, str]] = [
                    ("xgb",               "mlflow.xgboost",  "xgb_model"),
                    ("lgb",               "mlflow.lightgbm", "lgb_model"),
                    ("catboost",          "mlflow.catboost", "cb_model"),
                    ("logisticregression","mlflow.sklearn",  "sk_model"),
                    ("randomforest",      "mlflow.sklearn",  "sk_model"),
                    ("gradientboosting",  "mlflow.sklearn",  "sk_model"),
                    ("decisiontree",      "mlflow.sklearn",  "sk_model"),
                    ("svc",               "mlflow.sklearn",  "sk_model"),
                ]

                logged = False
                for fragment, module_path, kwarg_name in _FLAVOR_REGISTRY:
                    if fragment in name_lower:
                        try:
                            import importlib
                            flavor_module = importlib.import_module(module_path)
                            flavor_module.log_model(
                                **{
                                    kwarg_name: estimator,
                                    "artifact_path": artifact_path,
                                    "signature": signature,
                                    "input_example": input_example,
                                }
                            )
                            logged = True
                            break
                        except (ImportError, AttributeError) as flavor_err:
                            logger.warning(
                                "Flavor module %s unavailable (%s); falling back to sklearn.",
                                module_path,
                                flavor_err,
                            )

                if not logged:
                    # Generic fallback: sklearn for anything with predict(), else pyfunc
                    if hasattr(estimator, "predict"):
                        import mlflow.sklearn
                        mlflow.sklearn.log_model(
                            sk_model=estimator,
                            artifact_path=artifact_path,
                            signature=signature,
                            input_example=input_example,
                        )
                    else:
                        import mlflow.pyfunc
                        mlflow.pyfunc.log_model(
                            artifact_path=artifact_path,
                            python_model=estimator,
                            signature=signature,
                            input_example=input_example,
                        )

                model_uri = f"runs:/{self._active_run.info.run_id}/{artifact_path}"
                logger.info("Logged model to MLflow | URI=%s", model_uri)
                return model_uri
            except Exception as exc:
                raise TrackingError(f"Failed to log model to MLflow: {exc}") from exc

    def register_model(
        self,
        model_uri: str,
        name: str,
        stage: str = "Development",
    ) -> RegisteredModel:
        """
        Registers a logged model to the Model Registry database.

        Args:
            model_uri: Run-based source URI path.
            name: Registry model identifier.
            stage: Initial promotion stage target.

        Returns:
            RegisteredModel: Metadata descriptor for the versioned registry record.
        """
        with self._lock:
            try:
                # Register model version record
                model_details = mlflow.register_model(
                    model_uri=model_uri,
                    name=name,
                )
                
                # Promote registered model version stage if requested
                if stage and stage != "Development":
                    client = self._client or MlflowClient()
                    client.transition_model_version_stage(
                        name=name,
                        version=str(model_details.version),
                        stage=stage,
                        archive_existing_versions=(stage == "Production"),
                    )
                    stage_name = stage
                else:
                    stage_name = "Development"

                logger.info(
                    "Model registered successfully | name=%s | version=%s | stage=%s",
                    name,
                    model_details.version,
                    stage_name,
                )

                return RegisteredModel(
                    name=name,
                    version=str(model_details.version),
                    stage=stage_name,
                    source_run_id=model_details.run_id,
                    model_uri=model_uri,
                )
            except Exception as exc:
                raise RegistrationError(
                    f"Failed to register model name='{name}' with URI='{model_uri}': {exc}",
                    context={"model_name": name, "model_uri": model_uri, "target_stage": stage},
                ) from exc

    def end_run(self, status: str = "FINISHED") -> None:
        """Ends the active experiment run, calculating durations."""
        with self._lock:
            if not self._active_run:
                return
            try:
                mlflow.end_run(status=status)
                logger.info(
                    "MLflow run complete | run_id=%s | status=%s",
                    self._active_run.info.run_id,
                    status,
                )
                self._active_run = None
            except Exception as exc:
                raise TrackingError(f"Failed to end active MLflow run: {exc}") from exc
