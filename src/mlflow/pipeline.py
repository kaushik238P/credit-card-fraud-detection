"""
High-level orchestration pipeline for tracking experiment runs.

Main Class:
    MLflowPipeline

Connects training and evaluation payloads, infers model signature, uploads artifacts,
registers versioned models, and compiles the final ExperimentResult.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import settings
from src.evaluation.models import EvaluationResult
from src.mlflow.artifacts import ArtifactResolver
from src.mlflow.exceptions import PipelineError
from src.mlflow.models import (
    ArtifactMetadata,
    ExperimentReport,
    ExperimentResult,
    RegisteredModel,
    RunMetadata,
)
from src.mlflow.registry import SchemaRegistry
from src.mlflow.tracker import MLflowTracker
from src.training.models import TrainingResult

logger = logging.getLogger(__name__)


class MLflowPipeline:
    """
    Orchestrates the entire MLflow experiment tracking flow.

    Coordinates SchemaRegistry, ArtifactResolver, and MLflowTracker.
    """

    def __init__(self, cfg: Any = None) -> None:
        # Load configs, allowing settings.evaluation overrides or defaults
        self._cfg = cfg or getattr(settings, "evaluation", None)
        
        # Read parameters from environment or defaults
        self.experiment_name = os.environ.get("FRAUD_MLFLOW_EXPERIMENT_NAME", "FraudDetection")
        self.tracking_uri = os.environ.get("FRAUD_MLFLOW_TRACKING_URI", "http://localhost:5000")
        self.registry_name = os.environ.get("FRAUD_MLFLOW_REGISTRY_NAME", "credit_card_fraud_detection")
        self.target_stage = os.environ.get("FRAUD_MLFLOW_REGISTRY_STAGE", "Development")
        self.author = os.environ.get("FRAUD_MLFLOW_AUTHOR", "kaush")
        self.project_name = os.environ.get("FRAUD_MLFLOW_PROJECT", "FraudDetection")
        self.environment = os.environ.get("FRAUD_MLFLOW_ENVIRONMENT", "development")

        self._tracker = MLflowTracker(tracking_uri=self.tracking_uri)
        self._registry = SchemaRegistry(project_name=self.project_name, author=self.author)

    def track(
        self,
        training_result: TrainingResult,
        evaluation_result: EvaluationResult,
    ) -> ExperimentResult:
        """
        Executes the full tracking pipeline.

        Args:
            training_result: Payload from the Training Layer.
            evaluation_result: Payload from the Evaluation Layer.

        Returns:
            ExperimentResult: Metadata container of the registered run.

        Raises:
            PipelineError: If tracking or registration encounters a fatal error.
        """
        # ── 1. Pipeline Guards ────────────────────────────────────────────
        self._guard(training_result, evaluation_result)

        model_name = training_result.metadata.model_type
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_name = f"{model_name}_{timestamp}"

        # ── 2. Start Experiment Run ───────────────────────────────────────
        run_meta = self._tracker.start_run(
            experiment_name=self.experiment_name,
            run_name=run_name,
        )

        try:
            # ── 3. Map & Log Parameters ───────────────────────────────────
            params = self._registry.map_parameters(training_result, evaluation_result)
            self._tracker.log_params(params)
            logger.info("Parameters logged successfully.")

            # ── 4. Map & Log Metrics ──────────────────────────────────────
            metrics = self._registry.map_metrics(evaluation_result, training_result)
            self._tracker.log_metrics(metrics)
            logger.info("Metrics logged successfully.")

            # ── 5. Map & Log Tags ─────────────────────────────────────────
            tags = self._registry.map_tags(training_result, evaluation_result, self.environment)
            self._tracker.log_tags(tags)
            logger.info("Tags logged successfully.")

            # ── 6. Infer Model Signature & Input Example ──────────────────
            df_sample = self._load_preprocessed_sample(training_result.feature_names)
            
            # Save input example JSON physically in artifact directory
            input_example_path = Path(evaluation_result.artifacts.artifact_dir) / "input_example.json"
            try:
                with open(input_example_path, "w", encoding="utf-8") as fh:
                    json.dump(df_sample.to_dict(orient="records")[0], fh, indent=2)
                logger.info("Input example saved at %s", input_example_path)
            except Exception as exc:
                logger.warning("Failed to save input example JSON file: %s", exc)

            # Build MLflow model signature
            signature = None
            try:
                from mlflow.models.signature import infer_signature
                # Model predictions array representation
                dummy_output = np.array([0.0])
                signature = infer_signature(df_sample, dummy_output)
            except Exception as exc:
                logger.warning("Failed to infer model signature: %s", exc)

            # ── 7. Log Model Binaries ─────────────────────────────────────
            model_uri = self._tracker.log_model(
                estimator=training_result.estimator,
                artifact_path="model",
                signature=signature,
                input_example=df_sample,
            )
            logger.info("Model binaries logged successfully | model_uri=%s", model_uri)

            # ── 8. Log Artifacts (with Fail-Safe Error Handling) ──────────
            resolved_artifacts, artifact_metadata = ArtifactResolver.resolve_artifacts(
                training_result,
                evaluation_result,
            )

            # Append input_example to artifacts list if it exists
            if input_example_path.exists():
                # Compute hash and size
                size_bytes = input_example_path.stat().st_size
                sha256 = self._compute_sha256(input_example_path)
                created_iso = datetime.fromtimestamp(
                    input_example_path.stat().st_mtime, tz=timezone.utc
                ).isoformat()
                
                meta_example = ArtifactMetadata(
                    filename=input_example_path.name,
                    size_bytes=size_bytes,
                    sha256=sha256,
                    created_at=created_iso,
                    mime_type="application/json",
                    upload_status="PENDING",
                )
                resolved_artifacts[str(input_example_path.resolve())] = "predictions"
                artifact_metadata.append(meta_example)

            uploaded_artifacts = []
            for local_path, dest_folder in resolved_artifacts.items():
                meta = next(
                    (m for m in artifact_metadata if m.filename == Path(local_path).name),
                    None,
                )
                try:
                    self._tracker.log_artifacts(local_path, artifact_path=dest_folder)
                    if meta:
                        meta = ArtifactMetadata(
                            filename=meta.filename,
                            size_bytes=meta.size_bytes,
                            sha256=meta.sha256,
                            created_at=meta.created_at,
                            mime_type=meta.mime_type,
                            upload_status="UPLOADED",
                        )
                except Exception as exc:
                    logger.warning("Artifact upload failed for %s: %s. Continuing...", local_path, exc)
                    if meta:
                        meta = ArtifactMetadata(
                            filename=meta.filename,
                            size_bytes=meta.size_bytes,
                            sha256=meta.sha256,
                            created_at=meta.created_at,
                            mime_type=meta.mime_type,
                            upload_status="FAILED",
                        )
                if meta:
                    uploaded_artifacts.append(meta)

            logger.info("Artifacts upload processed | total=%d", len(uploaded_artifacts))

            # ── 9. Register Model in registry ─────────────────────────────
            reg_model = self._tracker.register_model(
                model_uri=model_uri,
                name=self.registry_name,
                stage=self.target_stage,
            )

        except Exception as exc:
            self._tracker.end_run(status="FAILED")
            raise PipelineError(f"Experiment tracking pipeline failed: {exc}") from exc

        # ── 10. End Run and Return ────────────────────────────────────────
        self._tracker.end_run(status="FINISHED")

        # Re-resolve run duration
        end_time_iso = datetime.now(tz=timezone.utc).isoformat()
        start_time_dt = datetime.fromisoformat(run_meta.start_time)
        duration_ms = (datetime.now(tz=timezone.utc) - start_time_dt).total_seconds() * 1000

        run_meta = RunMetadata(
            run_id=run_meta.run_id,
            experiment_id=run_meta.experiment_id,
            status="FINISHED",
            artifact_uri=run_meta.artifact_uri,
            start_time=run_meta.start_time,
            end_time=end_time_iso,
            duration_ms=duration_ms,
        )

        report = ExperimentReport(
            logged_metrics=metrics,
            logged_parameters=params,
            logged_tags=tags,
            uploaded_artifacts=tuple(uploaded_artifacts),
        )

        return ExperimentResult(
            report=report,
            metadata=run_meta,
            registered_model=reg_model,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _guard(self, tr: TrainingResult, er: EvaluationResult) -> None:
        """Enforces input validation constraints, raising PipelineError if invalid."""
        if tr is None:
            raise PipelineError("GUARD: TrainingResult is None.")
        if er is None:
            raise PipelineError("GUARD: EvaluationResult is None.")
        if tr.estimator is None:
            raise PipelineError("GUARD: TrainingResult estimator is empty.")
        if er.report is None:
            raise PipelineError("GUARD: EvaluationResult report is empty.")
        if not tr.metadata.model_version:
            raise PipelineError("GUARD: Training model version runs hash is missing.")

    @staticmethod
    def _load_preprocessed_sample(feature_names: tuple[str, ...]) -> pd.DataFrame:
        """Loads a real preprocessed sample row from disk, or falls back to a zeroed mock DataFrame."""
        try:
            # Check validation or test dataset parquet
            p_val = Path("data/preprocessed/X_validation.parquet")
            p_test = Path("data/preprocessed/X_test.parquet")
            if p_val.exists():
                return pd.read_parquet(p_val).head(1)
            elif p_test.exists():
                return pd.read_parquet(p_test).head(1)
        except Exception as exc:
            logger.warning("Failed to load preprocessed parquet sample from disk: %s", exc)

        # Fallback to a single zero-filled row matching target feature schema columns
        logger.info("Using fallback zero-filled preprocessed schema sample.")
        return pd.DataFrame([{col: 0.0 for col in feature_names}])

    @staticmethod
    def _compute_sha256(path: Path) -> str:
        h = hashlib.sha256()
        try:
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(65536), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return ""
