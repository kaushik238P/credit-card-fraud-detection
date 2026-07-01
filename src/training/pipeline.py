"""
TrainingPipeline — orchestrator for the Training Layer.

Public API:
    pipeline = TrainingPipeline()
    result = pipeline.train(preprocessed_dataset)

Internal ArtifactManager handles all serialization.
Pipeline only orchestrates steps; it never writes files directly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from config.settings import settings
from src.preprocessing.models import PreprocessedDataset
from src.training.exceptions import (
    ModelCreationError,
    PipelineError,
    SerializationError,
    TrainingFailureError,
)
from src.training.factory import ModelFactory
from src.training.models import (
    ModelMetadata,
    ModelType,
    SerializedModel,
    TrainingReport,
    TrainingResult,
)
from src.training.registry import MODEL_REGISTRY, get_entry
from src.training.trainer import Trainer

logger = logging.getLogger(__name__)

_IMBALANCE_CLASS_WEIGHT = "CLASS_WEIGHT"
_IMBALANCE_SCALE_POS = "SCALE_POS_WEIGHT"
_IMBALANCE_SAMPLE_WEIGHT = "SAMPLE_WEIGHT"
_IMBALANCE_NONE = "NONE"


# ===========================================================================
# Internal ArtifactManager
# ===========================================================================


class _ArtifactManager:
    """
    Handles all Training Layer artifact I/O.

    Creates versioned run directories under artifact_dir:
        artifacts/models/
            YYYYMMDD_HHMMSS/
                model.joblib
                training_metadata.json
                training_report.json
                feature_names.json
                manifest.json
            latest/          ← shutil.copytree copy of newest run

    All methods raise SerializationError on failure (caller catches).
    """

    def __init__(self, artifact_dir: str, run_id: str) -> None:
        self._run_id = run_id
        self._run_dir = Path(artifact_dir) / run_id
        self._latest_dir = Path(artifact_dir) / "latest"
        self._run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "ArtifactManager initialised | run_dir=%s", self._run_dir
        )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def save_model(self, estimator: Any) -> str:
        """Serialises the fitted estimator to model.joblib."""
        path = self._run_dir / "model.joblib"
        try:
            joblib.dump(estimator, path)
            size_kb = path.stat().st_size // 1024
            logger.info(
                "Artifact saved | name=model.joblib | path=%s | size=%dKB",
                path,
                size_kb,
            )
            return str(path.resolve())
        except Exception as exc:
            raise SerializationError(
                artifact_name="model.joblib",
                artifact_path=str(path),
                detail="joblib.dump failed.",
                original_exc=exc,
            ) from exc

    # ------------------------------------------------------------------
    # JSON helpers
    # ------------------------------------------------------------------

    def save_json(self, data: dict, filename: str) -> str:
        """Writes a JSON-serialisable dict with an explicit filename."""
        path = self._run_dir / filename
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, default=str)
            logger.info(
                "Artifact saved | name=%s | path=%s", filename, path
            )
            return str(path.resolve())
        except Exception as exc:
            raise SerializationError(
                artifact_name=filename,
                artifact_path=str(path),
                detail="JSON write failed.",
                original_exc=exc,
            ) from exc

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def save_manifest(
        self,
        artifact_paths: dict[str, str],
        metadata: ModelMetadata,
        preprocessing_report_hash: str | None,
    ) -> str:
        """Saves manifest.json with SHA-256 hashes of all artifacts."""
        entries: list[dict] = []
        for name, path_str in artifact_paths.items():
            p = Path(path_str)
            if p.exists():
                sha256 = self._sha256(p)
                size_bytes = p.stat().st_size
                mtime = datetime.fromtimestamp(
                    p.stat().st_mtime, tz=timezone.utc
                ).isoformat()
            else:
                sha256, size_bytes, mtime = "", 0, ""
            entries.append({
                "name": name,
                "filename": p.name,
                "sha256": sha256,
                "size_bytes": size_bytes,
                "created_at": mtime,
            })

        manifest = {
            "model_version": metadata.model_version,
            "model_type": metadata.model_type.value,
            "training_version": metadata.training_version,
            "preprocessing_version": metadata.preprocessing_version,
            "preprocessing_artifact_version": metadata.preprocessing_artifact_version,
            "dataset_hash": metadata.dataset_hash,
            "feature_schema_hash": metadata.feature_schema_hash,
            "preprocessing_report_hash": preprocessing_report_hash,
            "artifacts": entries,
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        return self.save_json(manifest, "manifest.json")

    # ------------------------------------------------------------------
    # Latest sync
    # ------------------------------------------------------------------

    def sync_latest(self) -> str:
        """Copies run directory to latest/ using shutil (Windows/Docker safe)."""
        try:
            if self._latest_dir.exists():
                shutil.rmtree(self._latest_dir)
            shutil.copytree(self._run_dir, self._latest_dir)
            logger.info(
                "Latest artifacts synced | latest=%s", self._latest_dir
            )
            return str(self._latest_dir.resolve())
        except Exception as exc:
            raise SerializationError(
                artifact_name="latest",
                artifact_path=str(self._latest_dir),
                detail="Failed to sync latest artifact directory.",
                original_exc=exc,
            ) from exc

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def run_dir(self) -> str:
        return str(self._run_dir.resolve())

    @property
    def latest_dir(self) -> str:
        return str(self._latest_dir.resolve())

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()


# ===========================================================================
# TrainingPipeline
# ===========================================================================


class TrainingPipeline:
    """
    Orchestrates the full Training Layer.

    Stateless across calls — each call to train() is independent.
    Configuration read once at construction from config.settings.training.

    Example:
        >>> pipeline = TrainingPipeline()
        >>> result = pipeline.train(preprocessed_dataset)
        >>> print(result.report)
    """

    def __init__(self, cfg=None) -> None:
        self._cfg = cfg or settings.training

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, preprocessed_dataset: PreprocessedDataset) -> TrainingResult:
        """
        Runs the full Training pipeline.

        Args:
            preprocessed_dataset: Output of the Preprocessing Layer.

        Returns:
            TrainingResult: Fitted estimator, cached validation predictions,
                metadata, report, and serialized artifact paths.

        Raises:
            PipelineError: If inputs are invalid or registry is empty.
            ModelCreationError: If factory fails to build the estimator.
            TrainingFailureError: If estimator.fit() raises.
        """
        _global_start = time.perf_counter()
        run_id = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")

        # ── 1. Guard ──────────────────────────────────────────────────────
        self._guard(preprocessed_dataset)

        X_train = preprocessed_dataset.X_train
        y_train = preprocessed_dataset.y_train
        X_val = preprocessed_dataset.X_validation
        y_val = preprocessed_dataset.y_validation
        feature_names = preprocessed_dataset.feature_names
        target_column = preprocessed_dataset.target_column
        pp_report = preprocessed_dataset.report

        dataset_hash = pp_report.dataset_hash
        preprocessing_version = getattr(
            pp_report.model_metadata if hasattr(pp_report, "model_metadata")
            else pp_report, "preprocessing_version", "unknown"
        ) if False else settings.preprocessing.preprocessing_version
        preprocessing_artifact_version = getattr(
            pp_report, "report_id", run_id
        )

        logger.info(
            "Training pipeline started | run_id=%s | model=%s | rows=%d | "
            "features=%d | dataset_hash=%s | version=%s",
            run_id,
            self._cfg.active_model,
            len(X_train),
            len(feature_names),
            dataset_hash or "unknown",
            self._cfg.training_version,
        )

        warnings_log: list[str] = []
        timeline: list[dict] = []
        artifact_paths: dict[str, str] = {}

        # ── 2a. Validate feature matrices ─────────────────────────────────
        # Ensure that every feature is numeric. If non-numeric columns exist,
        # raise PipelineError (fail fast).
        self._validate_all_numeric(X_train, X_val, model_type=self._cfg.active_model)

        # ── 2. Resolve registry entry ─────────────────────────────────────
        model_type = ModelType.from_string(self._cfg.active_model)
        registry_entry = get_entry(model_type)
        if not registry_entry.enabled:
            raise PipelineError(
                stage="REGISTRY",
                detail=f"Model '{model_type.value}' is disabled in MODEL_REGISTRY.",
            )

        # ── 3. Compute imbalance ──────────────────────────────────────────
        imbalance_ratio, class_weight_map, scale_pos_weight, sample_weight = (
            self._compute_imbalance(y_train)
        )

        # ── 4. Create model ───────────────────────────────────────────────
        step_start = time.perf_counter()
        step_start_iso = datetime.now(tz=timezone.utc).isoformat()

        estimator = ModelFactory.create(
            model_type=model_type,
            cfg=self._cfg,
            class_weight_map=class_weight_map,
            scale_pos_weight=scale_pos_weight,
        )

        step_ms = (time.perf_counter() - step_start) * 1_000
        timeline.append({
            "step": "MODEL_CREATION",
            "start_time": step_start_iso,
            "end_time": datetime.now(tz=timezone.utc).isoformat(),
            "duration_ms": round(step_ms, 2),
        })
        logger.info(
            "Model created | type=%s | strategy=%s | ratio=%.2f | duration=%.1fms",
            model_type.value,
            self._cfg.imbalance_strategy,
            imbalance_ratio,
            step_ms,
        )

        # ── 5. Train ──────────────────────────────────────────────────────
        step_start = time.perf_counter()
        step_start_iso = datetime.now(tz=timezone.utc).isoformat()

        estimator, fit_duration_ms = Trainer.train(
            estimator=estimator,
            X_train=X_train,
            y_train=y_train,
            model_type=model_type,
            sample_weight=sample_weight,
        )

        step_ms = (time.perf_counter() - step_start) * 1_000
        timeline.append({
            "step": "TRAINING",
            "start_time": step_start_iso,
            "end_time": datetime.now(tz=timezone.utc).isoformat(),
            "duration_ms": round(step_ms, 2),
        })

        # ── 6. Validation predictions (cached for Evaluation Layer) ───────
        val_preds = Trainer.predict(estimator, X_val, model_type)
        val_proba = Trainer.predict_proba(estimator, X_val, model_type)

        # ── 7. Build feature schema hash ──────────────────────────────────
        feature_schema_hash = hashlib.sha256(
            json.dumps(sorted(feature_names)).encode("utf-8")
        ).hexdigest()

        # ── 8. Build hyperparameter snapshot ─────────────────────────────
        hyperparameters = self._extract_hyperparameters(estimator, model_type)

        # ── 9. Build ModelMetadata ────────────────────────────────────────
        imbalance_value = (
            scale_pos_weight if scale_pos_weight is not None
            else class_weight_map if class_weight_map is not None
            else "N/A"
        )
        metadata = ModelMetadata(
            model_type=model_type,
            model_name=registry_entry.model_name,
            model_version=run_id,
            training_version=self._cfg.training_version,
            hyperparameters=hyperparameters,
            random_seed=self._cfg.random_seed,
            imbalance_strategy=self._cfg.imbalance_strategy,
            imbalance_ratio=imbalance_ratio,
            imbalance_value_applied=imbalance_value,
            feature_count=len(feature_names),
            feature_names=tuple(feature_names),
            feature_schema_hash=feature_schema_hash,
            dataset_hash=dataset_hash,
            preprocessing_version=preprocessing_version,
            preprocessing_artifact_version=preprocessing_artifact_version,
        )

        # ── 10. Serialize ─────────────────────────────────────────────────
        step_start = time.perf_counter()
        step_start_iso = datetime.now(tz=timezone.utc).isoformat()

        artifact_manager = _ArtifactManager(
            artifact_dir=self._cfg.artifact_dir,
            run_id=run_id,
        )

        serialized_model = self._serialize(
            artifact_manager=artifact_manager,
            estimator=estimator,
            metadata=metadata,
            feature_names=list(feature_names),
            artifact_paths=artifact_paths,
            warnings_log=warnings_log,
            pp_report_hash=dataset_hash,
        )

        step_ms = (time.perf_counter() - step_start) * 1_000
        timeline.append({
            "step": "SERIALIZATION",
            "start_time": step_start_iso,
            "end_time": datetime.now(tz=timezone.utc).isoformat(),
            "duration_ms": round(step_ms, 2),
        })

        # ── 11. Manifest ──────────────────────────────────────────────────
        step_start = time.perf_counter()
        step_start_iso = datetime.now(tz=timezone.utc).isoformat()

        try:
            manifest_path = artifact_manager.save_manifest(
                artifact_paths=artifact_paths,
                metadata=metadata,
                preprocessing_report_hash=dataset_hash,
            )
            artifact_paths["manifest"] = manifest_path
        except SerializationError as exc:
            warnings_log.append(f"Manifest save failed: {exc.message}")
            logger.warning("Manifest save failed: %s", exc.message)
            manifest_path = ""

        step_ms = (time.perf_counter() - step_start) * 1_000
        timeline.append({
            "step": "MANIFEST",
            "start_time": step_start_iso,
            "end_time": datetime.now(tz=timezone.utc).isoformat(),
            "duration_ms": round(step_ms, 2),
        })

        # ── 12. Sync latest/ ──────────────────────────────────────────────
        step_start = time.perf_counter()
        step_start_iso = datetime.now(tz=timezone.utc).isoformat()

        try:
            latest_dir = artifact_manager.sync_latest()
        except SerializationError as exc:
            warnings_log.append(f"Latest sync failed: {exc.message}")
            logger.warning("Latest sync failed: %s", exc.message)
            latest_dir = ""

        step_ms = (time.perf_counter() - step_start) * 1_000
        timeline.append({
            "step": "LATEST_SYNC",
            "start_time": step_start_iso,
            "end_time": datetime.now(tz=timezone.utc).isoformat(),
            "duration_ms": round(step_ms, 2),
        })

        # ── 13. Build report ──────────────────────────────────────────────
        fraud_count = int(y_train.sum())
        duration_ms = (time.perf_counter() - _global_start) * 1_000

        report = TrainingReport(
            report_id=str(uuid.uuid4()),
            model_metadata=metadata,
            training_rows=len(X_train),
            training_fraud_count=fraud_count,
            training_fraud_percentage=round(
                fraud_count / len(y_train) * 100, 6
            ) if len(y_train) > 0 else 0.0,
            imbalance_ratio=imbalance_ratio,
            feature_count=len(feature_names),
            hyperparameters=hyperparameters,
            artifact_paths=artifact_paths,
            training_timeline=tuple(timeline),
            warnings=tuple(warnings_log),
            generated_at=datetime.now(tz=timezone.utc).isoformat(),
            duration_ms=round(duration_ms, 2),
        )

        # Save training_report.json now that report is fully constructed
        self._save_report(artifact_manager, report, artifact_paths, warnings_log)

        serialized = SerializedModel(
            model_path=artifact_paths.get("model", ""),
            metadata_path=artifact_paths.get("training_metadata", ""),
            report_path=artifact_paths.get("training_report", ""),
            feature_names_path=artifact_paths.get("feature_names", ""),
            manifest_path=artifact_paths.get("manifest", ""),
            artifact_dir=artifact_manager.run_dir,
            latest_dir=latest_dir or artifact_manager.latest_dir,
        )


        logger.info(
            "Training pipeline complete | model=%s | version=%s | "
            "rows=%d | features=%d | artifacts=%d | duration=%.1fms | warnings=%d",
            model_type.value,
            run_id,
            len(X_train),
            len(feature_names),
            len(artifact_paths),
            duration_ms,
            len(warnings_log),
        )

        return TrainingResult(
            estimator=estimator,
            validation_predictions=val_preds,
            validation_probabilities=val_proba,
            feature_names=tuple(feature_names),
            target_column=target_column,
            metadata=metadata,
            report=report,
            serialized=serialized,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _guard(self, preprocessed_dataset: PreprocessedDataset) -> None:
        """Guards against invalid pipeline inputs."""
        if preprocessed_dataset is None:
            raise PipelineError(stage="GUARD", detail="PreprocessedDataset is None.")
        if not MODEL_REGISTRY:
            raise PipelineError(stage="GUARD", detail="MODEL_REGISTRY is empty.")
        if not hasattr(preprocessed_dataset, "X_train"):
            raise PipelineError(
                stage="GUARD", detail="PreprocessedDataset missing X_train."
            )
        if len(preprocessed_dataset.X_train) == 0:
            raise PipelineError(stage="GUARD", detail="X_train is empty.")
        if not preprocessed_dataset.feature_names:
            raise PipelineError(stage="GUARD", detail="feature_names is empty.")
        if not self._cfg.active_model:
            raise PipelineError(
                stage="GUARD", detail="active_model is not configured."
            )

    def _compute_imbalance(
        self,
        y_train: pd.Series,
    ) -> tuple[float, dict | None, float | None, np.ndarray | None]:
        """
        Computes imbalance ratio and resolves imbalance params for the factory.

        Returns:
            tuple: (ratio, class_weight_map, scale_pos_weight, sample_weight)
        """
        total = len(y_train)
        fraud = int(y_train.sum())
        legit = total - fraud
        ratio = float(legit / fraud) if fraud > 0 else 1.0

        strategy = self._cfg.imbalance_strategy.upper()

        class_weight_map: dict | None = None
        scale_pos_weight: float | None = None
        sample_weight: np.ndarray | None = None

        if strategy == _IMBALANCE_CLASS_WEIGHT:
            class_weight_map = {0: 1.0, 1: ratio}
        elif strategy == _IMBALANCE_SCALE_POS:
            scale_pos_weight = ratio
        elif strategy == _IMBALANCE_SAMPLE_WEIGHT:
            sample_weight = np.where(
                y_train.values == 1, ratio, 1.0
            ).astype(np.float32)
        # NONE: all remain None

        logger.info(
            "Imbalance | strategy=%s | ratio=%.2f | fraud=%d | legit=%d",
            strategy,
            ratio,
            fraud,
            legit,
        )
        return ratio, class_weight_map, scale_pos_weight, sample_weight

    def _serialize(
        self,
        artifact_manager: _ArtifactManager,
        estimator: Any,
        metadata: ModelMetadata,
        feature_names: list[str],
        artifact_paths: dict[str, str],
        warnings_log: list[str],
        pp_report_hash: str | None,
    ) -> SerializedModel:
        """Serializes all training artifacts, catching SerializationError per artifact."""
        # Model
        try:
            path = artifact_manager.save_model(estimator)
            artifact_paths["model"] = path
        except SerializationError as exc:
            warnings_log.append(f"model.joblib save failed: {exc.message}")
            logger.warning("model.joblib save failed: %s", exc.message)

        # Metadata JSON
        try:
            path = artifact_manager.save_json(
                metadata.to_dict(), "training_metadata.json"
            )
            artifact_paths["training_metadata"] = path
        except SerializationError as exc:
            warnings_log.append(f"training_metadata.json save failed: {exc.message}")
            logger.warning("training_metadata.json save failed: %s", exc.message)

        # Feature names JSON
        try:
            path = artifact_manager.save_json(
                {"features": feature_names, "count": len(feature_names)},
                "feature_names.json",
            )
            artifact_paths["feature_names"] = path
        except SerializationError as exc:
            warnings_log.append(f"feature_names.json save failed: {exc.message}")
            logger.warning("feature_names.json save failed: %s", exc.message)

        # Return placeholder — report path filled after TrainingReport is built
        return SerializedModel(
            model_path=artifact_paths.get("model", ""),
            metadata_path=artifact_paths.get("training_metadata", ""),
            report_path="",  # written below in pipeline after report is built
            feature_names_path=artifact_paths.get("feature_names", ""),
            manifest_path="",
            artifact_dir=artifact_manager.run_dir,
            latest_dir=artifact_manager.latest_dir,
        )

    def _save_report(
        self,
        artifact_manager: _ArtifactManager,
        report: TrainingReport,
        artifact_paths: dict[str, str],
        warnings_log: list[str],
    ) -> str:
        """Saves training_report.json after the report is fully built."""
        try:
            path = artifact_manager.save_json(
                report.to_dict(), "training_report.json"
            )
            artifact_paths["training_report"] = path
            return path
        except SerializationError as exc:
            warnings_log.append(f"training_report.json save failed: {exc.message}")
            logger.warning("training_report.json save failed: %s", exc.message)
            return ""

    @staticmethod
    def _extract_hyperparameters(estimator: Any, model_type: ModelType) -> dict:
        """
        Extracts serialisable hyperparameters from a fitted estimator.
        Falls back to an empty dict if get_params() is unavailable.
        """
        try:
            params = estimator.get_params()
            # Convert non-serialisable values to strings
            return {
                k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                for k, v in params.items()
            }
        except Exception:
            return {"model_type": model_type.value}

    @staticmethod
    def _validate_all_numeric(
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        model_type: str,
    ) -> None:
        """
        Validates that all columns in X_train and X_val are numeric.

        If non-numeric columns are detected, raises PipelineError (fail fast).
        """
        non_numeric_train = [
            col for col in X_train.columns
            if not pd.api.types.is_numeric_dtype(X_train[col])
        ]
        non_numeric_val = [
            col for col in X_val.columns
            if not pd.api.types.is_numeric_dtype(X_val[col])
        ]

        non_numeric = list(sorted(set(non_numeric_train + non_numeric_val)))
        if non_numeric:
            raise PipelineError(
                stage="SANITY_CHECK",
                detail=(
                    f"Non-numeric columns detected in feature matrices before training: {non_numeric}. "
                    f"The Preprocessing Layer must encode all categorical features."
                ),
                context={
                    "model_type": model_type,
                    "non_numeric_columns": non_numeric,
                },
            )
