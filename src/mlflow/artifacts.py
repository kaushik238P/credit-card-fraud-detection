"""
Artifact parser and discovery engine for the MLflow Layer.

Scans training and evaluation results to find physical files on disk,
determines metadata, MIME types, and returns mapped target destinations.
"""

from __future__ import annotations

import hashlib
import logging
import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.evaluation.models import EvaluationResult
from src.mlflow.models import ArtifactMetadata
from src.training.models import TrainingResult

logger = logging.getLogger(__name__)


class ArtifactResolver:
    """
    Registry-driven utility to discover physical artifacts, calculate SHA-256
    signatures, and map them to their MLflow destinations.
    """

    @staticmethod
    def resolve_artifacts(
        tr: TrainingResult,
        er: EvaluationResult,
    ) -> tuple[dict[str, str], list[ArtifactMetadata]]:
        """
        Parses all physical artifacts from training and evaluation payloads.

        Args:
            tr: upstream TrainingResult instance.
            er: upstream EvaluationResult instance.

        Returns:
            tuple[dict[str, str], list[ArtifactMetadata]]:
                - Dictionary mapping local absolute paths to MLflow destination folder structures.
                - List of populated ArtifactMetadata records.
        """
        raw_mappings: list[tuple[str, str]] = []

        # ── 1. Training Layer Serialised Artifacts ────────────────────────
        serialized = tr.serialized
        if serialized.model_path:
            raw_mappings.append((serialized.model_path, "model"))
        if serialized.report_path:
            raw_mappings.append((serialized.report_path, "metadata"))
        if serialized.metadata_path:
            raw_mappings.append((serialized.metadata_path, "metadata"))
        if serialized.feature_names_path:
            raw_mappings.append((serialized.feature_names_path, "features"))

        # ── 2. Evaluation Layer Serialised Artifacts ──────────────────────
        arts = er.artifacts
        if arts.report_json_path:
            raw_mappings.append((arts.report_json_path, "reports"))
        if arts.report_md_path:
            raw_mappings.append((arts.report_md_path, "reports"))
        if arts.metrics_json_path:
            raw_mappings.append((arts.metrics_json_path, "metrics"))
        if arts.business_metrics_json_path:
            raw_mappings.append((arts.business_metrics_json_path, "metrics"))
        if arts.threshold_json_path:
            raw_mappings.append((arts.threshold_json_path, "metadata"))
        if arts.feature_importance_csv_path:
            raw_mappings.append((arts.feature_importance_csv_path, "features"))
        if arts.feature_importance_json_path:
            raw_mappings.append((arts.feature_importance_json_path, "features"))
        if arts.confusion_matrix_csv_path:
            raw_mappings.append((arts.confusion_matrix_csv_path, "plots"))
        if arts.prediction_distribution_json_path:
            raw_mappings.append((arts.prediction_distribution_json_path, "predictions"))
        if arts.prediction_summary_json_path:
            raw_mappings.append((arts.prediction_summary_json_path, "predictions"))
        if arts.manifest_path:
            raw_mappings.append((arts.manifest_path, ""))

        # ── 3. Diagnostic Plots ───────────────────────────────────────────
        for plot_name, plot_path in arts.plot_paths.items():
            if plot_path:
                raw_mappings.append((plot_path, "plots"))

        # Process and validate discovered files
        path_destination_map: dict[str, str] = {}
        metadata_list: list[ArtifactMetadata] = []

        for local_path_str, dest_subfolder in raw_mappings:
            local_path = Path(local_path_str)
            if not local_path.exists() or not local_path.is_file():
                logger.warning("Discovered artifact path does not exist or is not a file: %s", local_path_str)
                continue

            # Calculate metadata details
            size_bytes = local_path.stat().st_size
            created_iso = datetime.fromtimestamp(
                local_path.stat().st_mtime, tz=timezone.utc
            ).isoformat()
            
            # Compute MIME type
            mime, _ = mimetypes.guess_type(local_path.name)
            if not mime:
                # Custom overrides for common extensions
                ext = local_path.suffix.lower()
                if ext == ".joblib":
                    mime = "application/octet-stream"
                elif ext == ".json":
                    mime = "application/json"
                elif ext == ".csv":
                    mime = "text/csv"
                elif ext == ".md":
                    mime = "text/markdown"
                elif ext == ".png":
                    mime = "image/png"
                else:
                    mime = "application/octet-stream"

            # Compute SHA-256 checksum
            sha256 = ArtifactResolver._compute_sha256(local_path)

            meta = ArtifactMetadata(
                filename=local_path.name,
                size_bytes=size_bytes,
                sha256=sha256,
                created_at=created_iso,
                mime_type=mime,
                upload_status="PENDING",
            )
            
            # Map absolute path to target subfolder structure in MLflow run
            path_destination_map[str(local_path.resolve())] = dest_subfolder
            metadata_list.append(meta)

        logger.info("ArtifactResolver resolved %d physical artifact files.", len(path_destination_map))
        return path_destination_map, metadata_list

    @staticmethod
    def _compute_sha256(path: Path) -> str:
        h = hashlib.sha256()
        try:
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(65536), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception as exc:
            logger.warning("Failed to calculate SHA-256 for file %s: %s", path, exc)
            return ""
