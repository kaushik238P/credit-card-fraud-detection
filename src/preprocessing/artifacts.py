"""
Artifact management for the Preprocessing Layer.

Handles saving/loading fitted transformers (joblib), metadata (JSON),
and processed dataset folds (Parquet). Implements versioned artifact
directories with a `latest/` copy pointing to the newest run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from src.preprocessing.exceptions import ArtifactError
from src.preprocessing.transformers import BaseTransformer

logger = logging.getLogger(__name__)

_ARTIFACT_NAMES = {
    "frequency_encoder": "frequency_encoder.joblib",
    "ohe_encoder": "ohe_encoder.joblib",
    "scaler": "scaler.joblib",
    "input_features": "input_features.json",
    "output_features": "output_features.json",
    "transformation_metadata": "transformation_metadata.json",
    "preprocessing_report": "preprocessing_report.json",
}


class ArtifactManager:
    """
    Manages all preprocessing artifact I/O.

    Creates versioned run directories under ``artifact_dir``:
        artifacts/preprocessing/
            20260630_163000/   ← run-specific directory
                frequency_encoder.joblib
                ohe_encoder.joblib
                scaler.joblib
                input_features.json
                output_features.json
                transformation_metadata.json
                preprocessing_report.json
            latest/            ← always reflects the newest run (copy)

    Args:
        artifact_dir: Root directory for preprocessing artifacts.
            Read from config.settings.preprocessing.artifact_dir.
        parquet_dir: Root directory for Parquet output files.
            Read from config.settings.preprocessing.output_dir.
        run_id: Timestamp string for the versioned subdirectory
            (format: YYYYMMDD_HHMMSS). Typically passed from the pipeline.
        compression: Parquet compression codec. Default "snappy".
    """

    def __init__(
        self,
        artifact_dir: str,
        parquet_dir: str,
        run_id: str,
        compression: str = "snappy",
    ) -> None:
        self._run_id = run_id
        self._compression = compression

        # Versioned run directory
        self._run_dir = Path(artifact_dir) / run_id
        self._latest_dir = Path(artifact_dir) / "latest"
        self._parquet_dir = Path(parquet_dir)

        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._parquet_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "ArtifactManager initialised | run_dir=%s | parquet_dir=%s",
            self._run_dir,
            self._parquet_dir,
        )

    # ------------------------------------------------------------------
    # Transformer serialisation
    # ------------------------------------------------------------------

    def save_transformer(self, transformer: BaseTransformer, name: str) -> str:
        """
        Serialises a fitted transformer to a joblib file.

        Args:
            transformer: Fitted BaseTransformer subclass instance.
            name: Artifact key from _ARTIFACT_NAMES (e.g. "frequency_encoder").

        Returns:
            str: Absolute path of the saved file.

        Raises:
            ArtifactError: On write failure.
        """
        filename = _ARTIFACT_NAMES.get(name, f"{name}.joblib")
        path = self._run_dir / filename
        try:
            joblib.dump(transformer, path)
            size_kb = path.stat().st_size // 1024
            logger.info("Artifact saved | name=%s | path=%s | size=%dKB", name, path, size_kb)
            return str(path.resolve())
        except Exception as exc:
            raise ArtifactError(
                artifact_name=name,
                artifact_path=str(path),
                detail="joblib.dump failed.",
                original_exc=exc,
            ) from exc

    def load_transformer(self, name: str) -> BaseTransformer:
        """
        Loads a fitted transformer from the run directory.

        Args:
            name: Artifact key (e.g. "frequency_encoder").

        Returns:
            BaseTransformer: Deserialised transformer.

        Raises:
            ArtifactError: If file is not found or corrupted.
        """
        filename = _ARTIFACT_NAMES.get(name, f"{name}.joblib")
        path = self._run_dir / filename
        if not path.exists():
            raise ArtifactError(
                artifact_name=name,
                artifact_path=str(path),
                detail="Artifact file not found.",
            )
        try:
            return joblib.load(path)
        except Exception as exc:
            raise ArtifactError(
                artifact_name=name,
                artifact_path=str(path),
                detail="joblib.load failed — file may be corrupted.",
                original_exc=exc,
            ) from exc

    # ------------------------------------------------------------------
    # JSON metadata
    # ------------------------------------------------------------------

    def save_json(self, data: dict, name: str) -> str:
        """
        Writes a JSON-serialisable dict to the run directory.

        Args:
            data: Dictionary to serialise.
            name: Artifact key (e.g. "input_features").

        Returns:
            str: Absolute path of the saved file.

        Raises:
            ArtifactError: On write failure.
        """
        filename = _ARTIFACT_NAMES.get(name, f"{name}.json")
        path = self._run_dir / filename
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, default=str)
            logger.info("JSON artifact saved | name=%s | path=%s", name, path)
            return str(path.resolve())
        except Exception as exc:
            raise ArtifactError(
                artifact_name=name,
                artifact_path=str(path),
                detail="JSON write failed.",
                original_exc=exc,
            ) from exc

    def load_json(self, name: str) -> dict:
        """
        Reads a JSON artifact from the run directory.

        Raises:
            ArtifactError: If file is not found.
        """
        filename = _ARTIFACT_NAMES.get(name, f"{name}.json")
        path = self._run_dir / filename
        if not path.exists():
            raise ArtifactError(
                artifact_name=name,
                artifact_path=str(path),
                detail="JSON artifact file not found.",
            )
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def save_named_json(self, data: dict, filename: str) -> str:
        """
        Writes a dict to the run directory using an explicit filename
        (not looked up via _ARTIFACT_NAMES).

        Args:
            data: JSON-serialisable dictionary.
            filename: Exact filename (e.g. "dataset_schema.json").

        Returns:
            str: Absolute path of the saved file.

        Raises:
            ArtifactError: On write failure.
        """
        path = self._run_dir / filename
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, default=str)
            logger.info("JSON artifact saved | file=%s | path=%s", filename, path)
            return str(path.resolve())
        except Exception as exc:
            raise ArtifactError(
                artifact_name=filename,
                artifact_path=str(path),
                detail="JSON write failed.",
                original_exc=exc,
            ) from exc

    def save_dataset_schema(
        self,
        dataset_name: str,
        feature_names_input: list[str],
        feature_names_output: list[str],
        target_column: str,
        categorical_features: list[str],
        numerical_features: list[str],
        preprocessing_version: str,
        filename: str,
    ) -> str:
        """
        Saves a machine-readable dataset schema artifact.

        Returns:
            str: Absolute path of saved file.
        """
        schema = {
            "dataset_name": dataset_name,
            "feature_count": len(feature_names_output),
            "target_column": target_column,
            "input_features": feature_names_input,
            "output_features": feature_names_output,
            "categorical_features": categorical_features,
            "numerical_features": numerical_features,
            "datetime_features": [],
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "preprocessing_version": preprocessing_version,
        }
        return self.save_named_json(schema, filename)

    def save_class_distribution(
        self,
        train_y: "pd.Series",
        val_y: "pd.Series",
        test_y: "pd.Series",
        filename: str,
    ) -> str:
        """
        Saves per-fold class distribution statistics.

        Returns:
            str: Absolute path of saved file.
        """
        def _dist(y: "pd.Series") -> dict:
            total = len(y)
            fraud = int(y.sum())
            legit = total - fraud
            return {
                "total": total,
                "legitimate": legit,
                "fraud": fraud,
                "fraud_rate": round(fraud / total * 100, 6) if total > 0 else 0.0,
            }

        data = {
            "train": _dist(train_y),
            "validation": _dist(val_y),
            "test": _dist(test_y),
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        return self.save_named_json(data, filename)

    def save_data_integrity(
        self,
        X_train: "pd.DataFrame",
        X_val: "pd.DataFrame",
        X_test: "pd.DataFrame",
        y_train: "pd.Series",
        y_val: "pd.Series",
        y_test: "pd.Series",
        feature_names_output: list[str],
        target_column: str,
        filename: str,
    ) -> tuple[str, list[str]]:
        """
        Runs pre-save data integrity checks and saves a report.

        Returns:
            tuple: (absolute_path, list_of_integrity_warnings)
        """
        integrity_warnings: list[str] = []
        checks: dict[str, dict] = {}

        # No NaN
        for name, frame in (("X_train", X_train), ("X_val", X_val), ("X_test", X_test)):
            nan_count = int(frame.isna().sum().sum())
            checks[f"no_nan_{name}"] = {"passed": nan_count == 0, "nan_count": nan_count}
            if nan_count > 0:
                integrity_warnings.append(f"NaN values detected in {name}: {nan_count}")

        # No infinite values
        import numpy as np
        for name, frame in (("X_train", X_train), ("X_val", X_val), ("X_test", X_test)):
            numeric = frame.select_dtypes(include="number")
            inf_count = int(np.isinf(numeric.values).sum())
            checks[f"no_inf_{name}"] = {"passed": inf_count == 0, "inf_count": inf_count}
            if inf_count > 0:
                integrity_warnings.append(f"Infinite values detected in {name}: {inf_count}")

        # No duplicate feature names
        dup_names = [c for c in X_train.columns if list(X_train.columns).count(c) > 1]
        checks["no_duplicate_feature_names"] = {
            "passed": len(dup_names) == 0,
            "duplicates": list(set(dup_names)),
        }
        if dup_names:
            integrity_warnings.append(f"Duplicate feature names: {set(dup_names)}")

        # Same feature count across folds
        col_counts = {
            "X_train": len(X_train.columns),
            "X_val": len(X_val.columns),
            "X_test": len(X_test.columns),
        }
        same_cols = len(set(col_counts.values())) == 1
        checks["same_feature_count_across_folds"] = {
            "passed": same_cols,
            "column_counts": col_counts,
        }
        if not same_cols:
            integrity_warnings.append(f"Feature count mismatch across folds: {col_counts}")

        # Target contains exactly two classes
        all_classes = set(y_train.unique()) | set(y_val.unique()) | set(y_test.unique())
        checks["target_binary"] = {
            "passed": all_classes.issubset({0, 1}),
            "unique_values": sorted(int(v) for v in all_classes),
        }
        if not all_classes.issubset({0, 1}):
            integrity_warnings.append(f"Target is not binary: {all_classes}")

        # Feature schema matches output
        schema_match = list(X_train.columns) == feature_names_output
        checks["feature_schema_matches_output"] = {
            "passed": schema_match,
            "expected": feature_names_output[:5],
            "actual": list(X_train.columns)[:5],
        }
        if not schema_match:
            integrity_warnings.append("Feature schema mismatch between report and X_train columns.")

        # All features numeric check
        non_numeric_cols = []
        for name, frame in (("X_train", X_train), ("X_val", X_val), ("X_test", X_test)):
            non_num = [c for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])]
            if non_num:
                non_numeric_cols.extend([(name, c) for c in non_num])

        checks["all_features_numeric"] = {
            "passed": len(non_numeric_cols) == 0,
            "non_numeric_columns": non_numeric_cols,
        }
        if non_numeric_cols:
            integrity_warnings.append(f"Non-numeric feature columns detected: {non_numeric_cols}")

        overall_passed = all(v["passed"] for v in checks.values())
        report = {
            "overall_passed": overall_passed,
            "check_count": len(checks),
            "checks": checks,
            "warnings": integrity_warnings,
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        path = self.save_named_json(report, filename)
        return path, integrity_warnings

    def save_manifest(
        self,
        artifact_paths: dict[str, str],
        run_id: str,
        preprocessing_version: str,
        pipeline_version: str,
        filename: str,
    ) -> str:
        """
        Saves a manifest.json with SHA-256 hashes of all artifacts.

        Returns:
            str: Absolute path of saved manifest.
        """
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
            "artifact_version": run_id,
            "pipeline_version": pipeline_version,
            "preprocessing_version": preprocessing_version,
            "artifacts": entries,
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        return self.save_named_json(manifest, filename)

    @staticmethod
    def _sha256(path: Path) -> str:
        """Computes the SHA-256 hex digest of a file."""
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    # ------------------------------------------------------------------
    # Parquet I/O
    # ------------------------------------------------------------------

    def save_parquet(
        self,
        df: pd.DataFrame | pd.Series,
        name: str,
    ) -> str:
        """
        Writes a DataFrame or Series to a Parquet file.

        Args:
            df: DataFrame or Series to save.
            name: Dataset name (e.g. "X_train", "y_validation").

        Returns:
            str: Absolute path of the saved file.

        Raises:
            ArtifactError: On write failure.
        """
        path = self._parquet_dir / f"{name}.parquet"
        try:
            frame = df.to_frame() if isinstance(df, pd.Series) else df
            frame.to_parquet(path, engine="pyarrow", compression=self._compression, index=False)
            size_mb = path.stat().st_size / (1024 * 1024)
            logger.info(
                "Parquet saved | name=%s | rows=%d | cols=%d | size=%.2fMB | path=%s",
                name,
                len(frame),
                len(frame.columns),
                size_mb,
                path,
            )
            return str(path.resolve())
        except Exception as exc:
            raise ArtifactError(
                artifact_name=name,
                artifact_path=str(path),
                detail="Parquet write failed.",
                original_exc=exc,
            ) from exc

    # ------------------------------------------------------------------
    # Latest directory maintenance
    # ------------------------------------------------------------------

    def sync_latest(self) -> None:
        """
        Copies the current run directory contents to ``latest/``.

        Uses file copy (not symlinks) for maximum cross-platform compatibility
        including Windows, Docker volumes, and network file systems.

        Raises:
            ArtifactError: On copy failure.
        """
        try:
            if self._latest_dir.exists():
                shutil.rmtree(self._latest_dir)
            shutil.copytree(self._run_dir, self._latest_dir)
            logger.info("Latest artifacts synced | latest=%s", self._latest_dir)
        except Exception as exc:
            raise ArtifactError(
                artifact_name="latest",
                artifact_path=str(self._latest_dir),
                detail="Failed to sync latest artifact directory.",
                original_exc=exc,
            ) from exc

    # ------------------------------------------------------------------
    # Path properties
    # ------------------------------------------------------------------

    @property
    def run_dir(self) -> str:
        """Absolute path to the versioned run directory."""
        return str(self._run_dir.resolve())

    @property
    def latest_dir(self) -> str:
        """Absolute path to the latest/ directory."""
        return str(self._latest_dir.resolve())

    @property
    def parquet_dir(self) -> str:
        """Absolute path to the Parquet output directory."""
        return str(self._parquet_dir.resolve())
