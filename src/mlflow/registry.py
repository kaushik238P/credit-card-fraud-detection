"""
Registry-driven schema mapping engine for the MLflow Layer.

Maps parameters, metrics, tags, and environment configurations from upstream
results (TrainingResult and EvaluationResult) without making direct calls to MLflow.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
from typing import Any

from config.settings import settings
from src.evaluation.models import EvaluationResult
from src.training.models import TrainingResult


class SchemaRegistry:
    """
    Translates training/evaluation model results into key-value sets
    expected by the MLflow tracker.
    """

    def __init__(self, project_name: str = "FraudDetection", author: str = "kaush") -> None:
        self.project_name = project_name
        self.author = author

    def map_parameters(self, tr: TrainingResult, er: EvaluationResult) -> dict[str, Any]:
        """
        Extracts execution parameters from upstream payloads.

        Args:
            tr: upstream TrainingResult instance.
            er: upstream EvaluationResult instance.

        Returns:
            dict[str, Any]: Compiled key-value parameters ready for logging.
        """
        # Read from training metadata or defaults
        pinned_dataset = settings.ingestion.zip_csv_target or "fraudTrain.csv"
        params = {
            "Model Name": tr.metadata.model_type.value,
            "Hyperparameters": tr.metadata.hyperparameters,
            "Random Seed": tr.metadata.random_seed,
            "Dataset Name": pinned_dataset,
            "Dataset Version": tr.metadata.preprocessing_version,
            "Dataset Hash": tr.metadata.dataset_hash,
            "Feature Schema Hash": tr.metadata.feature_schema_hash,
            "Training Version": tr.metadata.training_version,
            "Evaluation Version": er.report.model_version,
            "Preprocessing Version": tr.metadata.preprocessing_version,
            "Pipeline Version": tr.metadata.training_version,
            "Artifact Version": er.report.preprocessing_version,
        }
        return params

    def map_metrics(self, er: EvaluationResult, tr: TrainingResult) -> dict[str, float]:
        """
        Collects metrics across core scores, business rates, and profiling values.

        Args:
            er: upstream EvaluationResult instance.
            tr: upstream TrainingResult instance.

        Returns:
            dict[str, float]: Compiled metrics mapping.
        """
        # Core performance metrics on test fold
        test_m = er.report.test_metrics
        bus_m = er.report.business_metrics
        tr_meta = tr.metadata
        er_rep = er.report

        metrics = {
            "PR_AUC": float(test_m.get("PR_AUC", 0.0)),
            "ROC_AUC": float(test_m.get("ROC_AUC", 0.0)),
            "Precision": float(test_m.get("Precision", 0.0)),
            "Recall": float(test_m.get("Recall", 0.0)),
            "F1": float(test_m.get("F1", 0.0)),
            "Specificity": float(test_m.get("Specificity", 0.0)),
            "Balanced Accuracy": float(test_m.get("Balanced_Accuracy", 0.0)),
            "Matthews Correlation Coefficient": float(test_m.get("MCC", 0.0)),
            "Log Loss": float(test_m.get("Log_Loss", 0.0)),
            "Brier Score": float(test_m.get("Brier_Score", 0.0)),
            "Fraud Detection Rate": float(bus_m.get("Fraud_Detection_Rate", {}).get("value", 0.0)),
            "False Positive Rate": float(bus_m.get("False_Positive_Rate", {}).get("value", 0.0)),
            "Fraud Capture Rate": float(bus_m.get("Fraud_Capture_Rate", {}).get("value", 0.0)),
            "Alert Rate": float(bus_m.get("Alert_Rate", {}).get("value", 0.0)),
            "Precision_at_K": float(bus_m.get("Precision_At_K", {}).get("value", 0.0)),
            "Optimized Threshold": float(er_rep.threshold_result.optimized_threshold),
            "Training Time": float(tr.report.duration_ms / 1000.0),
            "Evaluation Time": float(er_rep.evaluation_duration / 1000.0),
        }
        return metrics

    def map_tags(self, tr: TrainingResult, er: EvaluationResult, env_name: str = "development") -> dict[str, str]:
        """
        Assembles tag attributes, Git markers, and target environments.

        Args:
            tr: upstream TrainingResult instance.
            er: upstream EvaluationResult instance.
            env_name: targeted deployment environment.

        Returns:
            dict[str, str]: Compiled key-value tags mapping.
        """
        # Resolve git markers safely
        git_commit = self._git_run("git rev-parse HEAD")
        git_branch = self._git_run("git rev-parse --abbrev-ref HEAD")
        git_tag = self._git_run("git describe --tags --abbrev=0")

        pinned_dataset = settings.ingestion.zip_csv_target or "fraudTrain.csv"
        tags = {
            "Project": self.project_name,
            "Experiment": f"{self.project_name}_Experiment",
            "Model": tr.metadata.model_type.value,
            "Dataset": pinned_dataset,
            "Dataset Version": tr.metadata.preprocessing_version,
            "Dataset Hash": tr.metadata.dataset_hash or "",
            "Feature Schema Hash": tr.metadata.feature_schema_hash,
            "Pipeline Version": tr.metadata.training_version,
            "Training Version": tr.metadata.model_version,
            "Evaluation Version": er.report.model_version,
            "Environment": env_name,
            "Author": self.author,
            "Git Commit": git_commit,
            "Git Branch": git_branch,
            "Git Tag": git_tag,
        }

        # Add OS and CPU details (Refinement 6 / System details)
        tags.update(self.get_environment_metadata())

        return tags

    @staticmethod
    def get_environment_metadata() -> dict[str, str]:
        """
        Resolves Python build versions, system architecture, and package configurations.
        """
        packages = ["xgboost", "scikit-learn", "numpy", "pandas", "mlflow"]
        pkg_versions = {}
        for pkg in packages:
            try:
                pkg_versions[f"dep.{pkg}"] = importlib.metadata.version(pkg)
            except importlib.metadata.PackageNotFoundError:
                pkg_versions[f"dep.{pkg}"] = "not_installed"

        # Safe RAM query
        ram_gb = "N/A"
        try:
            # Try psutil if present
            import psutil
            ram_gb = f"{psutil.virtual_memory().total / (1024**3):.1f} GB"
        except ImportError:
            # Fallback to system query or maxsize representation
            ram_gb = f"{sys.maxsize / (1024**3 * 8):.1f} GB"

        meta = {
            "env.python_version": platform.python_version(),
            "env.operating_system": platform.system(),
            "env.platform": platform.platform(),
            "env.architecture": platform.architecture()[0],
            "env.cpu_count": str(os.cpu_count() or 1),
            "env.ram": ram_gb,
        }
        meta.update(pkg_versions)
        return meta

    @staticmethod
    def _git_run(cmd: str) -> str:
        """Helper to call git CLI without generating run failures."""
        try:
            res = subprocess.run(
                cmd.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if res.returncode == 0:
                return res.stdout.strip()
        except Exception:
            pass
        return ""
