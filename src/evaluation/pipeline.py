"""
Orchestration pipeline for the Evaluation Layer.

Main Class:
    EvaluationPipeline

Internal Helper:
    _ArtifactManager (handles all serialization, manifest generation, latest syncing)
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import settings
from src.evaluation.exceptions import (
    MetricError,
    PipelineError,
    PlotError,
    ReportError,
    ThresholdError,
)
from src.evaluation.metrics import (
    compute_false_negative_rate,
    compute_false_positive_rate,
    compute_fraud_capture_rate,
    compute_fraud_detection_rate,
    compute_precision_at_k,
    compute_potential_fraud_amount_detected,
    compute_transactions_flagged,
)
from src.evaluation.models import (
    BusinessMetric,
    EvaluationArtifacts,
    EvaluationReport,
    EvaluationResult,
    MetricResult,
    PlotArtifact,
    ThresholdResult,
)
from src.evaluation.registry import METRIC_REGISTRY, PLOT_REGISTRY
from src.evaluation.report import EvaluationReportGenerator
from src.evaluation.threshold import ThresholdOptimizer
from src.preprocessing.models import PreprocessedDataset
from src.training.models import TrainingResult

logger = logging.getLogger(__name__)


# ===========================================================================
# Internal ArtifactManager
# ===========================================================================


class _ArtifactManager:
    """
    Handles all Evaluation Layer serialization, filesystem layout, manifest,
    and copying versioned artifacts to the 'latest' folder.
    """

    def __init__(self, output_dir: str, run_id: str) -> None:
        self.run_id = run_id
        self.output_dir = Path(output_dir)
        self.run_dir = self.output_dir / run_id
        self.latest_dir = self.output_dir / "latest"
        
        # Ensure directory structures
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "plots").mkdir(parents=True, exist_ok=True)
        
        logger.info("Evaluation ArtifactManager initialised at %s", self.run_dir)

    def save_json(self, data: Any, filename: str) -> str:
        """Saves a JSON file in the run directory."""
        path = self.run_dir / filename
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, default=str)
            return str(path.resolve())
        except Exception as exc:
            logger.warning("Failed to serialize JSON '%s': %s", filename, exc)
            return ""

    def save_markdown(self, content: str, filename: str) -> str:
        """Saves a Markdown file in the run directory."""
        path = self.run_dir / filename
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(content)
            return str(path.resolve())
        except Exception as exc:
            logger.warning("Failed to save markdown '%s': %s", filename, exc)
            return ""

    def save_csv(self, rows: list[list[Any]], filename: str) -> str:
        """Saves a CSV file in the run directory."""
        path = self.run_dir / filename
        try:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerows(rows)
            return str(path.resolve())
        except Exception as exc:
            logger.warning("Failed to save CSV '%s': %s", filename, exc)
            return ""

    def save_manifest(
        self,
        artifact_paths: dict[str, str],
        plot_paths: dict[str, str],
        dataset_hash: str | None,
        feature_schema_hash: str,
        model_version: str,
        preprocessing_version: str,
    ) -> str:
        """
        Generates and saves a manifest.json containing SHA-256 hashes of all artifacts.
        """
        manifest_entries = []
        all_paths = {**artifact_paths, **plot_paths}
        
        for key, path_str in all_paths.items():
            if not path_str:
                continue
            p = Path(path_str)
            if p.exists() and p.is_file():
                sha256 = self._compute_sha256(p)
                size_bytes = p.stat().st_size
                mtime = datetime.fromtimestamp(
                    p.stat().st_mtime, tz=timezone.utc
                ).isoformat()
                
                manifest_entries.append({
                    "name": key,
                    "filename": p.name,
                    "sha256": sha256,
                    "size_bytes": size_bytes,
                    "created_at": mtime,
                })

        manifest = {
            "model_version": model_version,
            "preprocessing_version": preprocessing_version,
            "dataset_hash": dataset_hash,
            "feature_schema_hash": feature_schema_hash,
            "artifacts": manifest_entries,
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        
        path = self.run_dir / "manifest.json"
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(manifest, fh, indent=2, default=str)
            return str(path.resolve())
        except Exception as exc:
            logger.warning("Failed to save manifest.json: %s", exc)
            return ""

    def sync_latest(self) -> str:
        """Copies versioned run directory to latest/ using copytree (Windows safe)."""
        try:
            if self.latest_dir.exists():
                shutil.rmtree(self.latest_dir)
            shutil.copytree(self.run_dir, self.latest_dir)
            logger.info("Latest evaluation artifacts synced to %s", self.latest_dir)
            return str(self.latest_dir.resolve())
        except Exception as exc:
            logger.warning("Failed to sync latest directory: %s", exc)
            return ""

    @staticmethod
    def _compute_sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()


# ===========================================================================
# EvaluationPipeline
# ===========================================================================


class EvaluationPipeline:
    """
    Orchestrates the model evaluation workflow.

    Accepts TrainingResult + PreprocessedDataset, optimizes the threshold,
    computes metrics/business metrics, generates diagnostic plots, compiles
    reports, and serializes everything using the internal ArtifactManager.
    """

    def __init__(self, cfg: Any = None) -> None:
        self._cfg = cfg or settings.evaluation

    def evaluate(
        self,
        training_result: TrainingResult,
        preprocessed_dataset: PreprocessedDataset,
    ) -> EvaluationResult:
        """
        Executes the full evaluation pipeline.

        Args:
            training_result: Primary payload from the Training Layer.
            preprocessed_dataset: Primary payload from the Preprocessing Layer.

        Returns:
            EvaluationResult: Populated evaluation report and artifact metadata.

        Raises:
            PipelineError: If guards or orchestration fails.
            ThresholdError: If threshold optimization fails.
            MetricError: If a critical metric computation fails.
        """
        _global_start = time.perf_counter()
        run_id = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        warnings_log: list[str] = []
        timeline: list[dict[str, Any]] = []

        logger.info(
            "Evaluation pipeline started | run_id=%s | model_version=%s | strategy=%s",
            run_id,
            training_result.metadata.model_version,
            self._cfg.threshold_strategy,
        )

        # ── 1. Pipeline Guards ────────────────────────────────────────────
        self._guard(training_result, preprocessed_dataset)

        estimator = training_result.estimator
        val_y = preprocessed_dataset.y_validation.values
        test_y = preprocessed_dataset.y_test.values
        X_test = preprocessed_dataset.X_test

        # Retrieve validation cache arrays
        val_preds_default = training_result.validation_predictions
        val_proba = training_result.validation_probabilities

        # Resolve metadata
        model_version = training_result.metadata.model_version
        preprocessing_version = training_result.metadata.preprocessing_version
        dataset_hash = training_result.metadata.dataset_hash
        feature_schema_hash = training_result.metadata.feature_schema_hash

        # ── 2. Threshold Optimization ─────────────────────────────────────
        step_start = time.perf_counter()
        step_start_iso = datetime.now(tz=timezone.utc).isoformat()

        optimizer = ThresholdOptimizer(
            strategy=self._cfg.threshold_strategy,
            default_threshold=self._cfg.default_threshold,
            target_recall=self._cfg.target_recall,
            target_precision=self._cfg.target_precision,
        )
        threshold_res = optimizer.optimize(val_y, val_proba)
        optimized_threshold = threshold_res.optimized_threshold

        timeline.append({
            "step": "Threshold Optimization",
            "start_time": step_start_iso,
            "end_time": datetime.now(tz=timezone.utc).isoformat(),
            "duration_ms": round((time.perf_counter() - step_start) * 1000, 2),
        })
        logger.info("Threshold optimized | value=%.4f", optimized_threshold)

        # Apply threshold to test set & validation set in-memory
        test_proba = np.asarray(estimator.predict_proba(X_test)[:, 1])
        test_preds = (test_proba >= optimized_threshold).astype(int)
        
        val_preds_opt = (val_proba >= optimized_threshold).astype(int)

        # ── 3. Metric Calculations (Validation vs Test) ───────────────────
        step_start = time.perf_counter()
        step_start_iso = datetime.now(tz=timezone.utc).isoformat()

        val_metrics: dict[str, Any] = {}
        test_metrics: dict[str, Any] = {}

        for entry in METRIC_REGISTRY:
            try:
                # Validation Set
                if entry.needs_proba:
                    val_score = val_proba
                else:
                    val_score = val_preds_opt
                val_val = entry.fn(val_y, val_score)
                val_metrics[entry.name] = val_val

                # Test Set
                if entry.needs_proba:
                    test_score = test_proba
                else:
                    test_score = test_preds
                test_val = entry.fn(test_y, test_score)
                test_metrics[entry.name] = test_val

            except Exception as exc:
                raise MetricError(
                    metric_name=entry.name,
                    detail=str(exc),
                    context={"original_exc": str(exc)},
                ) from exc

        timeline.append({
            "step": "Metric Calculation",
            "start_time": step_start_iso,
            "end_time": datetime.now(tz=timezone.utc).isoformat(),
            "duration_ms": round((time.perf_counter() - step_start) * 1000, 2),
        })
        logger.info("Core metrics calculated | metrics=%d", len(val_metrics))

        # ── 4. Business Metrics ──────────────────────────────────────────
        step_start = time.perf_counter()
        step_start_iso = datetime.now(tz=timezone.utc).isoformat()

        # Retrieve amount feature column from preprocessed dataset
        val_amounts = self._get_amounts(preprocessed_dataset.X_validation)
        test_amounts = self._get_amounts(X_test)

        business_metrics_raw: dict[str, BusinessMetric] = {}

        # FDR, FPR, FNR, Capture Rate
        fdr = compute_fraud_detection_rate(test_y, test_preds)
        business_metrics_raw["Fraud_Detection_Rate"] = BusinessMetric(
            name="Fraud_Detection_Rate",
            value=fdr,
            description="Percentage of actual fraud cases caught (Recall).",
        )

        fpr = compute_false_positive_rate(test_y, test_preds)
        business_metrics_raw["False_Positive_Rate"] = BusinessMetric(
            name="False_Positive_Rate",
            value=fpr,
            description="Percentage of legitimate transactions incorrectly flagged.",
        )

        fnr = compute_false_negative_rate(test_y, test_preds)
        business_metrics_raw["False_Negative_Rate"] = BusinessMetric(
            name="False_Negative_Rate",
            value=fnr,
            description="Percentage of fraud transactions missed.",
        )

        fcr = compute_fraud_capture_rate(test_y, test_preds, test_amounts)
        business_metrics_raw["Fraud_Capture_Rate"] = BusinessMetric(
            name="Fraud_Capture_Rate",
            value=fcr,
            description="Percentage of total fraud transaction monetary volume saved.",
        )

        # Alert Volume Metrics (Refinement 4)
        total_tx = len(test_preds)
        flagged_tx = int(test_preds.sum())
        alert_rate = float(flagged_tx / total_tx) if total_tx > 0 else 0.0

        business_metrics_raw["Transactions_Flagged"] = BusinessMetric(
            name="Transactions_Flagged",
            value=float(flagged_tx),
            description="Total number of transactions flagged as fraud by the model.",
        )
        business_metrics_raw["Alert_Rate"] = BusinessMetric(
            name="Alert_Rate",
            value=alert_rate,
            description="Ratio of flagged transactions to total transactions.",
        )
        business_metrics_raw["Alerts_per_1_000_Transactions"] = BusinessMetric(
            name="Alerts_per_1_000_Transactions",
            value=alert_rate * 1000,
            description="Number of alerts generated per 1,000 transactions.",
        )
        business_metrics_raw["Alerts_per_10_000_Transactions"] = BusinessMetric(
            name="Alerts_per_10_000_Transactions",
            value=alert_rate * 10000,
            description="Number of alerts generated per 10,000 transactions.",
        )
        business_metrics_raw["Alerts_per_100_000_Transactions"] = BusinessMetric(
            name="Alerts_per_100_000_Transactions",
            value=alert_rate * 100000,
            description="Number of alerts generated per 100,000 transactions.",
        )

        # Precision@K Validation (Refinement 5)
        k = self._cfg.top_k
        actual_k = min(k, len(test_y))
        idx = np.argsort(test_proba)[::-1]
        top_k_y_true = test_y[idx[:actual_k]]
        fraud_cases = int(top_k_y_true.sum())
        legit_cases = int(actual_k - fraud_cases)
        total_frauds = int(test_y.sum())

        prec_at_k = compute_precision_at_k(test_y, test_proba, k)
        business_metrics_raw["Precision_At_K"] = BusinessMetric(
            name="Precision_At_K",
            value=prec_at_k,
            description=f"Precision score evaluating only top {k} predicted scores.",
            metadata={
                "k": k,
                "fraud_cases_inside_top_k": fraud_cases,
                "legitimate_cases_inside_top_k": legit_cases,
                "total_frauds_in_test_fold": total_frauds,
            },
        )

        # Potential Fraud Amount Detected (Refinement 3)
        pot_detected = compute_potential_fraud_amount_detected(test_y, test_preds, test_amounts)
        business_metrics_raw["Potential_Fraud_Amount_Detected"] = BusinessMetric(
            name="Potential_Fraud_Amount_Detected",
            value=pot_detected,
            description="Sum of transaction amounts for correctly detected fraud transactions. Do NOT imply actual monetary savings.",
        )

        business_metrics = {k: v.to_dict() for k, v in business_metrics_raw.items()}

        timeline.append({
            "step": "Business Metrics",
            "start_time": step_start_iso,
            "end_time": datetime.now(tz=timezone.utc).isoformat(),
            "duration_ms": round((time.perf_counter() - step_start) * 1000, 2),
        })
        logger.info("Business metrics calculated.")

        # ── 5. Model-Native Feature Importance ────────────────────────────
        step_start = time.perf_counter()
        step_start_iso = datetime.now(tz=timezone.utc).isoformat()

        feature_importance = self._get_feature_importances(
            estimator, list(preprocessed_dataset.feature_names)
        )

        timeline.append({
            "step": "Feature Importance",
            "start_time": step_start_iso,
            "end_time": datetime.now(tz=timezone.utc).isoformat(),
            "duration_ms": round((time.perf_counter() - step_start) * 1000, 2),
        })

        # ── 6. Plot Generation ────────────────────────────────────────────
        step_start = time.perf_counter()
        step_start_iso = datetime.now(tz=timezone.utc).isoformat()

        # Set up versioned file path mapping for ArtifactManager
        artifact_manager = _ArtifactManager(
            output_dir=self._cfg.output_directory,
            run_id=run_id,
        )

        plot_artifacts_list: list[PlotArtifact] = []
        plot_paths: dict[str, str] = {}

        # Confusion Matrix dictionary (Test set) for the plot
        test_cm = test_metrics.get("Confusion_Matrix", {"tn": 0, "fp": 0, "fn": 0, "tp": 0})

        for entry in PLOT_REGISTRY:
            dest_path = artifact_manager.run_dir / "plots" / entry.file_name
            try:
                # Execute lambda wrapper matching general plot signature
                entry.fn(
                    test_y,
                    test_preds,
                    test_proba,
                    feature_importance,
                    test_cm,
                    str(dest_path),
                )
                
                art = PlotArtifact(
                    plot_name=entry.name,
                    file_name=entry.file_name,
                    absolute_path=str(dest_path.resolve()),
                )
                plot_artifacts_list.append(art)
                plot_paths[entry.name] = str(dest_path.resolve())

            except PlotError as exc:
                warnings_log.append(exc.message)
                logger.warning("Plot generation skipped: %s", exc.message)
            except Exception as exc:
                msg = f"Unexpected error generating plot '{entry.name}': {exc}"
                warnings_log.append(msg)
                logger.warning(msg)

        timeline.append({
            "step": "Plot Generation",
            "start_time": step_start_iso,
            "end_time": datetime.now(tz=timezone.utc).isoformat(),
            "duration_ms": round((time.perf_counter() - step_start) * 1000, 2),
        })
        logger.info("Plot generation completed | successes=%d", len(plot_artifacts_list))

        # ── 7. Report Compilation ────────────────────────────────────────
        step_start = time.perf_counter()
        step_start_iso = datetime.now(tz=timezone.utc).isoformat()

        duration_ms = (time.perf_counter() - _global_start) * 1000

        # Refinement 1: Fold statistics
        def _compute_fold_stats(y: pd.Series | np.ndarray) -> dict[str, Any]:
            y_arr = np.asarray(y)
            total = len(y_arr)
            fraud = int(y_arr.sum())
            legit = total - fraud
            rate = float(fraud / total * 100) if total > 0 else 0.0
            return {
                "rows": total,
                "legitimate_count": legit,
                "fraud_count": fraud,
                "fraud_rate": rate,
            }

        fold_statistics = {
            "Train": _compute_fold_stats(preprocessed_dataset.y_train),
            "Validation": _compute_fold_stats(preprocessed_dataset.y_validation),
            "Test": _compute_fold_stats(preprocessed_dataset.y_test),
        }

        # Refinement 2: Threshold comparison
        def _compute_comp_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
            from src.evaluation.metrics import (
                compute_precision,
                compute_recall,
                compute_f1,
                compute_specificity,
                compute_false_positive_rate,
                compute_false_negative_rate,
                compute_fraud_detection_rate,
            )
            return {
                "precision": compute_precision(y_true, y_pred),
                "recall": compute_recall(y_true, y_pred),
                "f1": compute_f1(y_true, y_pred),
                "specificity": compute_specificity(y_true, y_pred),
                "false_positive_rate": compute_false_positive_rate(y_true, y_pred),
                "false_negative_rate": compute_false_negative_rate(y_true, y_pred),
                "fraud_detection_rate": compute_fraud_detection_rate(y_true, y_pred),
            }

        test_preds_default = (test_proba >= 0.5).astype(int)
        test_preds_opt = (test_proba >= optimized_threshold).astype(int)

        threshold_comparison = {
            "default_threshold_0.5": _compute_comp_metrics(test_y, test_preds_default),
            "optimized_threshold": _compute_comp_metrics(test_y, test_preds_opt),
        }

        report = EvaluationReport(
            validation_metrics=val_metrics,
            test_metrics=test_metrics,
            business_metrics=business_metrics,
            threshold_result=threshold_res,
            feature_importance=feature_importance,
            plots=tuple(plot_artifacts_list),
            warnings=tuple(warnings_log),
            dataset_hash=dataset_hash,
            feature_schema_hash=feature_schema_hash,
            preprocessing_version=preprocessing_version,
            model_version=model_version,
            evaluation_duration=duration_ms,
            generated_at=datetime.now(tz=timezone.utc).isoformat(),
            fold_statistics=fold_statistics,
            threshold_comparison=threshold_comparison,
        )

        timeline.append({
            "step": "Report Generation",
            "start_time": step_start_iso,
            "end_time": datetime.now(tz=timezone.utc).isoformat(),
            "duration_ms": round((time.perf_counter() - step_start) * 1000, 2),
        })

        # ── 8. Serialization ──────────────────────────────────────────────
        step_start = time.perf_counter()
        step_start_iso = datetime.now(tz=timezone.utc).isoformat()

        # Files mapping
        saved_paths: dict[str, str] = {}

        # Save metrics lists
        saved_paths["metrics"] = artifact_manager.save_json(
            {"validation": val_metrics, "test": test_metrics}, "metrics.json"
        )
        saved_paths["business_metrics"] = artifact_manager.save_json(
            business_metrics, "business_metrics.json"
        )
        saved_paths["threshold"] = artifact_manager.save_json(
            threshold_res.to_dict(), "threshold.json"
        )

        # Save feature importance CSV & JSON
        sorted_importance = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )
        feat_rows = [["feature", "importance"]] + [
            [f, val] for f, val in sorted_importance
        ]
        saved_paths["feature_importance_csv"] = artifact_manager.save_csv(
            feat_rows, "feature_importance.csv"
        )
        saved_paths["feature_importance_json"] = artifact_manager.save_json(
            feature_importance, "feature_importance.json"
        )

        # Save confusion matrix CSV
        cm_rows = [
            ["actual", "predicted", "count"],
            ["0", "0", test_cm.get("tn", 0)],
            ["0", "1", test_cm.get("fp", 0)],
            ["1", "0", test_cm.get("fn", 0)],
            ["1", "1", test_cm.get("tp", 0)],
        ]
        saved_paths["confusion_matrix_csv"] = artifact_manager.save_csv(
            cm_rows, "confusion_matrix.csv"
        )

        # Save prediction distribution JSON
        pred_dist = {
            "threshold": round(optimized_threshold, 4),
            "predicted_fraud": int(test_preds.sum()),
            "predicted_legitimate": int(len(test_preds) - test_preds.sum()),
        }
        saved_paths["prediction_distribution"] = artifact_manager.save_json(
            pred_dist, "prediction_distribution.json"
        )

        # Save prediction summary JSON (Refinement 7)
        pred_summary = {
            "total_predictions": len(test_preds),
            "predicted_fraud": int(test_preds.sum()),
            "predicted_legitimate": int(len(test_preds) - test_preds.sum()),
            "actual_fraud": int(test_y.sum()),
            "actual_legitimate": int(len(test_y) - test_y.sum()),
            "optimized_threshold": float(optimized_threshold),
        }
        saved_paths["prediction_summary"] = artifact_manager.save_json(
            pred_summary, "prediction_summary.json"
        )

        # Save report JSON
        saved_paths["report_json"] = artifact_manager.save_json(
            report.to_dict(), "evaluation_report.json"
        )

        # Save report Markdown
        report_md_str = EvaluationReportGenerator.generate_markdown(report)
        saved_paths["report_markdown"] = artifact_manager.save_markdown(
            report_md_str, "evaluation_report.md"
        )

        timeline.append({
            "step": "Serialization",
            "start_time": step_start_iso,
            "end_time": datetime.now(tz=timezone.utc).isoformat(),
            "duration_ms": round((time.perf_counter() - step_start) * 1000, 2),
        })

        # ── 9. Manifest ───────────────────────────────────────────────────
        step_start = time.perf_counter()
        step_start_iso = datetime.now(tz=timezone.utc).isoformat()

        manifest_path = artifact_manager.save_manifest(
            artifact_paths=saved_paths,
            plot_paths=plot_paths,
            dataset_hash=dataset_hash,
            feature_schema_hash=feature_schema_hash,
            model_version=model_version,
            preprocessing_version=preprocessing_version,
        )

        timeline.append({
            "step": "Manifest",
            "start_time": step_start_iso,
            "end_time": datetime.now(tz=timezone.utc).isoformat(),
            "duration_ms": round((time.perf_counter() - step_start) * 1000, 2),
        })

        # ── 10. Latest Sync ────────────────────────────────────────────────
        step_start = time.perf_counter()
        step_start_iso = datetime.now(tz=timezone.utc).isoformat()

        latest_dir = artifact_manager.sync_latest()

        timeline.append({
            "step": "Latest Sync",
            "start_time": step_start_iso,
            "end_time": datetime.now(tz=timezone.utc).isoformat(),
            "duration_ms": round((time.perf_counter() - step_start) * 1000, 2),
        })

        # Final return assembly
        arts = EvaluationArtifacts(
            report_json_path=saved_paths["report_json"],
            report_md_path=saved_paths["report_markdown"],
            metrics_json_path=saved_paths["metrics"],
            business_metrics_json_path=saved_paths["business_metrics"],
            threshold_json_path=saved_paths["threshold"],
            feature_importance_csv_path=saved_paths["feature_importance_csv"],
            feature_importance_json_path=saved_paths["feature_importance_json"],
            confusion_matrix_csv_path=saved_paths["confusion_matrix_csv"],
            prediction_distribution_json_path=saved_paths["prediction_distribution"],
            prediction_summary_json_path=saved_paths["prediction_summary"],
            manifest_path=manifest_path,
            plot_paths=plot_paths,
            artifact_dir=str(artifact_manager.run_dir.resolve()),
            latest_dir=latest_dir,
        )

        # Update report timeline and finalize
        report = EvaluationReport(
            validation_metrics=val_metrics,
            test_metrics=test_metrics,
            business_metrics=business_metrics,
            threshold_result=threshold_res,
            feature_importance=feature_importance,
            plots=tuple(plot_artifacts_list),
            warnings=tuple(warnings_log),
            dataset_hash=dataset_hash,
            feature_schema_hash=feature_schema_hash,
            preprocessing_version=preprocessing_version,
            model_version=model_version,
            evaluation_duration=round((time.perf_counter() - _global_start) * 1000, 2),
            generated_at=report.generated_at,
            fold_statistics=fold_statistics,
            threshold_comparison=threshold_comparison,
        )

        logger.info(
            "Evaluation pipeline complete | run_id=%s | duration=%.1fms | warnings=%d",
            run_id,
            report.evaluation_duration,
            len(warnings_log),
        )

        return EvaluationResult(report=report, artifacts=arts)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _guard(
        self,
        training_result: TrainingResult,
        preprocessed_dataset: PreprocessedDataset,
    ) -> None:
        """Guards against invalid pipeline inputs, raising PipelineError if invalid."""
        if training_result is None:
            raise PipelineError("GUARD", "TrainingResult is None.")
        if preprocessed_dataset is None:
            raise PipelineError("GUARD", "PreprocessedDataset is None.")
        if training_result.estimator is None:
            raise PipelineError("GUARD", "Estimator inside TrainingResult is None.")
        if training_result.validation_predictions is None or len(training_result.validation_predictions) == 0:
            raise PipelineError("GUARD", "Validation predictions cache is empty.")
        if training_result.validation_probabilities is None or len(training_result.validation_probabilities) == 0:
            raise PipelineError("GUARD", "Validation probabilities cache is empty.")
        if preprocessed_dataset.X_test is None or len(preprocessed_dataset.X_test) == 0:
            raise PipelineError("GUARD", "Test features matrix X_test is empty.")
        if preprocessed_dataset.y_test is None or len(preprocessed_dataset.y_test) == 0:
            raise PipelineError("GUARD", "Test labels vector y_test is empty.")
        if not self._cfg.threshold_strategy:
            raise PipelineError("GUARD", "Threshold optimization strategy not configured.")
        if not training_result.metadata.feature_schema_hash:
            raise PipelineError("GUARD", "Feature schema hash is missing in model metadata.")

    @staticmethod
    def _get_amounts(X: pd.DataFrame) -> np.ndarray:
        """Helper to extract unscaled transaction amount column values safely."""
        if "amt" in X.columns:
            return X["amt"].values
        return np.ones(len(X))

    @staticmethod
    def _get_feature_importances(
        estimator: Any,
        feature_names: list[str],
    ) -> dict[str, float]:
        """Resolves model-native feature importances, using absolute coefficients as fallback."""
        importances = {}
        try:
            if hasattr(estimator, "feature_importances_"):
                scores = estimator.feature_importances_
                for col, val in zip(feature_names, scores):
                    importances[col] = float(val)
            elif hasattr(estimator, "coef_"):
                # LogisticRegression / linear models coefficients representation
                scores = np.abs(estimator.coef_[0])
                for col, val in zip(feature_names, scores):
                    importances[col] = float(val)
            else:
                for col in feature_names:
                    importances[col] = 0.0
        except Exception as exc:
            logger.warning("Failed to extract model-native feature importances: %s", exc)
            for col in feature_names:
                importances[col] = 0.0
        return importances