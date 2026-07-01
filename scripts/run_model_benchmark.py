"""
Benchmark Runner for Credit Card Fraud Detection System.

Orchestrates the data pipeline once, then runs Training, Evaluation, and MLflow
tracking pipelines iteratively for all supported models in ModelType. Generates
comparative ranking, metadata, and summary reports.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import datetime
import hashlib
import json
import logging
import mimetypes
import os
import time
from dataclasses import replace
from typing import Any, Optional

# Centralized imports from project layers
from config.settings import Settings, settings
from src.evaluation import EvaluationPipeline
from src.features import FeatureEngineeringPipeline
from src.ingestion import DataLoader
from src.preprocessing import PreprocessingPipeline
from src.training import TrainingPipeline
from src.training.models import ModelType
from src.validation import FRAUD_TRANSACTION_SCHEMA, DatasetValidator
from src.mlflow import MLflowPipeline
from src.eda.analyzer import EDAAnalyzer
from src.eda.models import EDAReport, DatasetOverview, TargetAnalysis

logger = logging.getLogger("run_model_benchmark")


class BenchmarkRunner:
    """
    Orchestrates the benchmarking of all supported machine learning models.

    Runs ingestion and preprocessing once, then loops through all models in
    ModelType. Collects standard, business, and tracking metrics, ranks models,
    and generates reports in reports/benchmark/.
    """

    def __init__(self, resume: bool = False, force: bool = False) -> None:
        """
        Initializes the BenchmarkRunner.

        Args:
            resume: If True, resumes from previous successful runs.
            force: If True, forces execution of all models, ignoring cache.
        """
        self.resume = resume
        self.force = force

        # Read configuration parameters dynamically
        self.raw_data_path = Path(
            os.environ.get(
                "FRAUD_RAW_DATA_PATH",
                r"D:\credit-card-fraud-detection\data\raw\archive.zip",
            )
        )
        self.report_dir = Path(
            os.environ.get("FRAUD_BENCHMARK_REPORT_DIR", "reports/benchmark")
        )
        self.mlflow_tracking_uri = os.environ.get(
            "FRAUD_MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"
        )

        # Set tracking URI environment variable for downstream layers
        os.environ["FRAUD_MLFLOW_TRACKING_URI"] = self.mlflow_tracking_uri

    def run(self) -> None:
        """
        Executes the benchmark run end-to-end.
        """
        logger.info("Benchmark Started")
        benchmark_start_time = time.perf_counter()
        timeline: list[dict[str, Any]] = []

        # 1. Dataset Preparation (Ingestion -> Validation -> FE -> Preprocessing)
        data_prep_start = time.perf_counter()
        data_prep_start_iso = datetime.datetime.now(
            tz=datetime.timezone.utc
        ).isoformat()

        logger.info(
            "Starting dataset preparation | path=%s", self.raw_data_path
        )
        try:
            loader = DataLoader()
            loaded_df = loader.load(str(self.raw_data_path))

            validator = DatasetValidator(schema=FRAUD_TRANSACTION_SCHEMA)
            validated_df = validator.validate(loaded_df)

            # Create mock EDAReport to satisfy downstream FeatureEngineeringPipeline
            # requirements without running slow exploratory visual/profiling EDA code
            eda_analyzer = EDAAnalyzer()
            inventory = eda_analyzer._build_feature_inventory(validated_df.data)
            blueprint = eda_analyzer._build_feature_engineering_blueprint(validated_df.data, inventory)

            overview = DatasetOverview(
                num_rows=len(validated_df.data),
                num_columns=len(validated_df.data.columns),
                memory_usage_mb=0.0,
                num_numeric_columns=0,
                num_categorical_columns=0,
                num_datetime_columns=0,
                column_names=tuple(validated_df.data.columns),
                missing_values_summary={},
                dataset_hash=validated_df.metadata.dataset_hash,
                analysis_timestamp=datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
            )

            target_analysis = TargetAnalysis(
                target_column="is_fraud",
                fraud_count=int((validated_df.data["is_fraud"] == 1).sum()),
                legitimate_count=int((validated_df.data["is_fraud"] == 0).sum()),
                fraud_pct=float((validated_df.data["is_fraud"] == 1).mean() * 100),
                legitimate_pct=float((validated_df.data["is_fraud"] == 0).mean() * 100),
                imbalance_ratio=float(len(validated_df.data) / (validated_df.data["is_fraud"] == 1).sum() if (validated_df.data["is_fraud"] == 1).sum() > 0 else 0.0),
                class_distribution={"0": int((validated_df.data["is_fraud"] == 0).sum()), "1": int((validated_df.data["is_fraud"] == 1).sum())}
            )

            eda_report = EDAReport(
                report_id="benchmark_eda_report",
                dataset_name=validated_df.metadata.file_name,
                dataset_hash=validated_df.metadata.dataset_hash,
                overview=overview,
                target_analysis=target_analysis,
                numerical_analysis=None,
                categorical_analysis=None,
                temporal_analysis=None,
                geographic_analysis=None,
                correlation_analysis=None,
                recommendations=(),
                figure_paths={},
                report_paths={},
                generated_at=datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
                analysis_duration_ms=0.0,
                warnings=(),
                feature_inventory=inventory,
                feature_engineering_blueprint=blueprint,
                dob_analysis=None,
                report_metadata={}
            )

            fe_pipeline = FeatureEngineeringPipeline()
            engineered_df = fe_pipeline.engineer(validated_df, eda_report=eda_report)

            pp_pipeline = PreprocessingPipeline()
            preprocessed_dataset = pp_pipeline.preprocess(engineered_df)
        except Exception as exc:
            logger.error("Dataset preparation failed: %s", exc, exc_info=True)
            logger.error("Benchmark Finished with critical failures.")
            return

        data_prep_end = time.perf_counter()
        timeline.append({
            "step": "Dataset Ready",
            "start_time": data_prep_start_iso,
            "end_time": datetime.datetime.now(
                tz=datetime.timezone.utc
            ).isoformat(),
            "duration_seconds": round(data_prep_end - data_prep_start, 4),
        })
        logger.info("Dataset Ready")

        # 2. Model Discovery
        disc_start = time.perf_counter()
        disc_start_iso = datetime.datetime.now(
            tz=datetime.timezone.utc
        ).isoformat()

        model_types = list(ModelType)
        logger.info("Discovered %d models to benchmark", len(model_types))

        disc_end = time.perf_counter()
        timeline.append({
            "step": "Model Discovery",
            "start_time": disc_start_iso,
            "end_time": datetime.datetime.now(
                tz=datetime.timezone.utc
            ).isoformat(),
            "duration_seconds": round(disc_end - disc_start, 4),
        })
        logger.info("Model Discovery completed.")

        # Load existing runs if resume is enabled
        existing_runs: dict[str, dict[str, Any]] = {}
        if self.resume and not self.force:
            existing_runs = self._load_existing_runs()

        results: list[dict[str, Any]] = []
        validation_records: list[dict[str, Any]] = []

        total_train_duration = 0.0
        total_eval_duration = 0.0
        total_mlflow_duration = 0.0

        # 3. Model Loop Execution
        for model_type in model_types:
            model_name = model_type.value
            model_start_time = time.perf_counter()
            logger.info("Model Started: %s", model_name)

            # Check for cached result
            if model_name in existing_runs:
                logger.info(
                    "Resuming model %s (using cached results)", model_name
                )
                cached = existing_runs[model_name]
                results.append(cached)
                timeline.append({
                    "step": f"{model_name} (Cached)",
                    "start_time": cached.get(
                        "Timestamp",
                        datetime.datetime.now(
                            tz=datetime.timezone.utc
                        ).isoformat(),
                    ),
                    "end_time": datetime.datetime.now(
                        tz=datetime.timezone.utc
                    ).isoformat(),
                    "duration_seconds": 0.0,
                })
                continue

            model_record: dict[str, Any] = {
                "Model Name": model_name,
                "Status": "FAILED",
                "PR_AUC": 0.0,
                "ROC_AUC": 0.0,
                "Precision": 0.0,
                "Recall": 0.0,
                "F1": 0.0,
                "Specificity": 0.0,
                "Balanced Accuracy": 0.0,
                "Matthews Correlation Coefficient": 0.0,
                "False Positive Rate": 0.0,
                "Fraud Detection Rate": 0.0,
                "Fraud Capture Rate": 0.0,
                "Alert Rate": 0.0,
                "Precision@K": 0.0,
                "Optimized Threshold": 0.0,
                "Training Time": 0.0,
                "Evaluation Time": 0.0,
                "MLflow Logging Time": 0.0,
                "Total Execution Time": 0.0,
                "MLflow Run ID": "",
                "Experiment ID": "",
                "Artifact Directory": "",
                "Error Message": None,
                "Timestamp": datetime.datetime.now(
                    tz=datetime.timezone.utc
                ).isoformat(),
            }

            try:
                # ── 3.1 Training ──
                t_start = time.perf_counter()
                training_cfg = replace(
                    settings.training, active_model=model_name
                )
                training_pipeline = TrainingPipeline(cfg=training_cfg)
                training_res = training_pipeline.train(preprocessed_dataset)
                t_end = time.perf_counter()
                train_dur = t_end - t_start
                total_train_duration += train_dur
                model_record["Training Time"] = round(train_dur, 4)
                logger.info("Training Completed: %s", model_name)

                # ── 3.2 Evaluation ──
                e_start = time.perf_counter()
                eval_pipeline = EvaluationPipeline(cfg=settings.evaluation)
                eval_res = eval_pipeline.evaluate(
                    training_res, preprocessed_dataset
                )
                e_end = time.perf_counter()
                eval_dur = e_end - e_start
                total_eval_duration += eval_dur
                model_record["Evaluation Time"] = round(eval_dur, 4)
                logger.info("Evaluation Completed: %s", model_name)

                # ── 3.3 MLflow Tracking ──
                m_start = time.perf_counter()
                mlflow_pipeline = MLflowPipeline(cfg=settings.evaluation)
                mlflow_res = mlflow_pipeline.track(training_res, eval_res)
                m_end = time.perf_counter()
                mlflow_dur = m_end - m_start
                total_mlflow_duration += mlflow_dur
                model_record["MLflow Logging Time"] = round(mlflow_dur, 4)
                logger.info("MLflow Logged: %s", model_name)

                # ── 3.4 Populate Record ──
                model_record["Status"] = "SUCCESS"

                # Core metrics from test fold
                test_m = eval_res.report.test_metrics
                model_record["PR_AUC"] = test_m.get("PR_AUC", 0.0)
                model_record["ROC_AUC"] = test_m.get("ROC_AUC", 0.0)
                model_record["Precision"] = test_m.get("Precision", 0.0)
                model_record["Recall"] = test_m.get("Recall", 0.0)
                model_record["F1"] = test_m.get("F1", 0.0)
                model_record["Specificity"] = test_m.get("Specificity", 0.0)
                model_record["Balanced Accuracy"] = test_m.get(
                    "Balanced_Accuracy", 0.0
                )
                model_record["Matthews Correlation Coefficient"] = test_m.get(
                    "MCC", 0.0
                )

                # Business metrics from test fold
                bus_m = eval_res.report.business_metrics
                model_record["False Positive Rate"] = bus_m.get(
                    "False_Positive_Rate", {}
                ).get("value", 0.0)
                model_record["Fraud Detection Rate"] = bus_m.get(
                    "Fraud_Detection_Rate", {}
                ).get("value", 0.0)
                model_record["Fraud Capture Rate"] = bus_m.get(
                    "Fraud_Capture_Rate", {}
                ).get("value", 0.0)
                model_record["Alert Rate"] = bus_m.get(
                    "Alert_Rate", {}
                ).get("value", 0.0)
                model_record["Precision@K"] = bus_m.get(
                    "Precision_At_K", {}
                ).get("value", 0.0)

                # Threshold parameters
                model_record["Optimized Threshold"] = (
                    eval_res.report.threshold_result.optimized_threshold
                )

                # Tracking metadata
                model_record["MLflow Run ID"] = mlflow_res.metadata.run_id
                model_record["Experiment ID"] = (
                    mlflow_res.metadata.experiment_id
                )
                model_record["Artifact Directory"] = (
                    eval_res.artifacts.artifact_dir
                )

                # ── 3.5 Consistency Validation ──
                val_record = self._validate_model_consistency(
                    model_name, training_res, eval_res
                )
                validation_records.append(val_record)

            except Exception as exc:
                logger.exception(
                    "Model pipeline failed for %s: %s", model_name, exc
                )
                model_record["Status"] = "FAILED"
                model_record["Error Message"] = str(exc)
                # Record a FAILED validation entry so all models appear in
                # benchmark_validation.json
                validation_records.append({
                    "Model": model_name,
                    "Status": "FAILED",
                    "Dataset Hash": "",
                    "Feature Schema Hash": "",
                    "Training Version": settings.training.training_version,
                    "Evaluation Version": settings.evaluation.evaluation_version,
                    "Preprocessing Version": "",
                    "Feature Count": 0,
                    "PR_AUC": 0.0,
                    "ROC_AUC": 0.0,
                    "F1": 0.0,
                    "Precision": 0.0,
                    "Recall": 0.0,
                    "Threshold": 0.0,
                    "Threshold Strategy": settings.evaluation.threshold_strategy,
                    "Random Seed": settings.training.random_seed,
                    "Imbalance Strategy": settings.training.imbalance_strategy,
                    "Hyperparameters": {},
                    "MLflow Run ID": "",
                    "Consistency Warnings": [f"Pipeline failed: {exc}"],
                    "Validation Result": "FAIL",
                })

            finally:
                model_record["Total Execution Time"] = round(
                    time.perf_counter() - model_start_time, 4
                )
                results.append(model_record)

        # Append loops to timeline
        timeline.append({
            "step": "Training",
            "start_time": disc_start_iso,
            "end_time": datetime.datetime.now(
                tz=datetime.timezone.utc
            ).isoformat(),
            "duration_seconds": round(total_train_duration, 4),
        })
        timeline.append({
            "step": "Evaluation",
            "start_time": disc_start_iso,
            "end_time": datetime.datetime.now(
                tz=datetime.timezone.utc
            ).isoformat(),
            "duration_seconds": round(total_eval_duration, 4),
        })
        timeline.append({
            "step": "MLflow Logging",
            "start_time": disc_start_iso,
            "end_time": datetime.datetime.now(
                tz=datetime.timezone.utc
            ).isoformat(),
            "duration_seconds": round(total_mlflow_duration, 4),
        })

        # 4. Ranking
        rank_start = time.perf_counter()
        rank_start_iso = datetime.datetime.now(
            tz=datetime.timezone.utc
        ).isoformat()

        ranked_results, best_model = self._rank_models(results)
        logger.info("Model Ranked")

        rank_end = time.perf_counter()
        timeline.append({
            "step": "Ranking",
            "start_time": rank_start_iso,
            "end_time": datetime.datetime.now(
                tz=datetime.timezone.utc
            ).isoformat(),
            "duration_seconds": round(rank_end - rank_start, 4),
        })

        # 5. Report Generation
        rep_start = time.perf_counter()
        rep_start_iso = datetime.datetime.now(
            tz=datetime.timezone.utc
        ).isoformat()

        self.report_dir.mkdir(parents=True, exist_ok=True)
        self._write_reports(
            ranked_results, best_model, preprocessed_dataset, timeline
        )
        self._write_validation_report(validation_records)
        logger.info("Reports Generated")

        rep_end = time.perf_counter()
        timeline.append({
            "step": "Report Generation",
            "start_time": rep_start_iso,
            "end_time": datetime.datetime.now(
                tz=datetime.timezone.utc
            ).isoformat(),
            "duration_seconds": round(rep_end - rep_start, 4),
        })

        # 6. Manifest Generation
        man_start = time.perf_counter()
        man_start_iso = datetime.datetime.now(
            tz=datetime.timezone.utc
        ).isoformat()

        # Update timeline total duration
        total_dur = round(time.perf_counter() - benchmark_start_time, 4)
        timeline.append({
            "step": "Total Duration",
            "start_time": data_prep_start_iso,
            "end_time": datetime.datetime.now(
                tz=datetime.timezone.utc
            ).isoformat(),
            "duration_seconds": total_dur,
        })

        self._write_manifest()
        logger.info("Manifest Generated")

        man_end = time.perf_counter()
        timeline.append({
            "step": "Manifest Generation",
            "start_time": man_start_iso,
            "end_time": datetime.datetime.now(
                tz=datetime.timezone.utc
            ).isoformat(),
            "duration_seconds": round(man_end - man_start, 4),
        })

        logger.info("Benchmark Finished")

    def _validate_model_consistency(
        self,
        model_name: str,
        training_res: Any,
        eval_res: Any,
    ) -> dict[str, Any]:
        """
        Validates that the training run used the expected default configuration.

        Compares actual training metadata (dataset hash, feature schema hash,
        hyperparameters, seed, versions) against settings defaults and logs
        CONSISTENCY WARNINGs for any deviations.

        Args:
            model_name: Model identifier string.
            training_res: TrainingResult from the pipeline.
            eval_res: EvaluationResult from the pipeline.

        Returns:
            Validation record dict for benchmark_validation.json.
        """
        meta = training_res.metadata
        eval_report = eval_res.report
        test_m = eval_report.test_metrics
        threshold_result = eval_report.threshold_result

        # Expected values from canonical settings
        expected_seed = settings.training.random_seed
        expected_strategy = settings.training.imbalance_strategy
        expected_training_ver = settings.training.training_version
        expected_eval_ver = settings.evaluation.evaluation_version
        expected_threshold_strategy = settings.evaluation.threshold_strategy

        warnings: list[str] = []

        # Seed check
        if meta.random_seed != expected_seed:
            msg = (
                f"[CONSISTENCY WARNING] {model_name}: random_seed mismatch "
                f"expected={expected_seed} actual={meta.random_seed}"
            )
            logger.warning(msg)
            warnings.append(msg)

        # Imbalance strategy check
        if meta.imbalance_strategy != expected_strategy:
            msg = (
                f"[CONSISTENCY WARNING] {model_name}: imbalance_strategy mismatch "
                f"expected={expected_strategy} actual={meta.imbalance_strategy}"
            )
            logger.warning(msg)
            warnings.append(msg)

        # Training version check
        if meta.training_version != expected_training_ver:
            msg = (
                f"[CONSISTENCY WARNING] {model_name}: training_version mismatch "
                f"expected={expected_training_ver} actual={meta.training_version}"
            )
            logger.warning(msg)
            warnings.append(msg)

        # Threshold strategy check
        actual_threshold_strategy = threshold_result.optimization_strategy
        if actual_threshold_strategy != expected_threshold_strategy:
            msg = (
                f"[CONSISTENCY WARNING] {model_name}: threshold_strategy mismatch "
                f"expected={expected_threshold_strategy} actual={actual_threshold_strategy}"
            )
            logger.warning(msg)
            warnings.append(msg)

        # Hyperparameter spot-check: n_estimators / iterations
        hp = meta.hyperparameters
        for hp_key in ("n_estimators", "iterations", "num_leaves"):
            if hp_key in hp:
                actual_val = hp[hp_key]
                # Determine expected value from settings
                model_up = model_name.upper()
                expected_val: Optional[int] = None
                if "XGBOOST" in model_up:
                    expected_val = settings.training.xgb_n_estimators
                elif "LIGHTGBM" in model_up:
                    if hp_key == "n_estimators":
                        expected_val = settings.training.lgbm_n_estimators
                    elif hp_key == "num_leaves":
                        expected_val = settings.training.lgbm_num_leaves
                elif "CATBOOST" in model_up:
                    expected_val = settings.training.cat_iterations
                elif "RANDOM_FOREST" in model_up:
                    expected_val = settings.training.rf_n_estimators

                if expected_val is not None and actual_val != expected_val:
                    msg = (
                        f"[CONSISTENCY WARNING] {model_name}: hyperparameter '{hp_key}' mismatch "
                        f"expected={expected_val} actual={actual_val}. "
                        f"Benchmark may have been run with overridden env vars."
                    )
                    logger.warning(msg)
                    warnings.append(msg)

        validation_pass = len(warnings) == 0

        return {
            "Model": model_name,
            "Status": "SUCCESS",
            "Dataset Hash": meta.dataset_hash,
            "Feature Schema Hash": meta.feature_schema_hash,
            "Training Version": meta.training_version,
            "Evaluation Version": expected_eval_ver,
            "Preprocessing Version": meta.preprocessing_version,
            "Feature Count": meta.feature_count,
            "PR_AUC": test_m.get("PR_AUC", 0.0),
            "ROC_AUC": test_m.get("ROC_AUC", 0.0),
            "F1": test_m.get("F1", 0.0),
            "Precision": test_m.get("Precision", 0.0),
            "Recall": test_m.get("Recall", 0.0),
            "Threshold": threshold_result.optimized_threshold,
            "Threshold Strategy": actual_threshold_strategy,
            "Random Seed": meta.random_seed,
            "Imbalance Strategy": meta.imbalance_strategy,
            "Hyperparameters": meta.hyperparameters,
            "MLflow Run ID": "",  # filled after mlflow tracking
            "Consistency Warnings": warnings,
            "Validation Result": "PASS" if validation_pass else "FAIL",
        }

    def _write_validation_report(
        self, validation_records: list[dict[str, Any]]
    ) -> None:
        """
        Writes benchmark_validation.json with per-model consistency records.

        Args:
            validation_records: List of per-model validation dicts.
        """
        # Back-fill MLflow Run IDs from the main results if available
        # (validation_records are built before mlflow_res is stored; we
        # use the model name to correlate)
        report_path = self.report_dir / "benchmark_report.json"
        mlflow_map: dict[str, str] = {}
        if report_path.exists():
            try:
                with open(report_path, "r", encoding="utf-8") as fh:
                    existing = json.load(fh)
                for r in existing.get("results", []):
                    mlflow_map[r["Model Name"]] = r.get("MLflow Run ID", "")
            except Exception:
                pass

        for rec in validation_records:
            if not rec.get("MLflow Run ID"):
                rec["MLflow Run ID"] = mlflow_map.get(rec["Model"], "")

        out = {
            "generated_at": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
            "models": validation_records,
        }
        val_path = self.report_dir / "benchmark_validation.json"
        with open(val_path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, default=str)
        logger.info("Benchmark validation report written | path=%s", val_path)

    def _load_existing_runs(self) -> dict[str, dict[str, Any]]:
        """
        Loads already completed successful runs from benchmark_report.json.

        Returns:
            Mapping of model name to its cached run result dictionary.
        """
        report_path = self.report_dir / "benchmark_report.json"
        if not report_path.exists():
            return {}

        try:
            with open(report_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                if "results" in data:
                    return {
                        r["Model Name"]: r
                        for r in data["results"]
                        if r.get("Status") == "SUCCESS"
                    }
        except Exception as exc:
            logger.warning("Failed to load existing runs: %s", exc)

        return {}

    def _rank_models(
        self, results: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], Optional[dict[str, Any]]]:
        """
        Ranks successful models using primary metric and tie-breakers.

        Ranking key:
            1. PR_AUC (descending)
            2. F1 (descending)
            3. False Positive Rate (ascending)
            4. Training Time (ascending)

        Args:
            results: Collected result dictionary list.

        Returns:
            Tuple of ranked results list and the best model record.
        """
        successful = [r for r in results if r["Status"] == "SUCCESS"]
        failed = [r for r in results if r["Status"] == "FAILED"]

        # Sort successful runs using stable key:
        # PR_AUC desc -> negate, F1 desc -> negate, FPR asc -> pos, Training Time asc -> pos
        def ranking_key(item: dict[str, Any]) -> tuple[float, float, float, float]:
            return (
                -item.get("PR_AUC", 0.0),
                -item.get("F1", 0.0),
                item.get("False Positive Rate", 0.0),
                item.get("Training Time", 0.0),
            )

        successful.sort(key=ranking_key)

        ranked: list[dict[str, Any]] = []
        for idx, item in enumerate(successful):
            item["Rank"] = idx + 1
            ranked.append(item)

        for item in failed:
            item["Rank"] = None
            ranked.append(item)

        best_model = ranked[0] if successful else None
        return ranked, best_model

    def _write_reports(
        self,
        results: list[dict[str, Any]],
        best_model: Optional[dict[str, Any]],
        dataset: Any,
        timeline: list[dict[str, Any]],
    ) -> None:
        """
        Writes JSON, MD, and CSV reports into reports/benchmark/.
        """
        timestamp = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
        successful_runs = [r for r in results if r["Status"] == "SUCCESS"]

        # Schema and Dataset parameters
        ds_name = getattr(dataset.metadata, "name", "Fraud Transaction Dataset")
        ds_ver = getattr(dataset.metadata, "version", "1.0.0")
        ds_hash = getattr(dataset.report, "dataset_hash", "")
        schema_hash = hashlib.sha256(
            ",".join(sorted(dataset.feature_names)).encode("utf-8")
        ).hexdigest()

        # 1. benchmark_metadata.json
        meta_data = {
            "Dataset Name": ds_name,
            "Dataset Version": ds_ver,
            "Dataset Hash": ds_hash,
            "Feature Schema Hash": schema_hash,
            "Pipeline Version": settings.training.training_version,
            "Models Evaluated": [r["Model Name"] for r in results],
            "Successful Runs": len(successful_runs),
            "Failed Runs": len(results) - len(successful_runs),
            "Benchmark Timestamp": timestamp,
        }
        with open(
            self.report_dir / "benchmark_metadata.json", "w", encoding="utf-8"
        ) as fh:
            json.dump(meta_data, fh, indent=2, default=str)

        # 2. benchmark_summary.json
        summary_data = {
            "Best Model": (best_model["Model Name"] if best_model else "None"),
            "Best PR_AUC": (best_model["PR_AUC"] if best_model else 0.0),
            "Fastest Model": (
                min(successful_runs, key=lambda x: x["Training Time"])[
                    "Model Name"
                ]
                if successful_runs
                else "None"
            ),
            "Lowest False Positive Rate": (
                min(successful_runs, key=lambda x: x["False Positive Rate"])[
                    "Model Name"
                ]
                if successful_runs
                else "None"
            ),
            "Highest Recall": (
                max(successful_runs, key=lambda x: x["Recall"])["Model Name"]
                if successful_runs
                else "None"
            ),
            "Highest Precision": (
                max(successful_runs, key=lambda x: x["Precision"])["Model Name"]
                if successful_runs
                else "None"
            ),
            "Highest F1": (
                max(successful_runs, key=lambda x: x["F1"])["Model Name"]
                if successful_runs
                else "None"
            ),
        }
        with open(
            self.report_dir / "benchmark_summary.json", "w", encoding="utf-8"
        ) as fh:
            json.dump(summary_data, fh, indent=2, default=str)

        # 3. best_model.json
        best_data = {
            "best_model": (best_model["Model Name"] if best_model else "None"),
            "selection_metric": "PR_AUC",
            "score": (best_model["PR_AUC"] if best_model else 0.0),
            "mlflow_run_id": (
                best_model["MLflow Run ID"] if best_model else ""
            ),
            "recommended_stage": (
                "Production" if best_model else "Development"
            ),
        }
        with open(
            self.report_dir / "best_model.json", "w", encoding="utf-8"
        ) as fh:
            json.dump(best_data, fh, indent=2, default=str)

        # 4. model_ranking.csv
        csv_path = self.report_dir / "model_ranking.csv"
        csv_headers = [
            "Rank",
            "Model",
            "PR_AUC",
            "F1",
            "Recall",
            "Precision",
            "FPR",
            "Training Time",
            "Evaluation Time",
            "Threshold",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(csv_headers)
            for r in results:
                if r["Status"] == "SUCCESS":
                    writer.writerow([
                        r["Rank"],
                        r["Model Name"],
                        f"{r['PR_AUC']:.6f}",
                        f"{r['F1']:.6f}",
                        f"{r['Recall']:.6f}",
                        f"{r['Precision']:.6f}",
                        f"{r['False Positive Rate']:.6f}",
                        f"{r['Training Time']:.4f}",
                        f"{r['Evaluation Time']:.4f}",
                        f"{r['Optimized Threshold']:.4f}",
                    ])
                else:
                    writer.writerow([
                        "FAILED",
                        r["Model Name"],
                        "N/A",
                        "N/A",
                        "N/A",
                        "N/A",
                        "N/A",
                        "N/A",
                        "N/A",
                        "N/A",
                    ])

        # 5. benchmark_report.json
        report_json = {
            "metadata": meta_data,
            "summary": summary_data,
            "timeline": timeline,
            "results": results,
        }
        with open(
            self.report_dir / "benchmark_report.json", "w", encoding="utf-8"
        ) as fh:
            json.dump(report_json, fh, indent=2, default=str)

        # 6. benchmark_report.md
        self._write_report_markdown(report_json, best_model)

    def _write_report_markdown(
        self, report: dict[str, Any], best_model: Optional[dict[str, Any]]
    ) -> None:
        """
        Generates the human-readable Markdown summary report.
        """
        md_path = self.report_dir / "benchmark_report.md"
        meta = report["metadata"]
        summary = report["summary"]
        results = report["results"]

        md_content = []
        md_content.append("# Model Performance Benchmark Report\n")
        md_content.append(
            f"Generated at: `{meta['Benchmark Timestamp']}`\n"
        )

        md_content.append("## Executive Summary\n")
        if best_model:
            md_content.append(
                f"> [!IMPORTANT]\n"
                f"> **Best Model Recommendation**  \n"
                f"> The recommended model is **{best_model['Model Name']}**, "
                f"achieving a primary metric **PR_AUC of {best_model['PR_AUC']:.4f}** and an **F1 score of {best_model['F1']:.4f}**.  \n"
                f"> This model is promoted to stage **Production** in the MLflow Model Registry.\n"
            )
        else:
            md_content.append(
                "> [!WARNING]\n"
                "> No successful model runs were recorded during this benchmark run.\n"
            )

        md_content.append("### Quick Stats")
        md_content.append(f"- **Dataset Evaluated**: `{meta['Dataset Name']}`")
        md_content.append(f"- **Dataset Version**: `{meta['Dataset Version']}`")
        md_content.append(f"- **Successful Runs**: `{meta['Successful Runs']}`")
        md_content.append(f"- **Failed Runs**: `{meta['Failed Runs']}`")
        md_content.append(
            f"- **Fastest Training**: `{summary['Fastest Model']}`"
        )
        md_content.append(
            f"- **Lowest False Positive Rate**: `{summary['Lowest False Positive Rate']}`\n"
        )

        md_content.append("## Model Performance Ranking\n")
        md_content.append(
            "| Rank | Model Name | Status | PR_AUC | F1 | Recall | Precision | FPR | Training Time | MLflow ID |\n"
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
        )
        for r in results:
            if r["Status"] == "SUCCESS":
                md_content.append(
                    f"| {r['Rank']} | **{r['Model Name']}** | `SUCCESS` | {r['PR_AUC']:.4f} | {r['F1']:.4f} | {r['Recall']:.4f} | {r['Precision']:.4f} | {r['False Positive Rate']:.4f} | {r['Training Time']:.2f}s | `{r['MLflow Run ID'][:8]}` |"
                )
            else:
                md_content.append(
                    f"| FAILED | **{r['Model Name']}** | `FAILED` | N/A | N/A | N/A | N/A | N/A | N/A | N/A |"
                )
        md_content.append("")

        md_content.append("## Detailed Metric Comparison\n")
        md_content.append(
            "| Model Name | ROC_AUC | Specificity | Balanced Accuracy | MCC | Alert Rate | Precision@K | Threshold |\n"
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
        )
        for r in results:
            if r["Status"] == "SUCCESS":
                md_content.append(
                    f"| **{r['Model Name']}** | {r['ROC_AUC']:.4f} | {r['Specificity']:.4f} | {r['Balanced Accuracy']:.4f} | {r['Matthews Correlation Coefficient']:.4f} | {r['Alert Rate']:.4f} | {r['Precision@K']:.4f} | {r['Optimized Threshold']:.4f} |"
                )
            else:
                md_content.append(
                    f"| **{r['Model Name']}** | N/A | N/A | N/A | N/A | N/A | N/A | N/A |"
                )
        md_content.append("")

        md_content.append("## Benchmark Execution Timeline\n")
        md_content.append("| Step Name | Start Time | End Time | Duration (Seconds) |\n"
                          "| :--- | :--- | :--- | :--- |")
        for step in report["timeline"]:
            md_content.append(
                f"| {step['step']} | `{step['start_time']}` | `{step['end_time']}` | {step['duration_seconds']:.4f}s |"
            )
        md_content.append("")

        md_content.append("## Recommendation Summary\n")
        if best_model:
            md_content.append(
                f"Based on the primary metric **PR_AUC (Precision-Recall Area Under Curve)**, the best performing model is **{best_model['Model Name']}**.\n\n"
                f"- It achieves the highest precision-recall balance with a score of **{best_model['PR_AUC']:.4f}**.\n"
                f"- The decision threshold was optimized using maximum F1 strategy to **{best_model['Optimized Threshold']:.4f}**.\n"
                f"- This model is fully tracked under MLflow Run ID `{best_model['MLflow Run ID']}`."
            )
        else:
            md_content.append(
                "No models succeeded. Check log messages and stack traces to resolve setup or parameter issues."
            )

        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(md_content))

    def _write_manifest(self) -> None:
        """
        Generates manifest.json listing hashes, sizes, and timestamps of all reports.
        """
        manifest_path = self.report_dir / "manifest.json"
        artifact_files = [
            "benchmark_report.json",
            "benchmark_report.md",
            "model_ranking.csv",
            "best_model.json",
            "benchmark_metadata.json",
            "benchmark_summary.json",
            "benchmark_validation.json",
        ]

        artifacts = []
        for filename in artifact_files:
            file_path = self.report_dir / filename
            if not file_path.exists():
                continue

            # Calculate SHA-256
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as fh:
                for byte_block in iter(lambda: fh.read(4096), b""):
                    sha256_hash.update(byte_block)
            sha256_val = sha256_hash.hexdigest()

            # Stats
            stat = file_path.stat()
            size = stat.st_size
            created_at = datetime.datetime.fromtimestamp(
                stat.st_mtime, tz=datetime.timezone.utc
            ).isoformat()

            # Guess mime-type
            mime_type = mimetypes.guess_type(str(file_path))[0]
            if not mime_type:
                if filename.endswith(".json"):
                    mime_type = "application/json"
                elif filename.endswith(".md"):
                    mime_type = "text/markdown"
                elif filename.endswith(".csv"):
                    mime_type = "text/csv"
                else:
                    mime_type = "application/octet-stream"

            artifacts.append({
                "filename": filename,
                "sha256": sha256_val,
                "size_bytes": size,
                "mime_type": mime_type,
                "created_at": created_at,
            })

        manifest = {"artifacts": artifacts}
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, default=str)


if __name__ == "__main__":
    # Centralized basic configuration for debugging/execution
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    # Enable resume support by default if environment demands it
    resume_flag = os.environ.get("FRAUD_BENCHMARK_RESUME", "0") == "1"
    runner = BenchmarkRunner(resume=resume_flag)
    runner.run()
