"""
FeatureEngineeringPipeline — orchestrator for the Feature Engineering Layer.

Public API:
    pipeline = FeatureEngineeringPipeline()
    engineered = pipeline.engineer(validated_dataset, eda_report)

The pipeline:
    1. Guards against invalid inputs.
    2. Executes all enabled modules in registry order.
    3. Gracefully degrades on individual module failures (warning, not crash).
    4. Drops PII and identifier columns.
    5. Assembles and returns EngineeredDataset.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone

import pandas as pd

from config.settings import settings
from src.eda.models import EDAReport
from src.features import amount, behavioral, categorical, demographic, frequency, geographic, temporal
from src.features.exceptions import FeatureCreationError, PipelineError, TransformationError
from src.features.models import (
    EngineeredDataset,
    EncodingRecommendation,
    FeatureDefinition,
    FeatureEngineeringReport,
)
from src.features.registry import FEATURE_REGISTRY
from src.validation.models import ValidatedDataset

logger = logging.getLogger(__name__)


class FeatureEngineeringPipeline:
    """
    Orchestrates the full Feature Engineering Layer.

    This class is stateless across calls — each call to ``engineer()``
    is independent. Configuration is read once at construction from
    ``config.settings.fe``.

    Example:
        >>> pipeline = FeatureEngineeringPipeline()
        >>> engineered = pipeline.engineer(validated_dataset, eda_report)
        >>> print(engineered.report)

    Args:
        cfg: Optional FESettings override. Defaults to ``settings.fe``.
    """

    def __init__(self, cfg=None) -> None:
        self._cfg = cfg or settings.fe

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def engineer(
        self,
        validated_dataset: ValidatedDataset,
        eda_report: EDAReport,
    ) -> EngineeredDataset:
        """
        Runs the full Feature Engineering pipeline.

        Args:
            validated_dataset: Output of the Validation Layer.
                Must have ``passed == True``.
            eda_report: Output of the EDA Layer. Must contain
                ``feature_inventory`` and ``feature_engineering_blueprint``.

        Returns:
            EngineeredDataset: Enriched DataFrame with full lineage report.

        Raises:
            PipelineError: If ``validated_dataset.passed`` is False, or
                ``eda_report.feature_inventory`` is missing.
        """
        _global_start = time.perf_counter()

        # ── 1. Guard ─────────────────────────────────────────────────────
        if not validated_dataset.passed:
            raise PipelineError(
                stage="GUARD",
                detail=(
                    "ValidatedDataset.passed is False. Feature Engineering "
                    "requires a validated dataset with no structural errors."
                ),
                context={
                    "dataset": validated_dataset.metadata.file_name,
                    "errors": validated_dataset.report.summary.errors,
                },
            )

        if eda_report is None or not hasattr(eda_report, "feature_inventory"):
            raise PipelineError(
                stage="GUARD",
                detail="EDAReport is missing or does not contain feature_inventory.",
            )

        if not FEATURE_REGISTRY:
            raise PipelineError(
                stage="GUARD",
                detail="Feature registry is empty. No modules to execute.",
            )

        inventory = eda_report.feature_inventory
        blueprint = eda_report.feature_engineering_blueprint
        dataset_name = validated_dataset.metadata.file_name
        dataset_hash = eda_report.dataset_hash

        logger.info(
            "Feature Engineering started | dataset='%s' | shape=(%d rows x %d cols)",
            dataset_name,
            validated_dataset.metadata.num_rows,
            validated_dataset.metadata.num_columns,
        )

        # ── 2. Copy ───────────────────────────────────────────────────────
        df = validated_dataset.data.copy()
        original_columns = tuple(df.columns.tolist())
        feature_count_before = len(original_columns)

        warnings_log: list[str] = []
        leakage_warnings: list[str] = []
        all_new_columns: list[str] = []
        all_feature_definitions: list[FeatureDefinition] = []
        transformation_summary: dict[str, int] = {}
        frequency_maps: dict[str, dict[str, int]] = {}
        encoding_recommendations: dict[str, EncodingRecommendation] = {}

        # ── 3. Execute modules ────────────────────────────────────────────
        for entry in FEATURE_REGISTRY:
            group = entry.group_name

            # Determine if module is enabled
            if group == "BEHAVIORAL":
                is_enabled = self._cfg.enable_behavioral
            else:
                is_enabled = entry.enabled_by_default

            if not is_enabled and group != "BEHAVIORAL":
                logger.info("Module '%s' is disabled — skipping.", group)
                continue

            # ── TEMPORAL ─────────────────────────────────────────────────
            if group == "TEMPORAL":
                new_df = self._run_module(
                    group=group,
                    fn=lambda: temporal.run_temporal(df),
                    warnings_log=warnings_log,
                )
                if new_df is not None:
                    self._validate_and_concat(df, new_df, group, all_new_columns, warnings_log)
                    df = pd.concat([df, new_df], axis=1)
                    transformation_summary[group] = len(new_df.columns)
                    all_feature_definitions.extend(entry.feature_definitions)

            # ── AMOUNT ───────────────────────────────────────────────────
            elif group == "AMOUNT":
                new_df = self._run_module(
                    group=group,
                    fn=lambda: amount.run_amount(df, self._cfg.high_value_threshold),
                    warnings_log=warnings_log,
                )
                if new_df is not None:
                    self._validate_and_concat(df, new_df, group, all_new_columns, warnings_log)
                    df = pd.concat([df, new_df], axis=1)
                    transformation_summary[group] = len(new_df.columns)
                    all_feature_definitions.extend(entry.feature_definitions)

            # ── DEMOGRAPHIC ───────────────────────────────────────────────
            elif group == "DEMOGRAPHIC":
                new_df = self._run_module(
                    group=group,
                    fn=lambda: demographic.run_demographic(
                        df,
                        age_bins=self._cfg.age_bins,
                        age_bin_labels=self._cfg.age_bin_labels,
                    ),
                    warnings_log=warnings_log,
                )
                if new_df is not None:
                    self._validate_and_concat(df, new_df, group, all_new_columns, warnings_log)
                    df = pd.concat([df, new_df], axis=1)
                    transformation_summary[group] = len(new_df.columns)
                    all_feature_definitions.extend(entry.feature_definitions)

            # ── GEOGRAPHIC ────────────────────────────────────────────────
            elif group == "GEOGRAPHIC":
                new_df = self._run_module(
                    group=group,
                    fn=lambda: geographic.run_geographic(
                        df,
                        far_transaction_threshold_km=self._cfg.far_transaction_threshold_km,
                    ),
                    warnings_log=warnings_log,
                )
                if new_df is not None:
                    self._validate_and_concat(df, new_df, group, all_new_columns, warnings_log)
                    df = pd.concat([df, new_df], axis=1)
                    transformation_summary[group] = len(new_df.columns)
                    all_feature_definitions.extend(entry.feature_definitions)

            # ── FREQUENCY ─────────────────────────────────────────────────
            elif group == "FREQUENCY":
                result = self._run_module(
                    group=group,
                    fn=lambda: frequency.run_frequency(df, self._cfg.frequency_columns),
                    warnings_log=warnings_log,
                )
                if result is not None:
                    new_df, freq_maps = result
                    frequency_maps.update(freq_maps)
                    self._validate_and_concat(df, new_df, group, all_new_columns, warnings_log)
                    df = pd.concat([df, new_df], axis=1)
                    transformation_summary[group] = len(new_df.columns)
                    all_feature_definitions.extend(entry.feature_definitions)

            # ── CATEGORICAL ───────────────────────────────────────────────
            elif group == "CATEGORICAL":
                cat_cols = [
                    c for c in df.columns
                    if str(df[c].dtype) in ("object", "str", "string", "category")
                    and c not in inventory.drop_before_training
                    and c not in inventory.temporal
                ]
                result = self._run_module(
                    group=group,
                    fn=lambda: categorical.run_categorical_annotation(
                        df,
                        categorical_columns=tuple(sorted(cat_cols)),
                        max_one_hot_cardinality=self._cfg.max_one_hot_cardinality,
                    ),
                    warnings_log=warnings_log,
                )
                if result is not None:
                    encoding_recommendations.update(result)
                    transformation_summary[group] = 0  # no new columns

            # ── BEHAVIORAL ────────────────────────────────────────────────
            elif group == "BEHAVIORAL":
                result = self._run_module(
                    group=group,
                    fn=lambda: behavioral.run_behavioral(
                        df,
                        enabled=self._cfg.enable_behavioral,
                        historical_reference_df=None,
                    ),
                    warnings_log=warnings_log,
                )
                if result is not None:
                    new_df, beh_warnings = result
                    for w in beh_warnings:
                        if "LEAKAGE" in w:
                            leakage_warnings.append(w)
                        else:
                            warnings_log.append(w)
                    if not new_df.empty:
                        self._validate_and_concat(df, new_df, group, all_new_columns, warnings_log)
                        df = pd.concat([df, new_df], axis=1)
                        transformation_summary[group] = len(new_df.columns)
                        all_feature_definitions.extend(entry.feature_definitions)
                    else:
                        transformation_summary[group] = 0

        # ── 4. Drop columns ───────────────────────────────────────────────
        drop_cols = [c for c in inventory.drop_before_training if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)
            logger.info("Dropped columns: %s", drop_cols)

        dropped_columns = tuple(drop_cols)

        # ── 5. Build report ───────────────────────────────────────────────
        duration_ms = (time.perf_counter() - _global_start) * 1_000

        report = FeatureEngineeringReport(
            report_id=str(uuid.uuid4()),
            dataset_name=dataset_name,
            dataset_hash=dataset_hash,
            feature_count_before=feature_count_before,
            feature_count_after=len(df.columns),
            feature_matrix_shape=(len(df), len(df.columns)),
            generated_features=tuple(all_new_columns),
            dropped_columns=dropped_columns,
            feature_definitions=tuple(all_feature_definitions),
            transformation_summary=transformation_summary,
            encoding_recommendations=encoding_recommendations,
            frequency_maps=frequency_maps,
            leakage_warnings=tuple(leakage_warnings),
            warnings=tuple(warnings_log),
            generated_at=datetime.now(tz=timezone.utc).isoformat(),
            duration_ms=round(duration_ms, 2),
        )

        logger.info(
            "Feature Engineering complete | dataset='%s' | features_before=%d | "
            "features_after=%d | generated=%d | dropped=%d | warnings=%d | duration=%.1fms",
            dataset_name,
            feature_count_before,
            len(df.columns),
            len(all_new_columns),
            len(dropped_columns),
            len(warnings_log),
            duration_ms,
        )

        return EngineeredDataset(
            data=df,
            original_columns=original_columns,
            engineered_columns=tuple(all_new_columns),
            dropped_columns=dropped_columns,
            metadata=validated_dataset.metadata,
            report=report,
        )

    def transform(self, df: pd.DataFrame) -> EngineeredDataset:
        """
        Executes feature engineering on raw transaction input DataFrame during inference.

        Args:
            df: Raw transaction input DataFrame.

        Returns:
            EngineeredDataset: Enriched DataFrame with dropped PII/identifiers.
        """
        from src.validation.models import ValidatedDataset, DatasetMetadata, ValidationReport, ValidationSummary
        from src.eda.models import EDAReport, DatasetOverview, TargetAnalysis
        from src.eda.analyzer import EDAAnalyzer
        from src.ingestion.models import LoadedDataset
        from src.validation import DatasetValidator, FRAUD_TRANSACTION_SCHEMA
        
        # 1. Prepare raw columns, appending placeholders if missing
        df_for_val = df.copy()
        if "Unnamed: 0" not in df_for_val.columns:
            df_for_val["Unnamed: 0"] = 0
        if "is_fraud" not in df_for_val.columns:
            df_for_val["is_fraud"] = 0
            
        metadata = DatasetMetadata(
            file_name="inference_request.csv",
            file_path="in-memory",
            file_extension=".csv",
            file_size_bytes=0,
            num_rows=len(df_for_val),
            num_columns=len(df_for_val.columns),
            column_names=tuple(df_for_val.columns),
            memory_usage_bytes=0,
            target_column="is_fraud",
            load_timestamp=datetime.now(tz=timezone.utc),
            dataset_hash="dummy_hash",
            dataset_version="1.0.0",
            loaded_by="inference",
        )
        
        loaded_dataset = LoadedDataset(data=df_for_val, metadata=metadata)
        validator = DatasetValidator(schema=FRAUD_TRANSACTION_SCHEMA, categories={"STRUCTURAL"})
        validated_dataset = validator.validate(loaded_dataset)
        
        # 2. Build minimal EDAReport using internal helpers
        analyzer = EDAAnalyzer()
        inventory = analyzer._build_feature_inventory(df_for_val)
        blueprint = analyzer._build_feature_engineering_blueprint(df_for_val, inventory)
        
        overview = DatasetOverview(
            num_rows=len(df_for_val),
            num_columns=len(df_for_val.columns),
            memory_usage_mb=0.0,
            num_numeric_columns=0,
            num_categorical_columns=0,
            num_datetime_columns=0,
            column_names=tuple(df_for_val.columns),
            missing_values_summary={},
            dataset_hash="dummy_hash",
            analysis_timestamp=datetime.now(tz=timezone.utc).isoformat()
        )
        target_analysis = TargetAnalysis(
            target_column="is_fraud",
            fraud_count=0,
            legitimate_count=len(df_for_val),
            fraud_pct=0.0,
            legitimate_pct=100.0,
            imbalance_ratio=0.0,
            class_distribution={"0": len(df_for_val), "1": 0}
        )
        eda_report = EDAReport(
            report_id="inference_eda_report",
            dataset_name="inference_request",
            dataset_hash="dummy_hash",
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
            generated_at=datetime.now(tz=timezone.utc).isoformat(),
            analysis_duration_ms=0.0,
            warnings=(),
            feature_inventory=inventory,
            feature_engineering_blueprint=blueprint,
            dob_analysis=None,
            report_metadata={}
        )
        
        # 3. Call engineer()
        try:
            return self.engineer(validated_dataset, eda_report=eda_report)
        except Exception as exc:
            raise TransformationError(f"Feature engineering pipeline transform failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_module(self, group: str, fn, warnings_log: list[str]):
        """
        Executes a module function with graceful error handling.

        Returns the module's return value on success, or None on failure
        (after appending a warning).
        """
        try:
            return fn()
        except (TransformationError, FeatureCreationError) as exc:
            msg = f"Module '{group}' failed: {exc.message} — skipping."
            logger.warning(msg)
            warnings_log.append(msg)
            return None
        except Exception as exc:
            msg = f"Module '{group}' raised unexpected error: {exc} — skipping."
            logger.warning(msg)
            warnings_log.append(msg)
            return None

    def _validate_and_concat(
        self,
        df: pd.DataFrame,
        new_df: pd.DataFrame,
        group: str,
        all_new_columns: list[str],
        warnings_log: list[str],
    ) -> None:
        """
        Validates module output shape and registers new column names.

        Raises:
            FeatureCreationError: If row counts diverge.
        """
        if len(new_df) != len(df):
            raise FeatureCreationError(
                module_name=group,
                expected=f"row_count={len(df)}",
                actual=f"row_count={len(new_df)}",
            )
        for col in new_df.columns:
            if col not in all_new_columns:
                all_new_columns.append(col)
                logger.debug("Feature created | group=%s | name=%s", group, col)
