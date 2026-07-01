"""
PreprocessingPipeline — orchestrator for the Preprocessing Layer.

Public API:
    pipeline = PreprocessingPipeline()
    preprocessed = pipeline.preprocess(engineered_dataset)

Execution follows PREPROCESSING_REGISTRY order exactly:
    DROP → SPLIT → FREQUENCY_ENCODING → ONE_HOT_ENCODING → SCALING → SAVE_ARTIFACTS
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone

import pandas as pd

from config.settings import settings
from src.features.models import EngineeredDataset
from src.preprocessing.artifacts import ArtifactManager
from src.preprocessing.exceptions import (
    ArtifactError,
    PipelineError,
    SplitError,
    TransformationError,
)
from src.preprocessing.models import (
    DatasetSplit,
    PreprocessedDataset,
    PreprocessingReport,
    TransformationMetadata,
    TransformationSummary,
)
from src.preprocessing.registry import (
    PREPROCESSING_REGISTRY,
    STEP_TYPE_ARTIFACT,
    STEP_TYPE_DROP,
    STEP_TYPE_ENCODER,
    STEP_TYPE_SCALER,
    STEP_TYPE_SPLIT,
    COL_SOURCE_FREQUENCY_RECS,
    COL_SOURCE_OHE_RECS,
    COL_SOURCE_SCALE_FEATURES,
)
from src.preprocessing.splitter import TimeBasedSplitter
from src.preprocessing.transformers import (
    BaseTransformer,
    FrequencyEncoder,
    IdentityTransformer,
    OneHotEncoder,
    RobustScalerTransformer,
)

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    Orchestrates the full Preprocessing Layer.

    Stateless across calls — each call to ``preprocess()`` is independent.
    Configuration is read once at construction from ``config.settings.preprocessing``.

    Example:
        >>> pipeline = PreprocessingPipeline()
        >>> preprocessed = pipeline.preprocess(engineered_dataset)
        >>> print(preprocessed.report)

    Args:
        cfg: Optional PreprocessingSettings override. Defaults to
            ``settings.preprocessing``.
    """

    def __init__(self, cfg=None) -> None:
        self._cfg = cfg or settings.preprocessing

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(self, engineered_dataset: EngineeredDataset) -> PreprocessedDataset:
        """
        Runs the full Preprocessing pipeline.

        Args:
            engineered_dataset: Output of the Feature Engineering Layer.

        Returns:
            PreprocessedDataset: Six folds, feature metadata, and full report.

        Raises:
            PipelineError: If inputs are invalid or registry is empty.
            SplitError: If time-based splitting produces an empty fold.
        """
        _global_start = time.perf_counter()
        run_id = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")

        # ── 1. Guard ─────────────────────────────────────────────────────
        self._guard(engineered_dataset)

        dataset_name = engineered_dataset.metadata.file_name
        dataset_hash = engineered_dataset.report.dataset_hash
        encoding_recs = engineered_dataset.report.encoding_recommendations
        blueprint = None  # resolved lazily from eda chain via report
        # Access blueprint through FE report (eda_report not directly available)
        # encoding_recommendations already consumed from FE report — no eda needed.

        logger.info(
            "Preprocessing started | dataset='%s' | shape=%s | run_id=%s | "
            "dataset_hash=%s | version=%s",
            dataset_name,
            engineered_dataset.data.shape,
            run_id,
            dataset_hash or "unknown",
            self._cfg.preprocessing_version,
        )

        # ── 2. Copy ───────────────────────────────────────────────────────
        df = engineered_dataset.data.copy()

        warnings_log: list[str] = []
        transformation_metas: list[TransformationMetadata] = []
        timeline: list[dict] = []
        fitted_transformers: dict[str, BaseTransformer] = {}
        artifact_paths: dict[str, str] = {}
        parquet_paths: dict[str, str] = {}

        train_split: DatasetSplit | None = None
        val_split: DatasetSplit | None = None
        test_split: DatasetSplit | None = None

        X_train = X_val = X_test = pd.DataFrame()
        y_train = y_val = y_test = pd.Series(dtype="int64")

        # ── 3. Iterate registry ───────────────────────────────────────────
        for entry in PREPROCESSING_REGISTRY:
            if not entry.enabled:
                logger.info("Step '%s' disabled — skipping.", entry.step_name)
                continue

            step_start = time.perf_counter()
            step_start_iso = datetime.now(tz=timezone.utc).isoformat()

            # ── DROP ─────────────────────────────────────────────────────
            if entry.step_type == STEP_TYPE_DROP:
                drop_targets = list(self._cfg.source_columns_to_drop)
                # Also drop datetime column if present (will be used for split)
                present_drops = [c for c in drop_targets if c in df.columns]
                # Keep datetime column alive until after split
                datetime_col = self._cfg.datetime_column
                present_drops_now = [
                    c for c in present_drops if c != datetime_col
                ]
                if present_drops_now:
                    df = df.drop(columns=present_drops_now)
                    logger.info("Dropped source columns: %s", present_drops_now)
                step_ms = (time.perf_counter() - step_start) * 1_000
                timeline.append({
                    "step": entry.step_name,
                    "start_time": step_start_iso,
                    "end_time": datetime.now(tz=timezone.utc).isoformat(),
                    "duration_ms": round(step_ms, 2),
                })

            # ── SPLIT ─────────────────────────────────────────────────────
            elif entry.step_type == STEP_TYPE_SPLIT:
                splitter = TimeBasedSplitter(
                    train_end_date=self._cfg.train_end_date,
                    validation_start_date=self._cfg.validation_start_date,
                    validation_end_date=self._cfg.validation_end_date,
                    test_start_date=self._cfg.test_start_date,
                    test_end_date=self._cfg.test_end_date,
                    datetime_column=self._cfg.datetime_column,
                    target_column=self._cfg.target_column,
                )
                # SplitError propagates — empty fold is unrecoverable.
                train_split, val_split, test_split = splitter.split(df)

                # Remove datetime column from all X folds immediately after split.
                X_train = self._drop_datetime_and_target(train_split.X)
                X_val = self._drop_datetime_and_target(val_split.X)
                X_test = self._drop_datetime_and_target(test_split.X)
                y_train = train_split.y.copy()
                y_val = val_split.y.copy()
                y_test = test_split.y.copy()

                step_ms = (time.perf_counter() - step_start) * 1_000
                logger.info(
                    "Split complete | train=%d | val=%d | test=%d | duration=%.1fms",
                    len(X_train),
                    len(X_val),
                    len(X_test),
                    step_ms,
                )
                timeline.append({
                    "step": entry.step_name,
                    "start_time": step_start_iso,
                    "end_time": datetime.now(tz=timezone.utc).isoformat(),
                    "duration_ms": round(step_ms, 2),
                })

            # ── ENCODER / SCALER ──────────────────────────────────────────
            elif entry.step_type in (STEP_TYPE_ENCODER, STEP_TYPE_SCALER):
                columns = self._resolve_columns(
                    entry.columns_source, encoding_recs, X_train
                )
                if not columns:
                    logger.info(
                        "Step '%s': no columns resolved — skipping.", entry.step_name
                    )
                    timeline.append(
                        {"step": entry.step_name, "duration_ms": 0.0, "skipped": True}
                    )
                    continue

                transformer = self._build_transformer(entry.step_name, columns)

                try:
                    in_shape = X_train.shape
                    transformer.fit(X_train, y_train)
                    X_train = transformer.transform(X_train)
                    X_val = transformer.transform(X_val)
                    X_test = transformer.transform(X_test)
                    out_shape = X_train.shape
                except TransformationError:
                    raise
                except Exception as exc:
                    err = TransformationError(
                        transformer_name=transformer.name,
                        step_name=entry.step_name,
                        detail=str(exc),
                        original_exc=exc,
                    )
                    msg = f"Step '{entry.step_name}' failed: {err.message} — skipping."
                    logger.warning(msg)
                    warnings_log.append(msg)
                    step_ms = (time.perf_counter() - step_start) * 1_000
                    timeline.append({
                        "step": entry.step_name,
                        "start_time": step_start_iso,
                        "end_time": datetime.now(tz=timezone.utc).isoformat(),
                        "duration_ms": round(step_ms, 2),
                        "error": msg,
                    })
                    continue

                fitted_transformers[entry.step_name] = transformer
                step_ms = (time.perf_counter() - step_start) * 1_000

                meta = TransformationMetadata(
                    transformer_name=transformer.name,
                    step_name=entry.step_name,
                    transformer_type=transformer.transformer_type,
                    columns_input=tuple(columns),
                    columns_output=tuple(transformer.get_feature_names_out()),
                    input_shape=in_shape,
                    output_shape=out_shape,
                    fit_on_fold="train",
                    parameters=transformer.get_params(),
                    duration_ms=step_ms,
                )
                transformation_metas.append(meta)
                step_ms = (time.perf_counter() - step_start) * 1_000
                timeline.append({
                    "step": entry.step_name,
                    "start_time": step_start_iso,
                    "end_time": datetime.now(tz=timezone.utc).isoformat(),
                    "duration_ms": round(step_ms, 2),
                })

                logger.info(
                    "Transform complete | step=%s | cols_in=%d | cols_out=%d | duration=%.1fms",
                    entry.step_name,
                    len(columns),
                    len(transformer.get_feature_names_out()),
                    step_ms,
                )

            # ── ARTIFACT SAVING ───────────────────────────────────────────
            elif entry.step_type == STEP_TYPE_ARTIFACT:
                artifact_manager = ArtifactManager(
                    artifact_dir=self._cfg.artifact_dir,
                    parquet_dir=self._cfg.output_dir,
                    run_id=run_id,
                    compression=self._cfg.parquet_compression,
                )

                feature_names_input = list(
                    train_split.X.columns
                    if train_split
                    else []
                )
                # Remove datetime col from feature_names_input tracking
                feature_names_input = [
                    c for c in feature_names_input
                    if c not in (self._cfg.datetime_column, self._cfg.target_column)
                ]
                feature_names_output = list(X_train.columns)

                # Save transformers
                transformer_artifact_map = {
                    "FREQUENCY_ENCODING": "frequency_encoder",
                    "ONE_HOT_ENCODING": "ohe_encoder",
                    "SCALING": "scaler",
                }
                for step_key, artifact_key in transformer_artifact_map.items():
                    if step_key in fitted_transformers:
                        try:
                            path = artifact_manager.save_transformer(
                                fitted_transformers[step_key], artifact_key
                            )
                            artifact_paths[artifact_key] = path
                        except ArtifactError as exc:
                            warnings_log.append(f"Artifact save failed: {exc.message}")
                            logger.warning("Artifact save failed: %s", exc.message)

                # Save feature schema JSONs
                for name, data in (
                    ("input_features", {"features": feature_names_input}),
                    ("output_features", {"features": feature_names_output}),
                    ("transformation_metadata", {"records": [m.to_dict() for m in transformation_metas]}),
                ):
                    try:
                        path = artifact_manager.save_json(data, name)
                        artifact_paths[name] = path
                    except ArtifactError as exc:
                        warnings_log.append(f"JSON save failed ({name}): {exc.message}")
                        logger.warning("JSON save failed (%s): %s", name, exc.message)

                # ── Schema validation before save ─────────────────────────
                self._validate_schema(
                    feature_names_output=feature_names_output,
                    X_train=X_train,
                    X_val=X_val,
                    X_test=X_test,
                )

                # Save Parquet splits
                parquet_targets = {
                    "X_train": X_train,
                    "X_validation": X_val,
                    "X_test": X_test,
                    "y_train": y_train,
                    "y_validation": y_val,
                    "y_test": y_test,
                }
                if train_split and val_split and test_split:
                    joined_targets = {
                        "train": pd.concat([X_train, y_train.rename(self._cfg.target_column)], axis=1),
                        "validation": pd.concat([X_val, y_val.rename(self._cfg.target_column)], axis=1),
                        "test": pd.concat([X_test, y_test.rename(self._cfg.target_column)], axis=1),
                    }
                    parquet_targets.update(joined_targets)

                for name, frame in parquet_targets.items():
                    try:
                        path = artifact_manager.save_parquet(frame, name)
                        parquet_paths[name] = path
                    except ArtifactError as exc:
                        warnings_log.append(f"Parquet save failed ({name}): {exc.message}")
                        logger.warning("Parquet save failed (%s): %s", name, exc.message)

                # ── New: dataset_schema ────────────────────────────────────
                categorical_features = list(
                    fitted_transformers["FREQUENCY_ENCODING"].columns
                    if "FREQUENCY_ENCODING" in fitted_transformers else []
                ) + list(
                    fitted_transformers["ONE_HOT_ENCODING"].columns
                    if "ONE_HOT_ENCODING" in fitted_transformers else []
                )
                numerical_features = [
                    c for c in feature_names_output if c not in categorical_features
                ]
                try:
                    dataset_schema_path = artifact_manager.save_dataset_schema(
                        dataset_name=dataset_name,
                        feature_names_input=feature_names_input,
                        feature_names_output=feature_names_output,
                        target_column=self._cfg.target_column,
                        categorical_features=categorical_features,
                        numerical_features=numerical_features,
                        preprocessing_version=self._cfg.preprocessing_version,
                        filename=self._cfg.dataset_schema_filename,
                    )
                    artifact_paths["dataset_schema"] = dataset_schema_path
                except ArtifactError as exc:
                    dataset_schema_path = ""
                    warnings_log.append(f"dataset_schema save failed: {exc.message}")
                    logger.warning("dataset_schema save failed: %s", exc.message)

                # ── New: class_distribution ────────────────────────────────
                try:
                    class_dist_path = artifact_manager.save_class_distribution(
                        train_y=y_train,
                        val_y=y_val,
                        test_y=y_test,
                        filename=self._cfg.class_distribution_filename,
                    )
                    artifact_paths["class_distribution"] = class_dist_path
                except ArtifactError as exc:
                    class_dist_path = ""
                    warnings_log.append(f"class_distribution save failed: {exc.message}")
                    logger.warning("class_distribution save failed: %s", exc.message)

                # ── New: data_integrity ────────────────────────────────────
                try:
                    data_integrity_path, integrity_warnings = artifact_manager.save_data_integrity(
                        X_train=X_train,
                        X_val=X_val,
                        X_test=X_test,
                        y_train=y_train,
                        y_val=y_val,
                        y_test=y_test,
                        feature_names_output=feature_names_output,
                        target_column=self._cfg.target_column,
                        filename=self._cfg.data_integrity_filename,
                    )
                    artifact_paths["data_integrity"] = data_integrity_path
                    warnings_log.extend(integrity_warnings)
                except ArtifactError as exc:
                    data_integrity_path = ""
                    warnings_log.append(f"data_integrity save failed: {exc.message}")
                    logger.warning("data_integrity save failed: %s", exc.message)

                # Sync latest/ before manifest (so manifest hashes are correct)
                try:
                    artifact_manager.sync_latest()
                except ArtifactError as exc:
                    warnings_log.append(f"Latest sync failed: {exc.message}")
                    logger.warning("Latest sync failed: %s", exc.message)

                # ── New: manifest ──────────────────────────────────────────
                try:
                    manifest_path = artifact_manager.save_manifest(
                        artifact_paths=artifact_paths,
                        run_id=run_id,
                        preprocessing_version=self._cfg.preprocessing_version,
                        pipeline_version="1.0.0",
                        filename=self._cfg.manifest_filename,
                    )
                    artifact_paths["manifest"] = manifest_path
                except ArtifactError as exc:
                    manifest_path = ""
                    warnings_log.append(f"manifest save failed: {exc.message}")
                    logger.warning("manifest save failed: %s", exc.message)

                step_ms = (time.perf_counter() - step_start) * 1_000
                timeline.append({
                    "step": entry.step_name,
                    "start_time": step_start_iso,
                    "end_time": datetime.now(tz=timezone.utc).isoformat(),
                    "duration_ms": round(step_ms, 2),
                })

        # ── 4. Build summary ──────────────────────────────────────────────
        freq_enc = fitted_transformers.get("FREQUENCY_ENCODING")
        ohe_enc = fitted_transformers.get("ONE_HOT_ENCODING")
        scaler = fitted_transformers.get("SCALING")

        ohe_expansion: dict[str, list[str]] = {}
        if ohe_enc and hasattr(ohe_enc, "_ohe") and ohe_enc._ohe is not None:
            for i, col in enumerate(ohe_enc.columns):
                ohe_expansion[col] = [
                    f for f in ohe_enc.get_feature_names_out()
                    if f.startswith(f"{col}_")
                ]

        summary = TransformationSummary(
            total_columns_input=len(feature_names_input) if "feature_names_input" in dir() else 0,
            total_columns_output=len(X_train.columns),
            columns_encoded_frequency=tuple(freq_enc.columns if freq_enc else []),
            columns_encoded_ohe=tuple(ohe_enc.columns if ohe_enc else []),
            columns_scaled=tuple(scaler.columns if scaler else []),
            columns_identity=tuple(
                scaler.columns if (scaler and scaler.transformer_type == "IDENTITY") else []
            ),
            columns_dropped=tuple(self._cfg.source_columns_to_drop),
            ohe_expansion_map=ohe_expansion,
            transformation_order=tuple(e.step_name for e in PREPROCESSING_REGISTRY if e.enabled),
        )

        duration_ms = (time.perf_counter() - _global_start) * 1_000

        # Resolve feature_names_input safely
        _feat_names_input: list[str] = []
        if train_split:
            _feat_names_input = [
                c for c in train_split.X.columns
                if c not in (self._cfg.datetime_column, self._cfg.target_column)
            ]

        report = PreprocessingReport(
            report_id=str(uuid.uuid4()),
            dataset_name=dataset_name,
            dataset_hash=dataset_hash,
            split_strategy="TIME_BASED",
            train_date_range=(
                train_split.start_date if train_split else "",
                train_split.end_date if train_split else "",
            ),
            validation_date_range=(
                val_split.start_date if val_split else "",
                val_split.end_date if val_split else "",
            ),
            test_date_range=(
                test_split.start_date if test_split else "",
                test_split.end_date if test_split else "",
            ),
            train_rows=len(X_train),
            validation_rows=len(X_val),
            test_rows=len(X_test),
            train_fraud_pct=train_split.fraud_pct if train_split else 0.0,
            validation_fraud_pct=val_split.fraud_pct if val_split else 0.0,
            test_fraud_pct=test_split.fraud_pct if test_split else 0.0,
            feature_names_input=tuple(_feat_names_input),
            feature_names_output=tuple(X_train.columns.tolist()),
            input_feature_count=len(_feat_names_input),
            output_feature_count=len(X_train.columns),
            transformation_summary=summary,
            transformation_metadata=tuple(transformation_metas),
            transformation_timeline=tuple(timeline),
            artifact_paths=artifact_paths,
            parquet_paths=parquet_paths,
            feature_schema_path=artifact_paths.get("input_features", ""),
            class_distribution_path=artifact_paths.get("class_distribution", ""),
            data_integrity_path=artifact_paths.get("data_integrity", ""),
            dataset_schema_path=artifact_paths.get("dataset_schema", ""),
            encoded_columns=tuple(sorted(list(summary.columns_encoded_frequency) + list(summary.columns_encoded_ohe))),
            frequency_encoded_columns=summary.columns_encoded_frequency,
            one_hot_encoded_columns=summary.columns_encoded_ohe,
            dropped_columns=summary.columns_dropped,
            scaling_enabled=self._cfg.enable_scaling,
            warnings=tuple(warnings_log),
            generated_at=datetime.now(tz=timezone.utc).isoformat(),
            duration_ms=round(duration_ms, 2),
        )

        logger.info(
            "Preprocessing complete | dataset='%s' | train=%d | val=%d | test=%d | "
            "features_in=%d | features_out=%d | artifacts=%d | duration=%.1fms",
            dataset_name,
            len(X_train),
            len(X_val),
            len(X_test),
            report.input_feature_count,
            report.output_feature_count,
            len(artifact_paths),
            duration_ms,
        )

        return PreprocessedDataset(
            X_train=X_train,
            X_validation=X_val,
            X_test=X_test,
            y_train=y_train,
            y_validation=y_val,
            y_test=y_test,
            feature_names=tuple(X_train.columns.tolist()),
            target_column=self._cfg.target_column,
            metadata=engineered_dataset.metadata,
            report=report,
        )

    def transform(self, engineered_dataset: EngineeredDataset) -> pd.DataFrame:
        """
        Transforms an EngineeredDataset using pre-fitted encoders/scalers at inference time.

        Args:
            engineered_dataset: Output of Feature Engineering.

        Returns:
            pd.DataFrame: Preprocessed DataFrame.
        """
        df = engineered_dataset.data.copy()
        
        # 1. Drop source/date columns and target is_fraud
        drop_cols = list(self._cfg.source_columns_to_drop) + ["is_fraud"]
        present_drops = [c for c in drop_cols if c in df.columns]
        if present_drops:
            df = df.drop(columns=present_drops)
            
        # 2. Apply pre-fitted transformers
        try:
            freq_enc = getattr(self, "frequency_encoder", None)
            if freq_enc:
                df = freq_enc.transform(df)
                
            ohe_enc = getattr(self, "ohe_encoder", None)
            if ohe_enc:
                df = ohe_enc.transform(df)
                
            scaler_enc = getattr(self, "scaler", None)
            if scaler_enc:
                df = scaler_enc.transform(df)
                
            return df
        except Exception as exc:
            from src.preprocessing.exceptions import TransformationError
            raise TransformationError(f"Preprocessing pipeline transform failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _guard(self, engineered_dataset: EngineeredDataset) -> None:
        """Guards against invalid pipeline inputs."""
        if engineered_dataset is None:
            raise PipelineError(stage="GUARD", detail="EngineeredDataset is None.")
        if engineered_dataset.report is None:
            raise PipelineError(
                stage="GUARD", detail="EngineeredDataset.report is None."
            )
        if not PREPROCESSING_REGISTRY:
            raise PipelineError(stage="GUARD", detail="PREPROCESSING_REGISTRY is empty.")
        if self._cfg.target_column not in engineered_dataset.data.columns:
            raise PipelineError(
                stage="GUARD",
                detail=(
                    f"Target column '{self._cfg.target_column}' not found in "
                    f"EngineeredDataset. Available columns: "
                    f"{list(engineered_dataset.data.columns)[:10]}..."
                ),
            )

    def _drop_datetime_and_target(self, X: pd.DataFrame) -> pd.DataFrame:
        """Removes datetime and target columns from a feature matrix."""
        drop = [
            c for c in (self._cfg.datetime_column, self._cfg.target_column)
            if c in X.columns
        ]
        if drop:
            X = X.drop(columns=drop)
        return X.copy()

    def _resolve_columns(
        self,
        columns_source: str,
        encoding_recs: dict,
        X_train: pd.DataFrame,
    ) -> list[str]:
        """
        Resolves the column list for a given registry step.

        Only returns columns that are actually present in X_train.
        """
        if columns_source == COL_SOURCE_FREQUENCY_RECS:
            candidates = [
                col for col, rec in encoding_recs.items()
                if rec.recommended_strategy == "FREQUENCY"
            ]
        elif columns_source == COL_SOURCE_OHE_RECS:
            candidates = [
                col for col, rec in encoding_recs.items()
                if rec.recommended_strategy == "ONE_HOT"
            ]
        elif columns_source == COL_SOURCE_SCALE_FEATURES:
            # Scale all numeric columns not already encoded
            encoded_cols = set(
                col for col, rec in encoding_recs.items()
            )
            candidates = [
                c for c in X_train.select_dtypes(include="number").columns
                if c not in encoded_cols
            ]
        else:
            candidates = []

        present = [c for c in candidates if c in X_train.columns]
        missing = [c for c in candidates if c not in X_train.columns]
        if missing:
            logger.warning(
                "Columns %s not found in X_train for step '%s' — skipped.",
                missing,
                columns_source,
            )
        return present

    def _build_transformer(
        self,
        step_name: str,
        columns: list[str],
    ) -> BaseTransformer:
        """Instantiates the correct transformer class for a registry step."""
        if step_name == "FREQUENCY_ENCODING":
            return FrequencyEncoder(
                columns=columns,
                unknown_value=self._cfg.frequency_unknown_value,
            )
        if step_name == "ONE_HOT_ENCODING":
            return OneHotEncoder(
                columns=columns,
                drop_first=self._cfg.ohe_drop_first,
                handle_unknown=self._cfg.ohe_handle_unknown,
            )
        if step_name == "SCALING":
            if self._cfg.enable_scaling:
                return RobustScalerTransformer(columns=columns)
            return IdentityTransformer(columns=columns)
        # Fallback for future entries
        return IdentityTransformer(columns=columns)

    def _validate_schema(
        self,
        feature_names_output: list[str],
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> None:
        """
        Validates output schema consistency across all three folds.

        Raises:
            PipelineError: If schema validation fails. This is unrecoverable —
                saving mismatched folds to disk would corrupt the training pipeline.

        Checks:
            - feature_names_output matches X_train.columns exactly.
            - X_train, X_val, X_test all share the same column set.
            - No duplicate column names exist.
        """
        actual_cols = list(X_train.columns)

        if actual_cols != feature_names_output:
            raise PipelineError(
                stage="SCHEMA_VALIDATION",
                detail=(
                    f"feature_names_output does not match X_train columns. "
                    f"Expected {feature_names_output[:5]}..., "
                    f"got {actual_cols[:5]}..."
                ),
            )

        val_cols = list(X_val.columns)
        test_cols = list(X_test.columns)

        if val_cols != actual_cols:
            raise PipelineError(
                stage="SCHEMA_VALIDATION",
                detail=(
                    f"X_validation column schema differs from X_train. "
                    f"X_train has {len(actual_cols)} cols, "
                    f"X_val has {len(val_cols)} cols."
                ),
            )

        if test_cols != actual_cols:
            raise PipelineError(
                stage="SCHEMA_VALIDATION",
                detail=(
                    f"X_test column schema differs from X_train. "
                    f"X_train has {len(actual_cols)} cols, "
                    f"X_test has {len(test_cols)} cols."
                ),
            )

        duplicates = [c for c in actual_cols if actual_cols.count(c) > 1]
        if duplicates:
            raise PipelineError(
                stage="SCHEMA_VALIDATION",
                detail=f"Duplicate column names detected: {set(duplicates)}",
            )

        logger.info(
            "Schema validation passed | features=%d | folds=3 consistent",
            len(actual_cols),
        )
