"""
EDAAnalyzer: the single public entry point of the EDA Layer.

Orchestrates all analysis modules via _ANALYSIS_REGISTRY, aggregates results,
invokes the InsightEngine, calls the Visualizer, exports reports, and
assembles the immutable EDAReport.

The analyzer NEVER modifies the DataFrame.

Registry execution order:
    overview → target → numerical → categorical →
    temporal → geographic → correlation

Fail-fast conditions (raises EDAError):
    - validated_dataset.data is None
    - target column absent from DataFrame

All other failures are caught per-module, recorded in EDAReport.warnings,
and the run continues.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config.logging import get_logger
from config.settings import settings
from src.eda.exceptions import AnalysisError, EDAError, VisualizationError
from src.eda.insights import InsightEngine
from src.eda.models import (
    CategoricalAnalysis,
    CorrelationAnalysis,
    DatasetOverview,
    EDAReport,
    GeographicAnalysis,
    NumericalAnalysis,
    TargetAnalysis,
    TemporalAnalysis,
    FeatureInventory,
    FeatureEngineeringBlueprint,
)
from src.eda.report import ReportExporter
from src.eda.visualizer import EDAVisualizer
from src.validation.models import ValidatedDataset

logger = get_logger(__name__)

# Target column name for the fraud detection dataset
_TARGET_COLUMN = "is_fraud"

# Datetime column for temporal analysis
_DATETIME_COLUMN = "trans_date_trans_time"

# Categorical columns to analyse (exclude PII columns from group-by fraud rates)
_CATEGORICAL_COLUMNS = (
    "merchant", "category", "gender", "state", "job", "city",
    "first", "last", "street", "trans_num", "dob",
)


class EDAAnalyzer:
    """
    Orchestrates the full EDA workflow for a ValidatedDataset.

    Public interface:
        >>> analyzer = EDAAnalyzer()
        >>> report = analyzer.analyze(validated_dataset)

    Args:
        output_dir: Root output directory. Default from settings.eda.output_dir.
        run_visualizations: If False, skips all plot generation.
        run_geographic: If False, skips geographic analysis (heavier I/O).
    """

    def __init__(
        self,
        output_dir: str | Path | None = None,
        run_visualizations: bool = True,
        run_geographic: bool = True,
    ) -> None:
        cfg = settings.eda
        self._output_dir = Path(output_dir or cfg.output_dir)
        self._reports_dir = self._output_dir / "reports"
        self._figures_dir = self._output_dir / "figures"
        self._run_visualizations = run_visualizations
        self._run_geographic = run_geographic
        self._cfg = cfg

        self._visualizer = EDAVisualizer(figures_dir=self._figures_dir)
        self._insight_engine = InsightEngine()

        # Registry maps result key → bound method
        # Populated in analyze() because methods need 'self'
        self._registry: list[tuple[str, Callable]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, validated_dataset: ValidatedDataset) -> EDAReport:
        """
        Executes the full EDA pipeline against a ValidatedDataset.

        Args:
            validated_dataset: Output from DatasetValidator.validate().

        Returns:
            EDAReport: Complete immutable EDA output.

        Raises:
            EDAError: If validated_dataset.data is None.
            AnalysisError: If the target column is absent from the DataFrame.
        """
        _global_start = time.perf_counter()
        warnings_log: list[str] = []

        df = validated_dataset.data
        metadata = validated_dataset.metadata
        dataset_name = metadata.file_name

        if df is None:
            raise EDAError(
                f"Cannot run EDA on '{dataset_name}': DataFrame is None.",
                {"dataset_name": dataset_name},
            )

        logger.info(
            "EDA started | dataset='%s' | shape=(%s rows x %s cols) | hash=%s",
            dataset_name,
            f"{len(df):,}",
            len(df.columns),
            (metadata.dataset_hash or "N/A")[:16],
        )

        self._ensure_output_dirs()

        # Build Feature Inventory (semantic roles)
        self._feature_inventory = self._build_feature_inventory(df)

        # Parse DOB safely and compute age metrics
        dob_analysis = self._run_dob_analysis(df, warnings_log)

        # Build the registry (bound methods referencing self)
        self._registry = [
            ("overview",     self._run_overview),
            ("target",       self._run_target_analysis),
            ("numerical",    self._run_numerical_analysis),
            ("categorical",  self._run_categorical_analysis),
            ("temporal",     self._run_temporal_analysis),
            ("geographic",   self._run_geographic_analysis),
            ("correlation",  self._run_correlation_analysis),
        ]

        # Execute registry
        results: dict = {}
        for key, method in self._registry:
            if key == "geographic" and not self._run_geographic:
                results[key] = None
                continue
            try:
                logger.debug("EDA module starting: '%s'.", key)
                _t = time.perf_counter()
                results[key] = method(df, metadata)
                _ms = (time.perf_counter() - _t) * 1_000
                logger.debug("EDA module complete: '%s' | %.1fms.", key, _ms)
            except AnalysisError as exc:
                logger.warning(
                    "EDA module '%s' failed (non-fatal): %s", key, exc.message
                )
                warnings_log.append(f"Module '{key}' failed: {exc.message}")
                results[key] = None
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "EDA module '%s' raised unexpected error: %s: %s",
                    key, type(exc).__name__, exc,
                )
                warnings_log.append(
                    f"Module '{key}' raised {type(exc).__name__}: {exc}"
                )
                results[key] = None

        # Fail-fast: target analysis must succeed
        if results.get("target") is None:
            raise AnalysisError(
                f"Target analysis failed for '{dataset_name}'. "
                "Cannot continue EDA without target column.",
                {"dataset_name": dataset_name, "target_column": _TARGET_COLUMN},
            )

        # Generate recommendations using Feature Inventory and DOB Analysis
        recommendations = self._insight_engine.generate(
            target=results["target"],
            numerical=results.get("numerical"),
            categorical=results.get("categorical"),
            temporal=results.get("temporal"),
            geographic=results.get("geographic"),
            correlation=results.get("correlation"),
            feature_inventory=self._feature_inventory,
            dob_analysis=dob_analysis,
        )
        logger.debug(
            "InsightEngine generated %d recommendation(s).", len(recommendations)
        )

        # Generate figures
        figure_paths: dict[str, str] = {}
        if self._run_visualizations:
            try:
                figure_paths = self._visualizer.generate_all(
                    df=df,
                    target=results["target"],
                    categorical=results.get("categorical"),
                    temporal=results.get("temporal"),
                    geographic=results.get("geographic"),
                    correlation=results.get("correlation"),
                )
                logger.info(
                    "Figures generated: %d figure(s).", len(figure_paths)
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Figure generation failed: %s", exc)
                warnings_log.append(f"Figure generation error: {exc}")

        # Build Feature Engineering Blueprint
        blueprint = self._build_feature_engineering_blueprint(df, self._feature_inventory)

        # Compile Extended Report Metadata
        duration_ms = (time.perf_counter() - _global_start) * 1_000
        import sys
        report_metadata = {
            "eda_version": "1.0.0",
            "project_version": "1.0.0",
            "python_version": sys.version.split()[0],
            "pandas_version": pd.__version__,
            "dataset_name": dataset_name,
            "dataset_hash": metadata.dataset_hash or "N/A",
            "analysis_timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "analysis_duration_ms": round(duration_ms, 2),
        }

        # Build report
        report = self._build_report(
            results=results,
            recommendations=recommendations,
            figure_paths=figure_paths,
            dataset_name=dataset_name,
            dataset_hash=metadata.dataset_hash,
            duration_ms=duration_ms,
            warnings_log=warnings_log,
            feature_inventory=self._feature_inventory,
            feature_engineering_blueprint=blueprint,
            dob_analysis=dob_analysis,
            report_metadata=report_metadata,
        )

        # Export reports
        report_paths: dict[str, str] = {}
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
        hash_prefix = (metadata.dataset_hash or "nohash")[:8]
        base_name = f"eda_report_{hash_prefix}_{timestamp}"

        try:
            self._reports_dir.mkdir(parents=True, exist_ok=True)
            md_path = self._reports_dir / f"{base_name}.md"
            json_path = self._reports_dir / f"{base_name}.json"

            ReportExporter(report).to_markdown(md_path)
            ReportExporter(report).to_json(json_path)

            # Store relative paths
            try:
                report_paths["markdown"] = str(md_path.relative_to(Path.cwd()))
                report_paths["json"] = str(json_path.relative_to(Path.cwd()))
            except ValueError:
                report_paths["markdown"] = str(md_path)
                report_paths["json"] = str(json_path)

            logger.info(
                "Reports written | markdown='%s' | json='%s'",
                report_paths.get("markdown"),
                report_paths.get("json"),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Report export failed: %s", exc)
            warnings_log.append(f"Report export error: {exc}")

        # Rebuild report with report_paths included (report is immutable)
        from dataclasses import replace
        report = replace(report, report_paths=report_paths)

        logger.info(
            "EDA complete | dataset='%s' | modules=%d | recs=%d | "
            "figures=%d | duration=%.1fms",
            dataset_name,
            len([v for v in results.values() if v is not None]),
            len(recommendations),
            len(figure_paths),
            duration_ms,
        )

        return report

    # ------------------------------------------------------------------
    # Registry methods — analysis modules
    # ------------------------------------------------------------------

    def _run_overview(self, df: pd.DataFrame, metadata) -> DatasetOverview:
        """Computes dataset shape, memory usage, and column type counts."""
        num_numeric = sum(
            1 for col in df.columns
            if hasattr(df[col].dtype, "kind") and df[col].dtype.kind in ("i", "u", "f")
        )
        num_categorical = sum(
            1 for col in df.columns
            if str(df[col].dtype) in ("object", "str", "string", "category")
        )
        num_datetime = sum(
            1 for col in df.columns
            if "datetime" in str(df[col].dtype)
        )
        missing = {
            col: int(df[col].isna().sum())
            for col in df.columns
            if df[col].isna().any()
        }
        memory_mb = round(df.memory_usage(deep=True).sum() / 1_048_576, 3)

        return DatasetOverview(
            num_rows=len(df),
            num_columns=len(df.columns),
            memory_usage_mb=memory_mb,
            num_numeric_columns=num_numeric,
            num_categorical_columns=num_categorical,
            num_datetime_columns=num_datetime,
            column_names=tuple(df.columns.tolist()),
            missing_values_summary=missing,
            dataset_hash=metadata.dataset_hash,
            analysis_timestamp=datetime.now(tz=timezone.utc).isoformat(),
        )

    def _run_target_analysis(self, df: pd.DataFrame, metadata) -> TargetAnalysis:
        """Computes fraud/legit distribution and imbalance ratio."""
        if _TARGET_COLUMN not in df.columns:
            raise AnalysisError(
                f"Target column '{_TARGET_COLUMN}' not found in DataFrame.",
                {"available_columns": list(df.columns[:10])},
            )

        total = len(df)
        fraud_count = int(df[_TARGET_COLUMN].sum())
        legit_count = total - fraud_count
        fraud_pct = round((fraud_count / total) * 100, 6) if total > 0 else 0.0
        legit_pct = round(100.0 - fraud_pct, 6)
        imbalance_ratio = round(legit_count / fraud_count, 2) if fraud_count > 0 else float("inf")

        return TargetAnalysis(
            target_column=_TARGET_COLUMN,
            fraud_count=fraud_count,
            legitimate_count=legit_count,
            fraud_pct=fraud_pct,
            legitimate_pct=legit_pct,
            imbalance_ratio=imbalance_ratio,
            class_distribution={
                "0": legit_count,
                "1": fraud_count,
            },
        )

    def _build_feature_inventory(self, df: pd.DataFrame) -> FeatureInventory:
        """Classifies every column into semantic roles."""
        identifiers = []
        pii = []
        numeric = []
        categorical = []
        temporal = []
        geographic = []
        target = []

        identifier_patterns = self._cfg.identifier_patterns
        pii_columns = self._cfg.pii_columns
        dob_patterns = self._cfg.dob_patterns

        for col in df.columns:
            col_lower = col.lower()
            if col == _TARGET_COLUMN:
                target.append(col)
            elif col_lower in ("first", "last", "street", "email", "phone"):
                pii.append(col)
            elif any(pat in col for pat in identifier_patterns) or any(pat in col_lower for pat in identifier_patterns):
                identifiers.append(col)
            elif any(pat in col_lower for pat in dob_patterns):
                temporal.append(col)
            elif col_lower in ("state", "city", "zip", "lat", "long", "merch_lat", "merch_long") or col_lower.endswith("zip") or col_lower.endswith("state"):
                geographic.append(col)
            elif col == _DATETIME_COLUMN or "date" in col_lower or "time" in col_lower:
                temporal.append(col)
            elif hasattr(df[col].dtype, "kind") and df[col].dtype.kind in ("i", "u", "f"):
                numeric.append(col)
            else:
                categorical.append(col)

        # Drop before training should include identifiers and PII, except card cc_num might be handled differently,
        # but to match the exact user output: Unnamed: 0, trans_num, first, last, street
        drop_before_training = sorted(list(set(identifiers + pii)))
        if "cc_num" in drop_before_training:
            drop_before_training.remove("cc_num")

        return FeatureInventory(
            identifiers=tuple(sorted(identifiers)),
            pii=tuple(sorted(pii)),
            numeric=tuple(sorted(numeric)),
            categorical=tuple(sorted(categorical)),
            temporal=tuple(sorted(temporal)),
            geographic=tuple(sorted(geographic)),
            target=tuple(sorted(target)),
            drop_before_training=tuple(sorted(drop_before_training)),
        )

    def _run_dob_analysis(self, df: pd.DataFrame, warnings_log: list[str]) -> dict | None:
        """Parses DOB column securely and computes age demographics statistics."""
        dob_patterns = self._cfg.dob_patterns
        dob_col = None
        for col in df.columns:
            if any(pat in col.lower() for pat in dob_patterns):
                dob_col = col
                break
        if dob_col is None:
            return None

        try:
            dob_series = pd.to_datetime(df[dob_col], errors="coerce")
            if dob_series.isna().all():
                raise ValueError("All DOB values are unparseable.")

            ref_date = datetime.now(tz=timezone.utc)
            if _DATETIME_COLUMN in df.columns:
                try:
                    ref_date = pd.to_datetime(df[_DATETIME_COLUMN]).max()
                except Exception:
                    pass

            ages = (ref_date - dob_series).dt.days / 365.25
            ages_clean = ages.dropna()

            if ages_clean.empty:
                raise ValueError("No valid ages computed.")

            bins = [0, 18, 25, 35, 45, 55, 65, 120]
            labels = ["<18", "18-25", "26-35", "36-45", "46-55", "56-65", "66+"]
            age_groups = pd.cut(ages_clean, bins=bins, labels=labels).value_counts().to_dict()
            age_groups_str = {str(k): int(v) for k, v in age_groups.items()}

            stats = {
                "mean": round(float(ages_clean.mean()), 2),
                "median": round(float(ages_clean.median()), 2),
                "min": round(float(ages_clean.min()), 2),
                "max": round(float(ages_clean.max()), 2),
                "std": round(float(ages_clean.std()), 2) if len(ages_clean) > 1 else 0.0,
            }

            return {
                "parsed": True,
                "column_used": dob_col,
                "statistics": stats,
                "age_groups": age_groups_str,
            }
        except Exception as exc:
            msg = f"DOB parsing/analysis failed: {exc}"
            logger.warning(msg)
            warnings_log.append(msg)
            return {
                "parsed": False,
                "column_used": dob_col,
                "error": str(exc),
            }

    def _build_feature_engineering_blueprint(
        self, df: pd.DataFrame, inventory: FeatureInventory
    ) -> FeatureEngineeringBlueprint:
        """Assembles concrete recipe guiding feature engineering."""
        candidates = []
        for col in list(inventory.numeric) + list(inventory.categorical) + list(inventory.temporal) + list(inventory.geographic):
            if col not in inventory.drop_before_training and col != _TARGET_COLUMN:
                candidates.append(col)

        # Append engineered features conceptually
        for conceptual in ("hour", "distance", "age"):
            if conceptual not in candidates:
                candidates.append(conceptual)

        drop = list(inventory.drop_before_training)

        transform = []
        if "amt" in df.columns:
            transform.append("amt")
        if "city_pop" in df.columns:
            transform.append("city_pop")

        encode = []
        for col in ("merchant", "category", "state", "job"):
            if col in df.columns:
                encode.append(col)

        scale = []
        if "amt" in df.columns:
            scale.append("amt")
        if "city_pop" in df.columns:
            scale.append("city_pop")

        return FeatureEngineeringBlueprint(
            candidate_features=tuple(sorted(candidates)),
            drop_features=tuple(sorted(drop)),
            transform_features=tuple(sorted(transform)),
            encode_features=tuple(sorted(encode)),
            scale_features=tuple(sorted(scale)),
        )

    def _run_numerical_analysis(self, df: pd.DataFrame, metadata) -> NumericalAnalysis | None:
        """Computes descriptive statistics for all numeric columns, excluding identifiers."""
        numeric_cols = [
            col for col in df.columns
            if hasattr(df[col].dtype, "kind")
            and df[col].dtype.kind in ("i", "u", "f")
            and col != _TARGET_COLUMN
            and col not in self._feature_inventory.identifiers
        ]
        if not numeric_cols:
            return None

        threshold_skew = self._cfg.skewness_threshold
        threshold_outlier = self._cfg.outlier_pct_threshold
        n = len(df)

        fraud_mask = df[_TARGET_COLUMN] == 1 if _TARGET_COLUMN in df.columns else None
        legit_mask = df[_TARGET_COLUMN] == 0 if _TARGET_COLUMN in df.columns else None

        summaries: dict[str, dict] = {}
        for col in numeric_cols:
            s = df[col]
            q25 = float(s.quantile(0.25))
            q75 = float(s.quantile(0.75))
            iqr = q75 - q25
            lower = q25 - 1.5 * iqr
            upper = q75 + 1.5 * iqr
            outlier_count = int(((s < lower) | (s > upper)).sum())
            outlier_pct = round((outlier_count / n) * 100, 4) if n > 0 else 0.0

            fraud_mean = (
                round(float(df.loc[fraud_mask, col].mean()), 6)
                if fraud_mask is not None and fraud_mask.any()
                else None
            )
            legit_mean = (
                round(float(df.loc[legit_mask, col].mean()), 6)
                if legit_mask is not None and legit_mask.any()
                else None
            )

            summaries[col] = {
                "mean": round(float(s.mean()), 6),
                "median": round(float(s.median()), 6),
                "std": round(float(s.std()), 6),
                "min": round(float(s.min()), 6),
                "max": round(float(s.max()), 6),
                "q25": round(q25, 6),
                "q75": round(q75, 6),
                "iqr": round(iqr, 6),
                "skewness": round(float(s.skew()), 6),
                "kurtosis": round(float(s.kurt()), 6),
                "outlier_count": outlier_count,
                "outlier_pct": outlier_pct,
                "fraud_mean": fraud_mean,
                "legit_mean": legit_mean,
            }

        high_skewness = tuple(
            col for col in numeric_cols
            if abs(summaries[col]["skewness"]) > threshold_skew
        )
        high_outlier = tuple(
            col for col in numeric_cols
            if summaries[col]["outlier_pct"] > threshold_outlier
        )

        return NumericalAnalysis(
            summaries=summaries,
            high_skewness_columns=high_skewness,
            high_outlier_columns=high_outlier,
            columns_analysed=tuple(numeric_cols),
        )

    def _run_categorical_analysis(
        self, df: pd.DataFrame, metadata
    ) -> CategoricalAnalysis | None:
        """Computes cardinality and fraud rates for categorical columns, excluding PII, DOB and identifiers."""
        categorical_cols = [
            col for col in _CATEGORICAL_COLUMNS
            if col in df.columns
            and str(df[col].dtype) in ("object", "str", "string", "category")
            and col not in self._feature_inventory.identifiers
            and col not in self._feature_inventory.pii
            and not any(pat in col.lower() for pat in self._cfg.dob_patterns)
        ]
        if not categorical_cols:
            return None

        threshold_card = self._cfg.high_cardinality_threshold
        has_target = _TARGET_COLUMN in df.columns
        min_support = self._cfg.min_category_support

        summaries: dict[str, dict] = {}
        for col in categorical_cols:
            s = df[col]
            unique_count = int(s.nunique())
            missing_count = int(s.isna().sum())
            is_high_card = unique_count > threshold_card

            all_vc = s.value_counts(dropna=True)

            if has_target:
                fraud_by_cat = (
                    df.groupby(col)[_TARGET_COLUMN]
                    .mean()
                    .dropna()
                )
                fraud_rates = {str(k): round(float(v), 6) for k, v in fraud_by_cat.items()}
            else:
                fraud_rates = {}

            # Top N by frequency
            vc = all_vc.head(self._cfg.top_n_categories)
            total = len(df)
            top_categories = []
            for val, count in vc.items():
                top_categories.append({
                    "value": str(val),
                    "count": int(count),
                    "pct": round(int(count) / total * 100, 4),
                    "fraud_rate": fraud_rates.get(str(val), 0.0),
                })

            # Exclude categories below minimum support from ranking
            supported_categories = [
                str(k) for k, v in all_vc.items() if v >= min_support
            ]
            supported_fraud_rates = {
                k: fraud_rates[k] for k in supported_categories if k in fraud_rates
            }

            highest_fraud_cat = ""
            highest_fraud_rate = 0.0
            if supported_fraud_rates:
                highest_fraud_cat = max(supported_fraud_rates, key=supported_fraud_rates.get)
                highest_fraud_rate = supported_fraud_rates[highest_fraud_cat]

            summaries[col] = {
                "unique_count": unique_count,
                "is_high_cardinality": is_high_card,
                "missing_count": missing_count,
                "top_categories": top_categories,
                "highest_fraud_category": highest_fraud_cat,
                "highest_fraud_rate": highest_fraud_rate,
            }

        high_cardinality = tuple(
            col for col in categorical_cols
            if summaries[col]["is_high_cardinality"]
        )

        return CategoricalAnalysis(
            summaries=summaries,
            high_cardinality_columns=high_cardinality,
            columns_analysed=tuple(categorical_cols),
        )

    def _run_temporal_analysis(self, df: pd.DataFrame, metadata) -> TemporalAnalysis | None:
        """Parses datetime column and computes fraud rates by time bucket."""
        if _DATETIME_COLUMN not in df.columns:
            raise AnalysisError(
                f"Datetime column '{_DATETIME_COLUMN}' not found.",
                {"available_columns": list(df.columns)},
            )

        try:
            dt = pd.to_datetime(df[_DATETIME_COLUMN])
        except Exception as exc:
            raise AnalysisError(
                f"Failed to parse '{_DATETIME_COLUMN}' as datetime: {exc}",
                {"error": str(exc)},
            ) from exc

        if _TARGET_COLUMN not in df.columns:
            raise AnalysisError(
                "Target column missing — cannot compute temporal fraud rates.",
                {},
            )

        temp_df = pd.DataFrame({
            "hour": dt.dt.hour,
            "weekday": dt.dt.day_name(),
            "month": dt.dt.month_name(),
            "is_fraud": df[_TARGET_COLUMN].values,
        })

        def _fraud_rate_dict(col: str) -> dict[str, float]:
            return {
                str(k): round(float(v), 6)
                for k, v in temp_df.groupby(col)["is_fraud"].mean().items()
            }

        def _volume_dict(col: str) -> dict[str, int]:
            return {
                str(k): int(v)
                for k, v in temp_df.groupby(col)["is_fraud"].count().items()
            }

        fraud_by_hour = _fraud_rate_dict("hour")
        fraud_by_weekday = _fraud_rate_dict("weekday")
        fraud_by_month = _fraud_rate_dict("month")
        volume_by_hour = _volume_dict("hour")

        peak_hour = int(max(fraud_by_hour, key=fraud_by_hour.get))  # type: ignore[arg-type]
        peak_weekday = max(fraud_by_weekday, key=fraud_by_weekday.get)  # type: ignore[arg-type]
        peak_month = max(fraud_by_month, key=fraud_by_month.get)  # type: ignore[arg-type]

        return TemporalAnalysis(
            fraud_by_hour=fraud_by_hour,
            fraud_by_weekday=fraud_by_weekday,
            fraud_by_month=fraud_by_month,
            peak_fraud_hour=peak_hour,
            peak_fraud_weekday=peak_weekday,
            peak_fraud_month=peak_month,
            transaction_volume_by_hour=volume_by_hour,
            datetime_column_used=_DATETIME_COLUMN,
        )

    def _run_geographic_analysis(
        self, df: pd.DataFrame, metadata
    ) -> GeographicAnalysis | None:
        """Computes geographic fraud rates, with minimum support filtering and state statistics."""
        if "state" not in df.columns or _TARGET_COLUMN not in df.columns:
            raise AnalysisError(
                "Columns 'state' or 'is_fraud' not found for geographic analysis.",
                {},
            )

        min_support = self._cfg.min_category_support

        state_counts = df["state"].value_counts()
        supported_states = [k for k, v in state_counts.items() if v >= min_support]

        # Calculate state_statistics
        state_statistics = {}
        for state in df["state"].dropna().unique():
            state_str = str(state)
            s_df = df[df["state"] == state]
            tx_count = len(s_df)
            f_count = int(s_df[_TARGET_COLUMN].sum())
            f_pct = round((f_count / tx_count) * 100, 4) if tx_count > 0 else 0.0
            state_statistics[state_str] = {
                "transaction_count": tx_count,
                "fraud_count": f_count,
                "fraud_percentage": f_pct,
            }

        fraud_by_state = {
            str(k): round(float(v), 6)
            for k, v in df.groupby("state")[_TARGET_COLUMN].mean().items()
        }
        fraud_count_by_state = {
            str(k): int(v)
            for k, v in df.groupby("state")[_TARGET_COLUMN].sum().items()
        }

        # Filter states ranking using minimum support
        supported_fraud_by_state = {
            k: v for k, v in fraud_by_state.items() if k in supported_states
        }
        top_states = sorted(supported_fraud_by_state, key=supported_fraud_by_state.get, reverse=True)[:10]

        top_cities: list[str] = []
        if "city" in df.columns:
            city_counts = df["city"].value_counts()
            supported_cities = [k for k, v in city_counts.items() if v >= min_support]
            city_fraud = df.groupby("city")[_TARGET_COLUMN].sum()
            supported_city_fraud = city_fraud.loc[city_fraud.index.isin(supported_cities)].sort_values(ascending=False)
            top_cities = [str(c) for c in supported_city_fraud.head(10).index.tolist()]

        return GeographicAnalysis(
            fraud_by_state=fraud_by_state,
            fraud_count_by_state=fraud_count_by_state,
            top_fraud_states=top_states,
            top_fraud_cities=top_cities,
            total_states=int(df["state"].nunique()),
            total_cities=int(df["city"].nunique()) if "city" in df.columns else 0,
            state_statistics=state_statistics,
        )

    def _run_correlation_analysis(
        self, df: pd.DataFrame, metadata
    ) -> CorrelationAnalysis | None:
        """Computes Pearson and Point-Biserial target correlations, excluding identifiers."""
        numeric_cols = [
            col for col in df.columns
            if hasattr(df[col].dtype, "kind")
            and df[col].dtype.kind in ("i", "u", "f")
            and col not in self._feature_inventory.identifiers
        ]
        if len(numeric_cols) < 2:
            return None

        threshold = self._cfg.high_correlation_threshold
        corr_matrix = df[numeric_cols].corr()

        target_corr = {
            "pearson": {},
            "point_biserial": {}
        }

        if _TARGET_COLUMN in df.columns:
            y = df[_TARGET_COLUMN]
            for col in numeric_cols:
                if col != _TARGET_COLUMN:
                    # Pearson target correlation
                    r_val = float(df[col].corr(y))
                    if not np.isnan(r_val):
                        target_corr["pearson"][col] = round(r_val, 6)

                    # Point-Biserial target correlation
                    mask_1 = y == 1
                    mask_0 = y == 0
                    n1 = mask_1.sum()
                    n0 = mask_0.sum()
                    n = n1 + n0
                    if n > 1 and n1 > 0 and n0 > 0:
                        m1 = df.loc[mask_1, col].mean()
                        m0 = df.loc[mask_0, col].mean()
                        std = df[col].std(ddof=1)
                        if std > 0:
                            import math
                            r_pb = ((m1 - m0) / std) * math.sqrt((n1 * n0) / (n * (n - 1)))
                            target_corr["point_biserial"][col] = round(float(r_pb), 6)
                        else:
                            target_corr["point_biserial"][col] = 0.0
                    else:
                        target_corr["point_biserial"][col] = 0.0

        sorted_corr = sorted(
            target_corr["pearson"].items(), key=lambda x: x[1], reverse=True
        )
        top_positive = [[col, round(r, 6)] for col, r in sorted_corr if r > 0][:5]
        top_negative = [[col, round(r, 6)] for col, r in sorted_corr if r < 0][:5]

        # High correlation pairs (upper triangle)
        high_pairs: list[dict] = []
        cols = list(corr_matrix.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r = corr_matrix.iloc[i, j]
                if not np.isnan(r) and abs(r) > threshold:
                    high_pairs.append({
                        "col_a": cols[i],
                        "col_b": cols[j],
                        "r": round(float(r), 6),
                    })

        return CorrelationAnalysis(
            target_correlations=target_corr,
            top_positive_correlations=top_positive,
            top_negative_correlations=top_negative,
            high_correlation_pairs=high_pairs,
            numeric_columns_used=tuple(numeric_cols),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_output_dirs(self) -> None:
        """Creates all required output directories."""
        self._reports_dir.mkdir(parents=True, exist_ok=True)
        self._figures_dir.mkdir(parents=True, exist_ok=True)

    def _build_report(
        self,
        results: dict,
        recommendations: tuple,
        figure_paths: dict[str, str],
        dataset_name: str,
        dataset_hash: str | None,
        duration_ms: float,
        warnings_log: list[str],
        feature_inventory: FeatureInventory,
        feature_engineering_blueprint: FeatureEngineeringBlueprint,
        dob_analysis: dict | None,
        report_metadata: dict,
    ) -> EDAReport:
        """Assembles the immutable EDAReport from all collected components."""
        return EDAReport(
            report_id=str(uuid.uuid4()),
            dataset_name=dataset_name,
            dataset_hash=dataset_hash,
            overview=results["overview"],
            target_analysis=results["target"],
            numerical_analysis=results.get("numerical"),
            categorical_analysis=results.get("categorical"),
            temporal_analysis=results.get("temporal"),
            geographic_analysis=results.get("geographic"),
            correlation_analysis=results.get("correlation"),
            recommendations=recommendations,
            figure_paths=figure_paths,
            report_paths={},  # populated after export
            generated_at=datetime.now(tz=timezone.utc).isoformat(),
            analysis_duration_ms=round(duration_ms, 2),
            warnings=tuple(warnings_log),
            feature_inventory=feature_inventory,
            feature_engineering_blueprint=feature_engineering_blueprint,
            dob_analysis=dob_analysis,
            report_metadata=report_metadata,
        )
