"""
Immutable data models for the EDA Layer.

All models are frozen dataclasses. Every model exposes to_dict() returning
a fully JSON-serialisable dict (no pandas, no numpy objects).

Model hierarchy:
    DatasetOverview        — shape, memory, column type counts
    TargetAnalysis         — fraud/legit counts, imbalance ratio
    NumericalAnalysis      — per-column descriptive stats as dicts
    CategoricalAnalysis    — per-column cardinality and fraud rates as dicts
    TemporalAnalysis       — fraud rates by hour/weekday/month
    GeographicAnalysis     — fraud rates by state and city
    CorrelationAnalysis    — correlation matrix, target correlations
    EDARecommendation      — single actionable recommendation
    EDAReport              — complete output of the EDA layer
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# DatasetOverview
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetOverview:
    """
    High-level dataset shape and memory summary.

    Attributes:
        num_rows: Total row count.
        num_columns: Total column count.
        memory_usage_mb: DataFrame memory usage in megabytes (deep=True).
        num_numeric_columns: Count of numeric dtype columns.
        num_categorical_columns: Count of object/string dtype columns.
        num_datetime_columns: Count of datetime dtype columns.
        column_names: Ordered tuple of all column names.
        missing_values_summary: Mapping of column name to null count.
            Only columns that have at least one null are included.
        dataset_hash: SHA-256 content hash from DatasetMetadata, or None.
        analysis_timestamp: ISO 8601 UTC timestamp for this run.
    """

    num_rows: int
    num_columns: int
    memory_usage_mb: float
    num_numeric_columns: int
    num_categorical_columns: int
    num_datetime_columns: int
    column_names: tuple[str, ...]
    missing_values_summary: dict[str, int] = field(hash=False, compare=False)
    dataset_hash: str | None
    analysis_timestamp: str

    def to_dict(self) -> dict:
        """Returns a JSON-serialisable dictionary."""
        return {
            "num_rows": self.num_rows,
            "num_columns": self.num_columns,
            "memory_usage_mb": self.memory_usage_mb,
            "num_numeric_columns": self.num_numeric_columns,
            "num_categorical_columns": self.num_categorical_columns,
            "num_datetime_columns": self.num_datetime_columns,
            "column_names": list(self.column_names),
            "missing_values_summary": self.missing_values_summary,
            "dataset_hash": self.dataset_hash,
            "analysis_timestamp": self.analysis_timestamp,
        }


# ---------------------------------------------------------------------------
# TargetAnalysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetAnalysis:
    """
    Distribution of the target column (is_fraud).

    Attributes:
        target_column: Name of the label column.
        fraud_count: Absolute count of positive (fraud=1) rows.
        legitimate_count: Absolute count of negative (fraud=0) rows.
        fraud_pct: Percentage of fraud rows (0-100).
        legitimate_pct: Percentage of legitimate rows (0-100).
        imbalance_ratio: legitimate_count / fraud_count.
        class_distribution: Mapping of class label string to count.
    """

    target_column: str
    fraud_count: int
    legitimate_count: int
    fraud_pct: float
    legitimate_pct: float
    imbalance_ratio: float
    class_distribution: dict[str, int] = field(hash=False, compare=False)

    def to_dict(self) -> dict:
        """Returns a JSON-serialisable dictionary."""
        return {
            "target_column": self.target_column,
            "fraud_count": self.fraud_count,
            "legitimate_count": self.legitimate_count,
            "fraud_pct": self.fraud_pct,
            "legitimate_pct": self.legitimate_pct,
            "imbalance_ratio": self.imbalance_ratio,
            "class_distribution": self.class_distribution,
        }


# ---------------------------------------------------------------------------
# NumericalAnalysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NumericalAnalysis:
    """
    Descriptive statistics for all numeric columns, stored as dicts.

    Attributes:
        summaries: Mapping of column name to a stats dict containing:
            mean, median, std, min, max, q25, q75, iqr, skewness,
            kurtosis, outlier_count, outlier_pct, fraud_mean, legit_mean.
        high_skewness_columns: Columns with abs(skewness) > threshold.
        high_outlier_columns: Columns with outlier_pct > threshold.
        columns_analysed: Ordered tuple of column names that were analysed.
    """

    summaries: dict[str, dict] = field(hash=False, compare=False)
    high_skewness_columns: tuple[str, ...]
    high_outlier_columns: tuple[str, ...]
    columns_analysed: tuple[str, ...]

    def to_dict(self) -> dict:
        """Returns a JSON-serialisable dictionary."""
        return {
            "summaries": self.summaries,
            "high_skewness_columns": list(self.high_skewness_columns),
            "high_outlier_columns": list(self.high_outlier_columns),
            "columns_analysed": list(self.columns_analysed),
        }


# ---------------------------------------------------------------------------
# CategoricalAnalysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CategoricalAnalysis:
    """
    Frequency and fraud-rate analysis for all categorical columns.

    Attributes:
        summaries: Mapping of column name to a summary dict containing:
            unique_count, is_high_cardinality, top_categories (list of
            {value, count, pct, fraud_rate}), highest_fraud_category,
            highest_fraud_rate, missing_count.
        high_cardinality_columns: Columns with unique_count > threshold.
        columns_analysed: Ordered tuple of column names that were analysed.
    """

    summaries: dict[str, dict] = field(hash=False, compare=False)
    high_cardinality_columns: tuple[str, ...]
    columns_analysed: tuple[str, ...]

    def to_dict(self) -> dict:
        """Returns a JSON-serialisable dictionary."""
        return {
            "summaries": self.summaries,
            "high_cardinality_columns": list(self.high_cardinality_columns),
            "columns_analysed": list(self.columns_analysed),
        }


# ---------------------------------------------------------------------------
# TemporalAnalysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemporalAnalysis:
    """
    Fraud rate distributions across time dimensions.

    Attributes:
        fraud_by_hour: Mapping of hour string (0-23) to fraud rate.
        fraud_by_weekday: Mapping of weekday name to fraud rate.
        fraud_by_month: Mapping of month name to fraud rate.
        peak_fraud_hour: Hour (0-23) with the highest fraud rate.
        peak_fraud_weekday: Weekday name with the highest fraud rate.
        peak_fraud_month: Month name with the highest fraud rate.
        transaction_volume_by_hour: Total transaction count per hour.
        datetime_column_used: Name of the column that was parsed.
    """

    fraud_by_hour: dict[str, float] = field(hash=False, compare=False)
    fraud_by_weekday: dict[str, float] = field(hash=False, compare=False)
    fraud_by_month: dict[str, float] = field(hash=False, compare=False)
    peak_fraud_hour: int
    peak_fraud_weekday: str
    peak_fraud_month: str
    transaction_volume_by_hour: dict[str, int] = field(hash=False, compare=False)
    datetime_column_used: str

    def to_dict(self) -> dict:
        """Returns a JSON-serialisable dictionary."""
        return {
            "fraud_by_hour": self.fraud_by_hour,
            "fraud_by_weekday": self.fraud_by_weekday,
            "fraud_by_month": self.fraud_by_month,
            "peak_fraud_hour": self.peak_fraud_hour,
            "peak_fraud_weekday": self.peak_fraud_weekday,
            "peak_fraud_month": self.peak_fraud_month,
            "transaction_volume_by_hour": self.transaction_volume_by_hour,
            "datetime_column_used": self.datetime_column_used,
        }


# ---------------------------------------------------------------------------
# GeographicAnalysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeographicAnalysis:
    """
    Geographic fraud distribution by state and city.

    Attributes:
        fraud_by_state: Mapping of state code to fraud rate.
        fraud_count_by_state: Mapping of state code to absolute fraud count.
        top_fraud_states: Top N states by fraud rate (list of state codes).
        top_fraud_cities: Top N cities by fraud count (list of city names).
        total_states: Number of unique states in the dataset.
        total_cities: Number of unique cities in the dataset.
    """

    fraud_by_state: dict[str, float] = field(hash=False, compare=False)
    fraud_count_by_state: dict[str, int] = field(hash=False, compare=False)
    top_fraud_states: list[str] = field(hash=False, compare=False)
    top_fraud_cities: list[str] = field(hash=False, compare=False)
    total_states: int
    total_cities: int
    state_statistics: dict[str, dict] = field(default_factory=dict, hash=False, compare=False)

    def to_dict(self) -> dict:
        """Returns a JSON-serialisable dictionary."""
        return {
            "fraud_by_state": self.fraud_by_state,
            "fraud_count_by_state": self.fraud_count_by_state,
            "top_fraud_states": self.top_fraud_states,
            "top_fraud_cities": self.top_fraud_cities,
            "total_states": self.total_states,
            "total_cities": self.total_cities,
            "state_statistics": self.state_statistics,
        }


# ---------------------------------------------------------------------------
# CorrelationAnalysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorrelationAnalysis:
    """
    Pairwise feature correlations and target correlation summary.

    Attributes:
        target_correlations: Mapping of feature name to its Pearson
            correlation with the target column.
        top_positive_correlations: Top 5 features most positively
            correlated with the target as list of [column, r] pairs.
        top_negative_correlations: Top 5 features most negatively
            correlated with the target as list of [column, r] pairs.
        high_correlation_pairs: Feature pairs with abs(r) > threshold.
            Each entry is {col_a, col_b, r}.
        numeric_columns_used: Columns included in the correlation matrix.
    """

    target_correlations: dict = field(hash=False, compare=False)
    top_positive_correlations: list[list] = field(hash=False, compare=False)
    top_negative_correlations: list[list] = field(hash=False, compare=False)
    high_correlation_pairs: list[dict] = field(hash=False, compare=False)
    numeric_columns_used: tuple[str, ...]

    def to_dict(self) -> dict:
        """Returns a JSON-serialisable dictionary."""
        return {
            "target_correlations": self.target_correlations,
            "top_positive_correlations": self.top_positive_correlations,
            "top_negative_correlations": self.top_negative_correlations,
            "high_correlation_pairs": self.high_correlation_pairs,
            "numeric_columns_used": list(self.numeric_columns_used),
        }


# ---------------------------------------------------------------------------
# EDARecommendation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EDARecommendation:
    """
    A single actionable recommendation produced by InsightEngine.

    Attributes:
        category: Concern this addresses. One of: IMBALANCE, ENCODING,
            FEATURE_ENGINEERING, SCALING, PREPROCESSING, MODEL_SELECTION.
        priority: Urgency level. One of: HIGH, MEDIUM, LOW.
        finding: What was observed in the data (factual, quantitative).
        recommendation: Specific implementation-independent action.
        affects_stage: Pipeline stages this recommendation applies to.
    """

    category: str
    priority: str
    finding: str
    recommendation: str
    affects_stage: tuple[str, ...]

    def to_dict(self) -> dict:
        """Returns a JSON-serialisable dictionary."""
        return {
            "category": self.category,
            "priority": self.priority,
            "finding": self.finding,
            "recommendation": self.recommendation,
            "affects_stage": list(self.affects_stage),
        }


# ---------------------------------------------------------------------------
# FeatureInventory
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureInventory:
    """
    Inventory classifying columns by semantic roles.
    """

    identifiers: tuple[str, ...]
    pii: tuple[str, ...]
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    temporal: tuple[str, ...]
    geographic: tuple[str, ...]
    target: tuple[str, ...]
    drop_before_training: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "identifiers": list(self.identifiers),
            "pii": list(self.pii),
            "numeric": list(self.numeric),
            "categorical": list(self.categorical),
            "temporal": list(self.temporal),
            "geographic": list(self.geographic),
            "target": list(self.target),
            "drop_before_training": list(self.drop_before_training),
        }


# ---------------------------------------------------------------------------
# FeatureEngineeringBlueprint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureEngineeringBlueprint:
    """
    Actionable recipe guiding feature engineering.
    """

    candidate_features: tuple[str, ...]
    drop_features: tuple[str, ...]
    transform_features: tuple[str, ...]
    encode_features: tuple[str, ...]
    scale_features: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "candidate_features": list(self.candidate_features),
            "drop_features": list(self.drop_features),
            "transform_features": list(self.transform_features),
            "encode_features": list(self.encode_features),
            "scale_features": list(self.scale_features),
        }


# ---------------------------------------------------------------------------
# EDAReport
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EDAReport:
    """
    Complete, immutable output of the EDA Layer.

    This is the only object passed between the EDA Layer and downstream
    consumers. All fields are JSON-serialisable via to_dict().
    """

    report_id: str
    dataset_name: str
    dataset_hash: str | None
    overview: DatasetOverview
    target_analysis: TargetAnalysis
    numerical_analysis: NumericalAnalysis | None
    categorical_analysis: CategoricalAnalysis | None
    temporal_analysis: TemporalAnalysis | None
    geographic_analysis: GeographicAnalysis | None
    correlation_analysis: CorrelationAnalysis | None
    recommendations: tuple[EDARecommendation, ...]
    figure_paths: dict[str, str] = field(hash=False, compare=False)
    report_paths: dict[str, str] = field(hash=False, compare=False)
    generated_at: str
    analysis_duration_ms: float
    warnings: tuple[str, ...]
    feature_inventory: FeatureInventory
    feature_engineering_blueprint: FeatureEngineeringBlueprint
    dob_analysis: dict | None
    report_metadata: dict

    def to_dict(self) -> dict:
        """Returns a fully JSON-serialisable dictionary."""
        return {
            "report_id": self.report_id,
            "dataset_name": self.dataset_name,
            "dataset_hash": self.dataset_hash,
            "overview": self.overview.to_dict(),
            "target_analysis": self.target_analysis.to_dict(),
            "numerical_analysis": (
                self.numerical_analysis.to_dict()
                if self.numerical_analysis is not None
                else None
            ),
            "categorical_analysis": (
                self.categorical_analysis.to_dict()
                if self.categorical_analysis is not None
                else None
            ),
            "temporal_analysis": (
                self.temporal_analysis.to_dict()
                if self.temporal_analysis is not None
                else None
            ),
            "geographic_analysis": (
                self.geographic_analysis.to_dict()
                if self.geographic_analysis is not None
                else None
            ),
            "correlation_analysis": (
                self.correlation_analysis.to_dict()
                if self.correlation_analysis is not None
                else None
            ),
            "recommendations": [r.to_dict() for r in self.recommendations],
            "figure_paths": self.figure_paths,
            "report_paths": self.report_paths,
            "generated_at": self.generated_at,
            "analysis_duration_ms": round(self.analysis_duration_ms, 2),
            "warnings": list(self.warnings),
            "feature_inventory": self.feature_inventory.to_dict(),
            "feature_engineering_blueprint": self.feature_engineering_blueprint.to_dict(),
            "dob_analysis": self.dob_analysis,
            "report_metadata": self.report_metadata,
        }

    def __str__(self) -> str:
        return (
            f"EDAReport("
            f"id={self.report_id[:8]}..., "
            f"dataset={self.dataset_name!r}, "
            f"analyses={sum(1 for a in [self.numerical_analysis, self.categorical_analysis, self.temporal_analysis, self.geographic_analysis, self.correlation_analysis] if a is not None) + 2}, "
            f"recommendations={len(self.recommendations)}, "
            f"figures={len(self.figure_paths)}, "
            f"duration={self.analysis_duration_ms:.0f}ms)"
        )
