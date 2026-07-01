"""
Immutable data models for the Feature Engineering Layer.

All models are frozen dataclasses. Every model exposes to_dict() returning
a fully JSON-serialisable dict (no pandas, no numpy objects).

Model hierarchy:
    FeatureDefinition         — metadata for a single engineered feature
    FeatureGroup              — a named collection of FeatureDefinitions
    EncodingRecommendation    — encoding strategy recommendation for a column
    FeatureEngineeringReport  — complete output summary of one FE run
    EngineeredDataset         — primary pipeline exchange object
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.ingestion.models import DatasetMetadata


# ---------------------------------------------------------------------------
# FeatureDefinition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureDefinition:
    """
    Metadata record for a single engineered feature.

    Attributes:
        name: The column name of the engineered feature in the output DataFrame.
        group: The feature group this belongs to (TEMPORAL, AMOUNT, etc.).
        source_columns: Raw columns consumed to produce this feature.
        description: Human-readable description of the transformation.
        leakage_risk: True if this feature carries a data leakage risk.
        leakage_note: Explanation of the leakage risk when leakage_risk is True.
    """

    name: str
    group: str
    source_columns: tuple[str, ...]
    description: str
    leakage_risk: bool = False
    leakage_note: str = ""

    def to_dict(self) -> dict:
        """Returns a JSON-serialisable dictionary."""
        return {
            "name": self.name,
            "group": self.group,
            "source_columns": list(self.source_columns),
            "description": self.description,
            "leakage_risk": self.leakage_risk,
            "leakage_note": self.leakage_note,
        }


# ---------------------------------------------------------------------------
# FeatureGroup
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureGroup:
    """
    A named collection of FeatureDefinitions produced by one module.

    Attributes:
        name: Group name, e.g. 'TEMPORAL', 'GEOGRAPHIC'.
        features: All FeatureDefinitions belonging to this group.
        enabled: Whether this group was executed in the current run.
    """

    name: str
    features: tuple[FeatureDefinition, ...]
    enabled: bool = True

    def to_dict(self) -> dict:
        """Returns a JSON-serialisable dictionary."""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "features": [f.to_dict() for f in self.features],
        }


# ---------------------------------------------------------------------------
# EncodingRecommendation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EncodingRecommendation:
    """
    Encoding strategy recommendation for a categorical column.

    Produced by the categorical annotation module. Consumed by the
    Preprocessing Layer to select encoding strategies without reimplementing
    cardinality detection.

    Attributes:
        column: Column name this recommendation applies to.
        unique_count: Number of unique values in the column.
        recommended_strategy: One of 'ONE_HOT', 'FREQUENCY',
            'TARGET_ENCODING_FUTURE'.
        note: Human-readable rationale for the recommendation.
    """

    column: str
    unique_count: int
    recommended_strategy: str
    note: str = ""

    def to_dict(self) -> dict:
        """Returns a JSON-serialisable dictionary."""
        return {
            "column": self.column,
            "unique_count": self.unique_count,
            "recommended_strategy": self.recommended_strategy,
            "note": self.note,
        }


# ---------------------------------------------------------------------------
# FeatureEngineeringReport
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureEngineeringReport:
    """
    Complete, immutable summary of one Feature Engineering run.

    This report is embedded inside EngineeredDataset and is the audit record
    for feature lineage, module execution, warnings, and recommendations.

    Attributes:
        report_id: UUID4 string uniquely identifying this run.
        dataset_name: Filename of the source dataset.
        dataset_hash: SHA-256 hash propagated from EDAReport.
        feature_count_before: Column count of the input DataFrame (pre-FE).
        feature_count_after: Column count of the output DataFrame (post-FE).
        feature_matrix_shape: (rows, cols) of the output DataFrame.
        generated_features: Names of all newly created feature columns.
        dropped_columns: Columns removed from the DataFrame (PII, identifiers).
        feature_definitions: Full FeatureDefinition records for each new column.
        transformation_summary: Mapping of group name → count of features created.
        encoding_recommendations: Mapping of column name → EncodingRecommendation.
        frequency_maps: Serialisable frequency count maps per column.
            Keys are column names; values are {value: count} dicts.
        leakage_warnings: Leakage-specific warnings (subset of warnings).
        warnings: All non-fatal module warnings from this run.
        generated_at: ISO 8601 UTC timestamp.
        duration_ms: Total wall-clock time for the full pipeline run.
    """

    report_id: str
    dataset_name: str
    dataset_hash: str | None
    feature_count_before: int
    feature_count_after: int
    feature_matrix_shape: tuple[int, int]
    generated_features: tuple[str, ...]
    dropped_columns: tuple[str, ...]
    feature_definitions: tuple[FeatureDefinition, ...]
    transformation_summary: dict[str, int] = field(hash=False, compare=False)
    encoding_recommendations: dict[str, EncodingRecommendation] = field(
        hash=False, compare=False
    )
    frequency_maps: dict[str, dict[str, int]] = field(hash=False, compare=False)
    leakage_warnings: tuple[str, ...]
    warnings: tuple[str, ...]
    generated_at: str
    duration_ms: float

    def to_dict(self) -> dict:
        """Returns a fully JSON-serialisable dictionary."""
        return {
            "report_id": self.report_id,
            "dataset_name": self.dataset_name,
            "dataset_hash": self.dataset_hash,
            "feature_count_before": self.feature_count_before,
            "feature_count_after": self.feature_count_after,
            "feature_matrix_shape": list(self.feature_matrix_shape),
            "generated_features": list(self.generated_features),
            "dropped_columns": list(self.dropped_columns),
            "feature_definitions": [fd.to_dict() for fd in self.feature_definitions],
            "transformation_summary": self.transformation_summary,
            "encoding_recommendations": {
                col: rec.to_dict()
                for col, rec in self.encoding_recommendations.items()
            },
            "frequency_maps": self.frequency_maps,
            "leakage_warnings": list(self.leakage_warnings),
            "warnings": list(self.warnings),
            "generated_at": self.generated_at,
            "duration_ms": round(self.duration_ms, 2),
        }

    def __str__(self) -> str:
        return (
            f"FeatureEngineeringReport("
            f"id={self.report_id[:8]}..., "
            f"dataset={self.dataset_name!r}, "
            f"generated={len(self.generated_features)}, "
            f"dropped={len(self.dropped_columns)}, "
            f"warnings={len(self.warnings)}, "
            f"duration={self.duration_ms:.0f}ms)"
        )


# ---------------------------------------------------------------------------
# EngineeredDataset
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EngineeredDataset:
    """
    Primary pipeline exchange object returned by FeatureEngineeringPipeline.

    Bundles the enriched DataFrame with full lineage metadata. This is the
    only object passed from the Feature Engineering Layer to the Preprocessing
    Layer.

    Design notes:
        - ``data`` carries ``compare=False, hash=False, repr=False`` because
          pandas DataFrames do not support reliable equality semantics.
        - ``frozen=True`` prevents field reassignment but does NOT prevent
          in-place mutation of the DataFrame — downstream consumers must work
          on explicit copies.
        - The original DataFrame from ValidatedDataset is never modified.
          ``data`` is always a fresh copy with new columns appended.

    Attributes:
        data: Enriched DataFrame. Contains all original columns (minus
            dropped ones) plus all engineered feature columns.
        original_columns: Column names from the ValidatedDataset input.
        engineered_columns: Names of newly created feature columns.
        dropped_columns: Columns removed during feature engineering.
        metadata: DatasetMetadata propagated from ValidatedDataset.
        report: Complete FeatureEngineeringReport for this run.
    """

    data: pd.DataFrame = field(repr=False, compare=False, hash=False)
    original_columns: tuple[str, ...]
    engineered_columns: tuple[str, ...]
    dropped_columns: tuple[str, ...]
    metadata: DatasetMetadata
    report: FeatureEngineeringReport

    def to_dict(self) -> dict:
        """
        Returns a JSON-serialisable dictionary excluding the DataFrame.

        The DataFrame is excluded because it is not JSON-serialisable.
        Use ``data`` directly for downstream processing.

        Returns:
            dict: metadata + report serialised, plus shape and column lists.
        """
        return {
            "original_columns": list(self.original_columns),
            "engineered_columns": list(self.engineered_columns),
            "dropped_columns": list(self.dropped_columns),
            "metadata": self.metadata.to_dict(),
            "report": self.report.to_dict(),
        }

    def __repr__(self) -> str:
        return (
            f"EngineeredDataset("
            f"dataset={self.metadata.file_name!r}, "
            f"shape={self.data.shape}, "
            f"original={len(self.original_columns)}, "
            f"engineered={len(self.engineered_columns)}, "
            f"dropped={len(self.dropped_columns)})"
        )

    def __str__(self) -> str:
        return repr(self)
