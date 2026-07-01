"""
Temporal feature module for the Feature Engineering Layer.

Generates time-based features from trans_date_trans_time.
All features are stateless — no side effects, no I/O.
"""

from __future__ import annotations

import logging
import time

import pandas as pd

from src.features.exceptions import TransformationError
from src.features.models import FeatureDefinition

logger = logging.getLogger(__name__)

_SOURCE_COLUMN = "trans_date_trans_time"

# ---------------------------------------------------------------------------
# Feature definitions (lineage registry)
# ---------------------------------------------------------------------------

FEATURE_DEFINITIONS: tuple[FeatureDefinition, ...] = (
    FeatureDefinition(
        name="hour",
        group="TEMPORAL",
        source_columns=(_SOURCE_COLUMN,),
        description="Hour of transaction (0–23).",
        leakage_risk=False,
    ),
    FeatureDefinition(
        name="day_of_week",
        group="TEMPORAL",
        source_columns=(_SOURCE_COLUMN,),
        description="Day of week as integer (0=Monday, 6=Sunday).",
        leakage_risk=False,
    ),
    FeatureDefinition(
        name="month",
        group="TEMPORAL",
        source_columns=(_SOURCE_COLUMN,),
        description="Month of year (1–12).",
        leakage_risk=False,
    ),
    FeatureDefinition(
        name="is_weekend",
        group="TEMPORAL",
        source_columns=(_SOURCE_COLUMN,),
        description="1 if transaction occurred on Saturday or Sunday, else 0.",
        leakage_risk=False,
    ),
    FeatureDefinition(
        name="is_night_transaction",
        group="TEMPORAL",
        source_columns=(_SOURCE_COLUMN,),
        description="1 if transaction occurred between 22:00 and 05:59, else 0.",
        leakage_risk=False,
    ),
    FeatureDefinition(
        name="business_hours",
        group="TEMPORAL",
        source_columns=(_SOURCE_COLUMN,),
        description="1 if transaction occurred between 09:00–17:59 on a weekday, else 0.",
        leakage_risk=False,
    ),
)


# ---------------------------------------------------------------------------
# Public module function
# ---------------------------------------------------------------------------


def run_temporal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes temporal features from the transaction datetime column.

    Args:
        df: Input DataFrame containing ``trans_date_trans_time``.

    Returns:
        pd.DataFrame: New DataFrame containing only the 6 temporal feature
            columns. Same row count as ``df``.

    Raises:
        TransformationError: If ``trans_date_trans_time`` is absent or entirely
            unparseable.
    """
    _start = time.perf_counter()
    logger.info("Temporal module started | rows=%d", len(df))

    if _SOURCE_COLUMN not in df.columns:
        raise TransformationError(
            feature_name="temporal_features",
            source_columns=[_SOURCE_COLUMN],
            detail=f"Column '{_SOURCE_COLUMN}' not found in DataFrame.",
        )

    try:
        dt = pd.to_datetime(df[_SOURCE_COLUMN], errors="coerce")
    except Exception as exc:
        raise TransformationError(
            feature_name="temporal_features",
            source_columns=[_SOURCE_COLUMN],
            detail=f"Failed to parse '{_SOURCE_COLUMN}' as datetime.",
            original_exc=exc,
        ) from exc

    if dt.isna().all():
        raise TransformationError(
            feature_name="temporal_features",
            source_columns=[_SOURCE_COLUMN],
            detail=f"All values in '{_SOURCE_COLUMN}' are unparseable.",
        )

    hour = dt.dt.hour
    day_of_week = dt.dt.dayofweek
    month = dt.dt.month
    is_weekend = (day_of_week >= 5).astype("int8")
    is_night = ((hour < 6) | (hour >= 22)).astype("int8")
    business_hours = (
        ((hour >= 9) & (hour < 18)) & (day_of_week < 5)
    ).astype("int8")

    result = pd.DataFrame(
        {
            "hour": hour.values,
            "day_of_week": day_of_week.values,
            "month": month.values,
            "is_weekend": is_weekend.values,
            "is_night_transaction": is_night.values,
            "business_hours": business_hours.values,
        },
        index=df.index,
    )

    duration_ms = (time.perf_counter() - _start) * 1_000
    logger.info(
        "Temporal module complete | features=%d | duration=%.1fms",
        len(result.columns),
        duration_ms,
    )
    return result
