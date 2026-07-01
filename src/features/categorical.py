"""
Categorical annotation module for the Feature Engineering Layer.

Does NOT create any new DataFrame columns.
Produces only an encoding recommendation manifest consumed by the
Preprocessing Layer.
"""

from __future__ import annotations

import logging
import time

import pandas as pd

from src.features.models import EncodingRecommendation

logger = logging.getLogger(__name__)

_STRATEGY_ONE_HOT = "ONE_HOT"
_STRATEGY_FREQUENCY = "FREQUENCY"
_STRATEGY_TARGET_FUTURE = "TARGET_ENCODING_FUTURE"

# ---------------------------------------------------------------------------
# Public module function
# ---------------------------------------------------------------------------


def run_categorical_annotation(
    df: pd.DataFrame,
    categorical_columns: tuple[str, ...],
    max_one_hot_cardinality: int,
) -> dict[str, EncodingRecommendation]:
    """
    Produces encoding strategy recommendations for categorical columns.

    No new columns are added to the DataFrame. The output is consumed by the
    Preprocessing Layer to select encoding strategies without reimplementing
    cardinality detection.

    Strategy selection logic:
        - unique_count <= max_one_hot_cardinality  → ONE_HOT
        - unique_count > max_one_hot_cardinality   → FREQUENCY
        - Future: target-correlated high-cardinality → TARGET_ENCODING_FUTURE

    Args:
        df: Input DataFrame. Used to compute unique value counts.
        categorical_columns: Columns to annotate. Should come from
            ``EDAReport.feature_inventory.categorical``.
        max_one_hot_cardinality: Max unique values for ONE_HOT recommendation.
            Read from ``config.settings.fe.max_one_hot_cardinality``.

    Returns:
        dict[str, EncodingRecommendation]: Keyed by column name.
    """
    _start = time.perf_counter()
    logger.info(
        "Categorical annotation module started | columns=%s | max_one_hot=%d",
        list(categorical_columns),
        max_one_hot_cardinality,
    )

    recommendations: dict[str, EncodingRecommendation] = {}

    for col in categorical_columns:
        if col not in df.columns:
            logger.warning(
                "Categorical annotation: column '%s' not found in DataFrame — skipped.", col
            )
            continue

        unique_count = int(df[col].nunique(dropna=True))

        if unique_count <= max_one_hot_cardinality:
            strategy = _STRATEGY_ONE_HOT
            note = (
                f"Low cardinality ({unique_count} unique values ≤ threshold {max_one_hot_cardinality}). "
                f"Suitable for one-hot encoding."
            )
        else:
            strategy = _STRATEGY_FREQUENCY
            note = (
                f"High cardinality ({unique_count} unique values > threshold {max_one_hot_cardinality}). "
                f"Frequency encoding recommended. "
                f"Target encoding is a future extension (TARGET_ENCODING_FUTURE)."
            )

        recommendations[col] = EncodingRecommendation(
            column=col,
            unique_count=unique_count,
            recommended_strategy=strategy,
            note=note,
        )
        logger.debug(
            "Categorical annotation: %s | unique=%d | strategy=%s",
            col,
            unique_count,
            strategy,
        )

    duration_ms = (time.perf_counter() - _start) * 1_000
    logger.info(
        "Categorical annotation module complete | annotated=%d | duration=%.1fms",
        len(recommendations),
        duration_ms,
    )
    return recommendations
