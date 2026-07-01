"""
Amount feature module for the Feature Engineering Layer.

Generates amount-based features from the transaction amount column.
All features are stateless — no side effects, no I/O.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from src.features.exceptions import TransformationError
from src.features.models import FeatureDefinition

logger = logging.getLogger(__name__)

_SOURCE_COLUMN = "amt"

# ---------------------------------------------------------------------------
# Feature definitions (lineage registry)
# ---------------------------------------------------------------------------

FEATURE_DEFINITIONS: tuple[FeatureDefinition, ...] = (
    FeatureDefinition(
        name="log_amount",
        group="AMOUNT",
        source_columns=(_SOURCE_COLUMN,),
        description=(
            "Natural log of (1 + amt). Compresses the heavy right-skewed "
            "distribution of transaction amounts identified in EDA."
        ),
        leakage_risk=False,
    ),
    FeatureDefinition(
        name="high_value_transaction",
        group="AMOUNT",
        source_columns=(_SOURCE_COLUMN,),
        description=(
            "Binary flag: 1 if amt exceeds the configured high-value threshold "
            "(FRAUD_FE_HIGH_VALUE_THRESHOLD), else 0."
        ),
        leakage_risk=False,
    ),
)


# ---------------------------------------------------------------------------
# Public module function
# ---------------------------------------------------------------------------


def run_amount(df: pd.DataFrame, high_value_threshold: float) -> pd.DataFrame:
    """
    Computes amount-based features from the transaction amount column.

    Args:
        df: Input DataFrame containing ``amt``.
        high_value_threshold: Amount above which a transaction is flagged as
            high-value. Read from ``config.settings.fe.high_value_threshold``.

    Returns:
        pd.DataFrame: New DataFrame with ``log_amount`` and
            ``high_value_transaction`` columns. Same row count as ``df``.

    Raises:
        TransformationError: If ``amt`` is absent or contains invalid values
            that prevent computation.
    """
    _start = time.perf_counter()
    logger.info("Amount module started | rows=%d | threshold=%.2f", len(df), high_value_threshold)

    if _SOURCE_COLUMN not in df.columns:
        raise TransformationError(
            feature_name="amount_features",
            source_columns=[_SOURCE_COLUMN],
            detail=f"Column '{_SOURCE_COLUMN}' not found in DataFrame.",
        )

    amt = df[_SOURCE_COLUMN]

    if amt.isna().all():
        raise TransformationError(
            feature_name="amount_features",
            source_columns=[_SOURCE_COLUMN],
            detail=f"Column '{_SOURCE_COLUMN}' contains only NaN values.",
        )

    try:
        log_amount = np.log1p(amt.fillna(0).clip(lower=0))
        high_value = (amt >= high_value_threshold).astype("int8")
    except Exception as exc:
        raise TransformationError(
            feature_name="amount_features",
            source_columns=[_SOURCE_COLUMN],
            detail="Computation of log_amount or high_value_transaction failed.",
            original_exc=exc,
        ) from exc

    result = pd.DataFrame(
        {
            "log_amount": log_amount.values,
            "high_value_transaction": high_value.values,
        },
        index=df.index,
    )

    duration_ms = (time.perf_counter() - _start) * 1_000
    logger.info(
        "Amount module complete | features=%d | duration=%.1fms",
        len(result.columns),
        duration_ms,
    )
    return result
