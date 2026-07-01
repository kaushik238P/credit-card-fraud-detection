"""
Geographic feature module for the Feature Engineering Layer.

Generates distance-based features using vectorised Haversine computation.
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

_REQUIRED_COLUMNS = ("lat", "long", "merch_lat", "merch_long")
_EARTH_RADIUS_KM = 6371.0088

# ---------------------------------------------------------------------------
# Feature definitions (lineage registry)
# ---------------------------------------------------------------------------

FEATURE_DEFINITIONS: tuple[FeatureDefinition, ...] = (
    FeatureDefinition(
        name="customer_merchant_distance_km",
        group="GEOGRAPHIC",
        source_columns=("lat", "long", "merch_lat", "merch_long"),
        description=(
            "Haversine distance in kilometres between the customer's home "
            "coordinates (lat, long) and the merchant's coordinates "
            "(merch_lat, merch_long). Vectorised NumPy computation."
        ),
        leakage_risk=False,
    ),
    FeatureDefinition(
        name="is_far_transaction",
        group="GEOGRAPHIC",
        source_columns=("lat", "long", "merch_lat", "merch_long"),
        description=(
            "Binary flag: 1 if customer_merchant_distance_km exceeds the "
            "configured far-transaction threshold "
            "(FRAUD_FE_FAR_TRANSACTION_THRESHOLD_KM), else 0."
        ),
        leakage_risk=False,
    ),
)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _haversine_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """
    Vectorised Haversine distance computation.

    Args:
        lat1: Customer latitude array (degrees).
        lon1: Customer longitude array (degrees).
        lat2: Merchant latitude array (degrees).
        lon2: Merchant longitude array (degrees).

    Returns:
        np.ndarray: Distance in kilometres, same length as inputs.
    """
    lat1_r = np.radians(lat1)
    lat2_r = np.radians(lat2)
    d_lat = np.radians(lat2 - lat1)
    d_lon = np.radians(lon2 - lon1)

    a = (
        np.sin(d_lat / 2.0) ** 2
        + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(d_lon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return _EARTH_RADIUS_KM * c


# ---------------------------------------------------------------------------
# Public module function
# ---------------------------------------------------------------------------


def run_geographic(
    df: pd.DataFrame,
    far_transaction_threshold_km: float,
) -> pd.DataFrame:
    """
    Computes geographic features using Haversine distance.

    Args:
        df: Input DataFrame containing ``lat``, ``long``,
            ``merch_lat``, ``merch_long``.
        far_transaction_threshold_km: Distance threshold above which a
            transaction is flagged as far. Read from
            ``config.settings.fe.far_transaction_threshold_km``.

    Returns:
        pd.DataFrame: New DataFrame with ``customer_merchant_distance_km``
            and ``is_far_transaction``. Same row count as ``df``.

    Raises:
        TransformationError: If any required coordinate column is absent.
    """
    _start = time.perf_counter()
    logger.info(
        "Geographic module started | rows=%d | far_threshold=%.1f km",
        len(df),
        far_transaction_threshold_km,
    )

    missing = [col for col in _REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise TransformationError(
            feature_name="geographic_features",
            source_columns=list(_REQUIRED_COLUMNS),
            detail=f"Required coordinate column(s) missing: {missing}.",
        )

    try:
        lat1 = df["lat"].to_numpy(dtype=float, na_value=np.nan)
        lon1 = df["long"].to_numpy(dtype=float, na_value=np.nan)
        lat2 = df["merch_lat"].to_numpy(dtype=float, na_value=np.nan)
        lon2 = df["merch_long"].to_numpy(dtype=float, na_value=np.nan)

        distance_km = _haversine_km(lat1, lon1, lat2, lon2)
        is_far = (distance_km >= far_transaction_threshold_km).astype(np.int8)
    except Exception as exc:
        raise TransformationError(
            feature_name="geographic_features",
            source_columns=list(_REQUIRED_COLUMNS),
            detail="Haversine computation failed.",
            original_exc=exc,
        ) from exc

    result = pd.DataFrame(
        {
            "customer_merchant_distance_km": distance_km,
            "is_far_transaction": is_far,
        },
        index=df.index,
    )

    duration_ms = (time.perf_counter() - _start) * 1_000
    logger.info(
        "Geographic module complete | features=%d | duration=%.1fms",
        len(result.columns),
        duration_ms,
    )
    return result
