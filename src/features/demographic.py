"""
Demographic feature module for the Feature Engineering Layer.

Generates age-based features from the date-of-birth column.
Reference date is always the transaction timestamp — never datetime.now().
All features are stateless — no side effects, no I/O.
"""

from __future__ import annotations

import logging
import time

import pandas as pd

from src.features.exceptions import TransformationError
from src.features.models import FeatureDefinition

logger = logging.getLogger(__name__)

_DOB_COLUMN = "dob"
_DATETIME_COLUMN = "trans_date_trans_time"

# ---------------------------------------------------------------------------
# Feature definitions (lineage registry)
# ---------------------------------------------------------------------------

FEATURE_DEFINITIONS: tuple[FeatureDefinition, ...] = (
    FeatureDefinition(
        name="age",
        group="DEMOGRAPHIC",
        source_columns=(_DOB_COLUMN, _DATETIME_COLUMN),
        description=(
            "Customer age in fractional years at time of transaction. "
            "Reference date is trans_date_trans_time, not current datetime. "
            "DOB column should be dropped after this feature is created."
        ),
        leakage_risk=False,
    ),
    FeatureDefinition(
        name="age_group",
        group="DEMOGRAPHIC",
        source_columns=(_DOB_COLUMN, _DATETIME_COLUMN),
        description=(
            "Ordinal integer age group label derived from age. "
            "Bins are configured via FRAUD_FE_AGE_BINS and FRAUD_FE_AGE_BIN_LABELS."
        ),
        leakage_risk=False,
    ),
)


# ---------------------------------------------------------------------------
# Public module function
# ---------------------------------------------------------------------------


def run_demographic(
    df: pd.DataFrame,
    age_bins: tuple[float, ...],
    age_bin_labels: tuple[str, ...],
) -> pd.DataFrame:
    """
    Computes demographic features from the date-of-birth column.

    The reference date for age calculation is ``trans_date_trans_time``,
    not ``datetime.now()``. This ensures reproducibility during batch inference.

    Rows where ``dob`` fails to parse receive NaN for ``age`` and ``age_group``.
    A warning is logged but the pipeline does not raise.

    Args:
        df: Input DataFrame containing ``dob`` and ``trans_date_trans_time``.
        age_bins: Bin edges for age grouping. Must have len(age_bin_labels) + 1
            elements. Read from ``config.settings.fe.age_bins``.
        age_bin_labels: String labels for each age bin. Read from
            ``config.settings.fe.age_bin_labels``.

    Returns:
        pd.DataFrame: New DataFrame with ``age`` and ``age_group`` columns.
            Same row count as ``df``.

    Raises:
        TransformationError: If both ``dob`` and ``trans_date_trans_time``
            are entirely absent, making age computation impossible.
    """
    _start = time.perf_counter()
    logger.info("Demographic module started | rows=%d", len(df))

    if _DOB_COLUMN not in df.columns:
        raise TransformationError(
            feature_name="demographic_features",
            source_columns=[_DOB_COLUMN],
            detail=f"Column '{_DOB_COLUMN}' not found in DataFrame.",
        )
    if _DATETIME_COLUMN not in df.columns:
        raise TransformationError(
            feature_name="demographic_features",
            source_columns=[_DATETIME_COLUMN],
            detail=f"Reference column '{_DATETIME_COLUMN}' not found in DataFrame.",
        )

    dob_parsed = pd.to_datetime(df[_DOB_COLUMN], errors="coerce")
    ref_date = pd.to_datetime(df[_DATETIME_COLUMN], errors="coerce")

    unparseable_count = int(dob_parsed.isna().sum())
    if unparseable_count > 0:
        logger.warning(
            "Demographic module: %d row(s) have unparseable DOB values — "
            "age will be NaN for those rows.",
            unparseable_count,
        )

    if dob_parsed.isna().all():
        raise TransformationError(
            feature_name="demographic_features",
            source_columns=[_DOB_COLUMN],
            detail=f"All values in '{_DOB_COLUMN}' are unparseable.",
        )

    # Compute age in fractional years using the per-row transaction timestamp.
    # If ref_date is NaN for a row, age will also be NaN for that row.
    age_days = (ref_date - dob_parsed).dt.days
    age = age_days / 365.25

    # pd.cut produces categorical; convert to ordinal int label (NaN-safe).
    bins_list = list(age_bins)
    labels_list = list(range(len(age_bin_labels)))  # 0, 1, 2, ... ordinal ints

    age_group_cat = pd.cut(
        age,
        bins=bins_list,
        labels=labels_list,
        right=True,
        include_lowest=True,
    )
    age_group = age_group_cat.astype("float64")  # keeps NaN as NaN

    result = pd.DataFrame(
        {
            "age": age.values,
            "age_group": age_group.values,
        },
        index=df.index,
    )

    duration_ms = (time.perf_counter() - _start) * 1_000
    logger.info(
        "Demographic module complete | features=%d | nan_dob=%d | duration=%.1fms",
        len(result.columns),
        unparseable_count,
        duration_ms,
    )
    return result
