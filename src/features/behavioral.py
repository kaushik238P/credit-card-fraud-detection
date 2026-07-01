"""
Behavioral feature module for the Feature Engineering Layer.

PHASE 1: This module is DISABLED by default.

Behavioral features require:
    - Time-based train/test splitting before computation.
    - A historical reference DataFrame (train fold only) for test/inference.
    - Strict leakage prevention to avoid card-level future data contaminating
      past model inputs.

When disabled (default), this module returns an empty DataFrame and appends
an informational warning to the pipeline report.

When enabled (FRAUD_FE_ENABLE_BEHAVIORAL=1), behavioral aggregates are
computed from the provided historical_reference_df. If historical_reference_df
is None, they are computed from df itself with an additional leakage warning.
"""

from __future__ import annotations

import logging
import time

import pandas as pd

from src.features.exceptions import TransformationError
from src.features.models import FeatureDefinition

logger = logging.getLogger(__name__)

_CC_NUM_COLUMN = "cc_num"
_AMT_COLUMN = "amt"
_MERCHANT_COLUMN = "merchant"
_CATEGORY_COLUMN = "category"

# ---------------------------------------------------------------------------
# Feature definitions (lineage registry)
# ---------------------------------------------------------------------------

FEATURE_DEFINITIONS: tuple[FeatureDefinition, ...] = (
    FeatureDefinition(
        name="transactions_per_customer",
        group="BEHAVIORAL",
        source_columns=(_CC_NUM_COLUMN,),
        description=(
            "Total number of transactions associated with the card number "
            "in the historical reference dataset."
        ),
        leakage_risk=True,
        leakage_note=(
            "Must be computed using the training fold as historical reference. "
            "Computing on the full dataset before splitting leaks future "
            "transaction counts into training features."
        ),
    ),
    FeatureDefinition(
        name="average_customer_amount",
        group="BEHAVIORAL",
        source_columns=(_CC_NUM_COLUMN, _AMT_COLUMN),
        description=(
            "Mean transaction amount for the card number "
            "in the historical reference dataset."
        ),
        leakage_risk=True,
        leakage_note=(
            "Must be computed using the training fold as historical reference. "
            "Computing on the full dataset before splitting leaks future "
            "amount patterns into training features."
        ),
    ),
    FeatureDefinition(
        name="merchant_diversity",
        group="BEHAVIORAL",
        source_columns=(_CC_NUM_COLUMN, _MERCHANT_COLUMN),
        description=(
            "Count of distinct merchants visited by the card number "
            "in the historical reference dataset."
        ),
        leakage_risk=True,
        leakage_note=(
            "Must be computed using the training fold as historical reference. "
            "Future merchant visits must not contribute to training-time counts."
        ),
    ),
    FeatureDefinition(
        name="category_diversity",
        group="BEHAVIORAL",
        source_columns=(_CC_NUM_COLUMN, _CATEGORY_COLUMN),
        description=(
            "Count of distinct transaction categories used by the card number "
            "in the historical reference dataset."
        ),
        leakage_risk=True,
        leakage_note=(
            "Must be computed using the training fold as historical reference. "
            "Future category usage must not contribute to training-time counts."
        ),
    ),
)

# ---------------------------------------------------------------------------
# Future extensions (documented, not implemented)
# ---------------------------------------------------------------------------

_FUTURE_FEATURE_DEFINITIONS: tuple[FeatureDefinition, ...] = (
    FeatureDefinition(
        name="transaction_velocity",
        group="BEHAVIORAL_FUTURE",
        source_columns=(_CC_NUM_COLUMN, "trans_date_trans_time"),
        description="Transaction count for a card in a rolling time window.",
        leakage_risk=True,
        leakage_note=(
            "Requires strictly ordered time-based windows. Window boundaries "
            "must be constrained to t < t_current to prevent future leakage. "
            "NOT IMPLEMENTED in Phase 1."
        ),
    ),
    FeatureDefinition(
        name="time_since_previous_transaction",
        group="BEHAVIORAL_FUTURE",
        source_columns=(_CC_NUM_COLUMN, "trans_date_trans_time"),
        description="Seconds since the previous transaction for the same card.",
        leakage_risk=True,
        leakage_note=(
            "Requires time-ordered card history. At inference time, the "
            "previous transaction timestamp must come from a pre-computed "
            "card history store, not from the current batch."
            "NOT IMPLEMENTED in Phase 1."
        ),
    ),
    FeatureDefinition(
        name="merchant_risk",
        group="BEHAVIORAL_FUTURE",
        source_columns=(_MERCHANT_COLUMN, "is_fraud"),
        description="Mean fraud rate per merchant in the training fold.",
        leakage_risk=True,
        leakage_note=(
            "Extreme leakage risk: uses the target column is_fraud. "
            "Must NEVER be computed before train/test split. "
            "Implement using sklearn TargetEncoder inside the Preprocessing Layer. "
            "NOT IMPLEMENTED in Feature Engineering."
        ),
    ),
    FeatureDefinition(
        name="category_risk",
        group="BEHAVIORAL_FUTURE",
        source_columns=(_CATEGORY_COLUMN, "is_fraud"),
        description="Mean fraud rate per category in the training fold.",
        leakage_risk=True,
        leakage_note=(
            "Extreme leakage risk: uses the target column is_fraud. "
            "Must NEVER be computed before train/test split. "
            "Implement using sklearn TargetEncoder inside the Preprocessing Layer. "
            "NOT IMPLEMENTED in Feature Engineering."
        ),
    ),
    FeatureDefinition(
        name="state_risk",
        group="BEHAVIORAL_FUTURE",
        source_columns=("state", "is_fraud"),
        description="Mean fraud rate per state in the training fold.",
        leakage_risk=True,
        leakage_note=(
            "Extreme leakage risk: uses the target column is_fraud. "
            "Must NEVER be computed before train/test split. "
            "Implement using sklearn TargetEncoder inside the Preprocessing Layer. "
            "NOT IMPLEMENTED in Feature Engineering."
        ),
    ),
)


# ---------------------------------------------------------------------------
# Public module function
# ---------------------------------------------------------------------------


def run_behavioral(
    df: pd.DataFrame,
    enabled: bool,
    historical_reference_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Computes behavioral aggregate features.

    Phase 1: Disabled by default. Returns empty DataFrame when ``enabled``
    is False.

    Args:
        df: Input DataFrame to compute features for.
        enabled: Whether behavioral feature computation is active. Read from
            ``config.settings.fe.enable_behavioral``.
        historical_reference_df: Historical reference DataFrame (training fold).
            If None and enabled, features are computed from ``df`` itself with
            a leakage warning. Must be provided in production to prevent leakage.

    Returns:
        tuple:
            - pd.DataFrame: Behavioral feature columns (empty if disabled).
            - list[str]: Warning messages accumulated during this call.
    """
    _start = time.perf_counter()
    module_warnings: list[str] = []

    if not enabled:
        msg = (
            "Behavioral features are disabled (FRAUD_FE_ENABLE_BEHAVIORAL=0). "
            "Enable after ensuring time-based train/test splitting is in place. "
            "See behavioral.FEATURE_DEFINITIONS for leakage documentation."
        )
        logger.info("Behavioral module: %s", msg)
        module_warnings.append(msg)
        return pd.DataFrame(index=df.index), module_warnings

    logger.info("Behavioral module started | rows=%d", len(df))

    if historical_reference_df is None:
        leakage_msg = (
            "LEAKAGE WARNING: behavioral features computed from df itself because "
            "historical_reference_df is None. This is only valid if df is already "
            "the training fold. Provide historical_reference_df for test/inference."
        )
        logger.warning(leakage_msg)
        module_warnings.append(leakage_msg)
        ref = df
    else:
        ref = historical_reference_df

    required = [_CC_NUM_COLUMN, _AMT_COLUMN, _MERCHANT_COLUMN, _CATEGORY_COLUMN]
    missing_in_ref = [c for c in required if c not in ref.columns]
    if missing_in_ref:
        raise TransformationError(
            feature_name="behavioral_features",
            source_columns=required,
            detail=f"Required column(s) missing from reference DataFrame: {missing_in_ref}.",
        )
    if _CC_NUM_COLUMN not in df.columns:
        raise TransformationError(
            feature_name="behavioral_features",
            source_columns=[_CC_NUM_COLUMN],
            detail=f"Column '{_CC_NUM_COLUMN}' not found in input DataFrame.",
        )

    # Aggregate on historical reference
    tx_counts = ref.groupby(_CC_NUM_COLUMN).size().rename("transactions_per_customer")
    avg_amounts = ref.groupby(_CC_NUM_COLUMN)[_AMT_COLUMN].mean().rename("average_customer_amount")
    merch_div = (
        ref.groupby(_CC_NUM_COLUMN)[_MERCHANT_COLUMN]
        .nunique()
        .rename("merchant_diversity")
    )
    cat_div = (
        ref.groupby(_CC_NUM_COLUMN)[_CATEGORY_COLUMN]
        .nunique()
        .rename("category_diversity")
    )

    agg = pd.concat([tx_counts, avg_amounts, merch_div, cat_div], axis=1)

    result = df[[_CC_NUM_COLUMN]].join(agg, on=_CC_NUM_COLUMN, how="left")
    result = result.drop(columns=[_CC_NUM_COLUMN])
    result = result.fillna(0)
    result[["transactions_per_customer", "merchant_diversity", "category_diversity"]] = (
        result[["transactions_per_customer", "merchant_diversity", "category_diversity"]].astype("int32")
    )

    duration_ms = (time.perf_counter() - _start) * 1_000
    logger.info(
        "Behavioral module complete | features=%d | duration=%.1fms",
        len(result.columns),
        duration_ms,
    )
    return result, module_warnings
