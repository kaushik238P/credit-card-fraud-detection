"""
Frequency feature module for the Feature Engineering Layer.

Computes raw occurrence frequency for high-cardinality categorical columns.
No target information is used — zero leakage risk.
All features are stateless — no side effects, no I/O.
"""

from __future__ import annotations

import logging
import time

import pandas as pd

from src.features.exceptions import TransformationError
from src.features.models import FeatureDefinition

logger = logging.getLogger(__name__)

_SUPPORTED_COLUMNS = (
    "merchant",
    "category",
    "job",
    "state",
    "city",
    "gender",
)

# ---------------------------------------------------------------------------
# Feature definitions (lineage registry)
# ---------------------------------------------------------------------------

FEATURE_DEFINITIONS: tuple[FeatureDefinition, ...] = tuple(
    FeatureDefinition(
        name=f"{col}_frequency",
        group="FREQUENCY",
        source_columns=(col,),
        description=(
            f"Occurrence count of each unique value of '{col}' in the dataset. "
            f"Must be computed on the training fold only; unknown values at "
            f"inference receive 0."
        ),
        leakage_risk=False,
        leakage_note=(
            "Frequency maps must be fitted on the training fold only and "
            "serialised for reuse at inference. Computing on the full dataset "
            "before splitting does not leak target information but does leak "
            "test-set distribution. Fit on train only."
        ),
    )
    for col in _SUPPORTED_COLUMNS
)


# ---------------------------------------------------------------------------
# Public module function
# ---------------------------------------------------------------------------


def run_frequency(
    df: pd.DataFrame,
    columns: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """
    Computes frequency-based features for the specified categorical columns.

    For each column, maps each value to its occurrence count in ``df``.
    Unknown values at inference time should receive 0 (handled by the
    Preprocessing Layer using the serialised frequency maps).

    Args:
        df: Input DataFrame. Must contain all columns listed in ``columns``.
        columns: Categorical column names to compute frequency for. Read from
            ``config.settings.fe.frequency_columns``.

    Returns:
        tuple:
            - pd.DataFrame: New DataFrame with one frequency column per input
              column (named ``{col}_frequency``). Same row count as ``df``.
            - dict[str, dict[str, int]]: Frequency maps keyed by column name.
              Each map is ``{value: count}`` and is JSON-serialisable.

    Raises:
        TransformationError: If no valid columns are found in ``df``.
    """
    _start = time.perf_counter()
    logger.info(
        "Frequency module started | rows=%d | requested_columns=%s",
        len(df),
        list(columns),
    )

    present_cols = [col for col in columns if col in df.columns]
    skipped_cols = [col for col in columns if col not in df.columns]

    if skipped_cols:
        logger.warning(
            "Frequency module: column(s) %s not found in DataFrame — skipped.",
            skipped_cols,
        )

    if not present_cols:
        raise TransformationError(
            feature_name="frequency_features",
            source_columns=list(columns),
            detail="None of the requested frequency columns are present in the DataFrame.",
        )

    frequency_maps: dict[str, dict[str, int]] = {}
    new_columns: dict[str, pd.Series] = {}

    for col in present_cols:
        try:
            counts = df[col].value_counts(dropna=False)
            freq_map: dict[str, int] = {
                str(k): int(v) for k, v in counts.items()
            }
            frequency_maps[col] = freq_map

            mapped = df[col].map(freq_map).fillna(0).astype("int32")
            new_columns[f"{col}_frequency"] = mapped.values
            logger.debug("Frequency feature created | column=%s | unique_values=%d", col, len(freq_map))
        except Exception as exc:
            logger.warning(
                "Frequency module: failed to compute frequency for '%s': %s — skipping.",
                col,
                exc,
            )

    if not new_columns:
        raise TransformationError(
            feature_name="frequency_features",
            source_columns=present_cols,
            detail="All frequency computations failed.",
        )

    result = pd.DataFrame(new_columns, index=df.index)

    duration_ms = (time.perf_counter() - _start) * 1_000
    logger.info(
        "Frequency module complete | features=%d | maps=%d | duration=%.1fms",
        len(result.columns),
        len(frequency_maps),
        duration_ms,
    )
    return result, frequency_maps
