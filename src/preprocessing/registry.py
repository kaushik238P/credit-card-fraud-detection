"""
Preprocessing step registry for the Preprocessing Layer.

Defines the ordered sequence of preprocessing steps. The pipeline iterates
over PREPROCESSING_REGISTRY entirely — no hardcoded execution order, no
if/elif chains in pipeline.py.

Adding a new step requires only a new PreprocessingStepEntry here.
pipeline.py requires no changes (Open/Closed Principle).
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Valid step types
# ---------------------------------------------------------------------------

STEP_TYPE_DROP = "DROP"
STEP_TYPE_SPLIT = "SPLIT"
STEP_TYPE_ENCODER = "ENCODER"
STEP_TYPE_SCALER = "SCALER"
STEP_TYPE_ARTIFACT = "ARTIFACT"

# ---------------------------------------------------------------------------
# Valid column sources
# ---------------------------------------------------------------------------

# Columns resolved from EngineeredDataset.report.encoding_recommendations
# where recommended_strategy == "FREQUENCY"
COL_SOURCE_FREQUENCY_RECS = "FREQUENCY_RECOMMENDATIONS"

# Columns resolved from encoding_recommendations where strategy == "ONE_HOT"
COL_SOURCE_OHE_RECS = "OHE_RECOMMENDATIONS"

# Numeric columns from blueprint.scale_features (filtered to present cols)
COL_SOURCE_SCALE_FEATURES = "SCALE_FEATURES"

# Resolved inside the pipeline directly (drop, split) — no column list needed
COL_SOURCE_STATIC = "STATIC"


# ---------------------------------------------------------------------------
# PreprocessingStepEntry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreprocessingStepEntry:
    """
    Descriptor for a single step in the preprocessing pipeline.

    Attributes:
        step_name: Human-readable step identifier (e.g. "FREQUENCY_ENCODING").
        step_type: Execution category — DROP, SPLIT, ENCODER, SCALER, ARTIFACT.
        enabled: Whether this step is executed. Controlled per-step here;
            pipeline reads this flag before executing each step.
        columns_source: How to resolve the column list for this step.
            One of the COL_SOURCE_* constants above.
    """

    step_name: str
    step_type: str
    enabled: bool
    columns_source: str


# ---------------------------------------------------------------------------
# PREPROCESSING_REGISTRY — ordered execution sequence
# ---------------------------------------------------------------------------

PREPROCESSING_REGISTRY: tuple[PreprocessingStepEntry, ...] = (
    PreprocessingStepEntry(
        step_name="DROP_SOURCE_COLUMNS",
        step_type=STEP_TYPE_DROP,
        enabled=True,
        columns_source=COL_SOURCE_STATIC,
    ),
    PreprocessingStepEntry(
        step_name="TIME_BASED_SPLIT",
        step_type=STEP_TYPE_SPLIT,
        enabled=True,
        columns_source=COL_SOURCE_STATIC,
    ),
    PreprocessingStepEntry(
        step_name="FREQUENCY_ENCODING",
        step_type=STEP_TYPE_ENCODER,
        enabled=True,
        columns_source=COL_SOURCE_FREQUENCY_RECS,
    ),
    PreprocessingStepEntry(
        step_name="ONE_HOT_ENCODING",
        step_type=STEP_TYPE_ENCODER,
        enabled=True,
        columns_source=COL_SOURCE_OHE_RECS,
    ),
    PreprocessingStepEntry(
        step_name="SCALING",
        step_type=STEP_TYPE_SCALER,
        enabled=True,  # Pipeline swaps RobustScaler ↔ IdentityTransformer via config
        columns_source=COL_SOURCE_SCALE_FEATURES,
    ),
    PreprocessingStepEntry(
        step_name="SAVE_ARTIFACTS",
        step_type=STEP_TYPE_ARTIFACT,
        enabled=True,
        columns_source=COL_SOURCE_STATIC,
    ),
)
