"""
Model registry for the Training Layer.

Defines the ordered sequence of supported models. The pipeline resolves
the active model entirely from MODEL_REGISTRY — no if/elif chains anywhere.

Adding a new model requires only:
    1. Add a ModelType enum value in models.py
    2. Add a ModelRegistryEntry here
    3. Add creation logic to ModelFactory
    Pipeline code requires zero changes (Open/Closed Principle).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.training.models import ModelType


# ---------------------------------------------------------------------------
# ModelRegistryEntry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelRegistryEntry:
    """
    Descriptor for a single supported model type.

    Attributes:
        model_type: ModelType enum member.
        model_name: Human-readable label used in logs and reports.
        enabled: Whether this model is available for training.
            Controlled here; pipeline reads this before creating any model.
        description: One-line summary for documentation and manifest.
    """

    model_type: ModelType
    model_name: str
    enabled: bool
    description: str


# ---------------------------------------------------------------------------
# MODEL_REGISTRY — ordered tuple of all supported model descriptors
# ---------------------------------------------------------------------------

MODEL_REGISTRY: tuple[ModelRegistryEntry, ...] = (
    ModelRegistryEntry(
        model_type=ModelType.LOGISTIC_REGRESSION,
        model_name="Logistic Regression",
        enabled=True,
        description=(
            "Linear baseline. Supports class_weight for imbalance. "
            "Best used with scaling enabled."
        ),
    ),
    ModelRegistryEntry(
        model_type=ModelType.RANDOM_FOREST,
        model_name="Random Forest",
        enabled=True,
        description=(
            "Ensemble of decision trees. Supports class_weight. "
            "No feature scaling required."
        ),
    ),
    ModelRegistryEntry(
        model_type=ModelType.XGBOOST,
        model_name="XGBoost Classifier",
        enabled=True,
        description=(
            "Gradient boosted trees. Supports scale_pos_weight. "
            "Production default. Fastest on tabular data."
        ),
    ),
    ModelRegistryEntry(
        model_type=ModelType.LIGHTGBM,
        model_name="LightGBM Classifier",
        enabled=True,
        description=(
            "Leaf-wise gradient boosting. Supports class_weight / "
            "scale_pos_weight. Fastest training on large datasets."
        ),
    ),
    ModelRegistryEntry(
        model_type=ModelType.CATBOOST,
        model_name="CatBoost Classifier",
        enabled=True,
        description=(
            "Gradient boosting with native categorical support. "
            "Supports class_weights. Robust to overfitting."
        ),
    ),
)

# ---------------------------------------------------------------------------
# Registry lookup helpers
# ---------------------------------------------------------------------------


def get_entry(model_type: ModelType) -> ModelRegistryEntry:
    """
    Returns the ModelRegistryEntry for a given ModelType.

    Args:
        model_type: Enum member to look up.

    Returns:
        ModelRegistryEntry: Matching entry.

    Raises:
        KeyError: If model_type is not found in MODEL_REGISTRY.
    """
    for entry in MODEL_REGISTRY:
        if entry.model_type == model_type:
            return entry
    raise KeyError(
        f"ModelType '{model_type.value}' not found in MODEL_REGISTRY. "
        f"Available: {[e.model_type.value for e in MODEL_REGISTRY]}"
    )


def get_enabled_entries() -> tuple[ModelRegistryEntry, ...]:
    """Returns all enabled entries from MODEL_REGISTRY."""
    return tuple(e for e in MODEL_REGISTRY if e.enabled)
