"""
Feature registry for the Feature Engineering Layer.

Defines the ordered sequence of feature modules. The pipeline iterates over
the registry rather than calling modules directly, following the Open/Closed
Principle — new modules are added here without modifying pipeline.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from src.features import (
    temporal,
    amount,
    demographic,
    geographic,
    frequency,
    categorical,
    behavioral,
)
from src.features.models import FeatureDefinition


# ---------------------------------------------------------------------------
# FeatureModuleEntry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureModuleEntry:
    """
    Descriptor for a single feature module in the registry.

    Attributes:
        group_name: Identifier for this feature group (e.g. 'TEMPORAL').
        feature_definitions: FeatureDefinition records for lineage tracking.
        enabled_by_default: Whether this module runs by default.
        leakage_aware: True if this module has leakage risk documentation.
        requires_eda_report: True if the module consumes EDAReport fields.
    """

    group_name: str
    feature_definitions: tuple[FeatureDefinition, ...]
    enabled_by_default: bool = True
    leakage_aware: bool = False
    requires_eda_report: bool = False


# ---------------------------------------------------------------------------
# Registry constant — ordered execution sequence
# ---------------------------------------------------------------------------

FEATURE_REGISTRY: tuple[FeatureModuleEntry, ...] = (
    FeatureModuleEntry(
        group_name="TEMPORAL",
        feature_definitions=temporal.FEATURE_DEFINITIONS,
        enabled_by_default=True,
        leakage_aware=False,
        requires_eda_report=False,
    ),
    FeatureModuleEntry(
        group_name="AMOUNT",
        feature_definitions=amount.FEATURE_DEFINITIONS,
        enabled_by_default=True,
        leakage_aware=False,
        requires_eda_report=False,
    ),
    FeatureModuleEntry(
        group_name="DEMOGRAPHIC",
        feature_definitions=demographic.FEATURE_DEFINITIONS,
        enabled_by_default=True,
        leakage_aware=False,
        requires_eda_report=False,
    ),
    FeatureModuleEntry(
        group_name="GEOGRAPHIC",
        feature_definitions=geographic.FEATURE_DEFINITIONS,
        enabled_by_default=True,
        leakage_aware=False,
        requires_eda_report=False,
    ),
    FeatureModuleEntry(
        group_name="FREQUENCY",
        feature_definitions=frequency.FEATURE_DEFINITIONS,
        enabled_by_default=True,
        leakage_aware=False,
        requires_eda_report=False,
    ),
    FeatureModuleEntry(
        group_name="CATEGORICAL",
        feature_definitions=(),  # categorical produces no DataFrame columns
        enabled_by_default=True,
        leakage_aware=False,
        requires_eda_report=True,
    ),
    FeatureModuleEntry(
        group_name="BEHAVIORAL",
        feature_definitions=behavioral.FEATURE_DEFINITIONS,
        enabled_by_default=False,
        leakage_aware=True,
        requires_eda_report=False,
    ),
)
