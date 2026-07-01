"""
Registry descriptors and collections for the Evaluation Layer.

Defines:
    - MetricDescriptor: standard/business metrics execution
    - PlotDescriptor: diagnostic plots execution
    - METRIC_REGISTRY
    - PLOT_REGISTRY
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.evaluation.metrics import (
    compute_balanced_accuracy,
    compute_brier_score,
    compute_classification_report,
    compute_confusion_matrix,
    compute_f1,
    compute_log_loss,
    compute_mcc,
    compute_pr_auc,
    compute_precision,
    compute_recall,
    compute_roc_auc,
    compute_sensitivity,
    compute_specificity,
)
from src.evaluation.plots import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_gain_curve,
    plot_lift_curve,
    plot_precision_recall_curve,
    plot_probability_distribution,
    plot_roc_curve,
)

# ---------------------------------------------------------------------------
# Metric Descriptor & Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricDescriptor:
    """
    Descriptor for a standard evaluation metric.

    Attributes:
        name: Short identifier for the metric.
        fn: Metric function. Can accept (y_true, y_score) if needs_proba=True,
            or (y_true, y_pred) if needs_proba=False.
        needs_proba: True if the metric requires probability scores.
    """

    name: str
    fn: Callable[[np.ndarray, np.ndarray], Any]
    needs_proba: bool = False


METRIC_REGISTRY: tuple[MetricDescriptor, ...] = (
    MetricDescriptor("ROC_AUC", compute_roc_auc, needs_proba=True),
    MetricDescriptor("PR_AUC", compute_pr_auc, needs_proba=True),
    MetricDescriptor("Precision", compute_precision, needs_proba=False),
    MetricDescriptor("Recall", compute_recall, needs_proba=False),
    MetricDescriptor("F1", compute_f1, needs_proba=False),
    MetricDescriptor("Balanced_Accuracy", compute_balanced_accuracy, needs_proba=False),
    MetricDescriptor("MCC", compute_mcc, needs_proba=False),
    MetricDescriptor("Specificity", compute_specificity, needs_proba=False),
    MetricDescriptor("Sensitivity", compute_sensitivity, needs_proba=False),
    MetricDescriptor("Confusion_Matrix", compute_confusion_matrix, needs_proba=False),
    MetricDescriptor("Classification_Report", compute_classification_report, needs_proba=False),
    MetricDescriptor("Log_Loss", compute_log_loss, needs_proba=True),
    MetricDescriptor("Brier_Score", compute_brier_score, needs_proba=True),
)


# ---------------------------------------------------------------------------
# Plot Descriptor & Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlotDescriptor:
    """
    Descriptor for diagnostic plot generation.

    Attributes:
        name: Plot title name.
        file_name: Output plot image filename.
        fn: Execution wrapper. Accepts:
            (y_true, y_pred, y_score, feature_importance, confusion_matrix, save_path)
            and returns the saved path.
    """

    name: str
    file_name: str
    fn: Callable[[np.ndarray, np.ndarray, np.ndarray, dict, dict, str], str]


PLOT_REGISTRY: tuple[PlotDescriptor, ...] = (
    PlotDescriptor(
        name="ROC Curve",
        file_name="roc_curve.png",
        fn=lambda yt, yp, ys, fi, cm, path: plot_roc_curve(yt, ys, path),
    ),
    PlotDescriptor(
        name="Precision-Recall Curve",
        file_name="precision_recall_curve.png",
        fn=lambda yt, yp, ys, fi, cm, path: plot_precision_recall_curve(yt, ys, path),
    ),
    PlotDescriptor(
        name="Confusion Matrix Plot",
        file_name="confusion_matrix.png",
        fn=lambda yt, yp, ys, fi, cm, path: plot_confusion_matrix(cm, path),
    ),
    PlotDescriptor(
        name="Feature Importance Plot",
        file_name="feature_importance.png",
        fn=lambda yt, yp, ys, fi, cm, path: plot_feature_importance(fi, path),
    ),
    PlotDescriptor(
        name="Probability Distribution",
        file_name="probability_distribution.png",
        fn=lambda yt, yp, ys, fi, cm, path: plot_probability_distribution(yt, ys, path),
    ),
    PlotDescriptor(
        name="Calibration Curve",
        file_name="calibration_curve.png",
        fn=lambda yt, yp, ys, fi, cm, path: plot_calibration_curve(yt, ys, path),
    ),
    PlotDescriptor(
        name="Lift Curve",
        file_name="lift_curve.png",
        fn=lambda yt, yp, ys, fi, cm, path: plot_lift_curve(yt, ys, path),
    ),
    PlotDescriptor(
        name="Gain Curve",
        file_name="gain_curve.png",
        fn=lambda yt, yp, ys, fi, cm, path: plot_gain_curve(yt, ys, path),
    ),
)
