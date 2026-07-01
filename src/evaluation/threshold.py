"""
ThresholdOptimizer for the Evaluation Layer.

Optimises prediction thresholds using validation set probabilities only.
Supports strategies:
    - Maximum F1
    - Maximum Recall
    - Maximum Precision
    - Minimum False Positives
    - Business Threshold / Default
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.metrics import precision_recall_curve

from src.evaluation.exceptions import ThresholdError
from src.evaluation.models import ThresholdResult

logger = logging.getLogger(__name__)


class ThresholdOptimizer:
    """
    Optimises decision thresholds using validation set probabilities.

    Operates strictly on cached predictions/probabilities — never retrains
    the model or mutates datasets.
    """

    def __init__(
        self,
        strategy: str = "MAX_F1",
        default_threshold: float = 0.5,
        target_recall: float = 0.8,
        target_precision: float = 0.8,
    ) -> None:
        """
        Initialises the ThresholdOptimizer.

        Args:
            strategy: Strategy name (e.g. "MAX_F1", "MAX_RECALL", "MAX_PRECISION",
                "MIN_FP", "BUSINESS_THRESHOLD").
            default_threshold: Fallback/default prediction threshold.
            target_recall: Target recall constraint for MAX_RECALL / MIN_FP.
            target_precision: Target precision constraint for MAX_PRECISION.
        """
        self.strategy = strategy.upper()
        self.default_threshold = default_threshold
        self.target_recall = target_recall
        self.target_precision = target_precision

    def optimize(self, y_true: np.ndarray, y_score: np.ndarray) -> ThresholdResult:
        """
        Optimises the threshold on validation targets and probabilities.

        Args:
            y_true: Validation labels (binary).
            y_score: Validation predicted probabilities.

        Returns:
            ThresholdResult: Selected optimal threshold and validation stats.

        Raises:
            ThresholdError: If arrays are empty or optimization fails.
        """
        if len(y_true) == 0 or len(y_score) == 0:
            raise ThresholdError(
                strategy=self.strategy,
                detail="Empty validation arrays provided to ThresholdOptimizer.",
            )

        try:
            # Handle trivial case or default strategy
            if self.strategy in ("BUSINESS_THRESHOLD", "DEFAULT"):
                opt_thresh = self.default_threshold
            else:
                opt_thresh = self._find_optimal_threshold(y_true, y_score)

            # Compute validation metrics at optimized threshold
            precision, recall, f1 = self._compute_metrics_at(y_true, y_score, opt_thresh)

            logger.info(
                "Threshold optimized | strategy=%s | threshold=%.4f | "
                "val_prec=%.4f | val_rec=%.4f | val_f1=%.4f",
                self.strategy,
                opt_thresh,
                precision,
                recall,
                f1,
            )

            return ThresholdResult(
                optimization_strategy=self.strategy,
                optimized_threshold=opt_thresh,
                validation_precision=precision,
                validation_recall=recall,
                validation_f1=f1,
                default_threshold=self.default_threshold,
            )

        except ThresholdError:
            raise
        except Exception as exc:
            raise ThresholdError(
                strategy=self.strategy,
                detail=str(exc),
                context={"original_exc": str(exc)},
            ) from exc

    # ------------------------------------------------------------------
    # Private search methods
    # ------------------------------------------------------------------

    def _find_optimal_threshold(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Finds the optimal threshold matching the configured strategy."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)

        if len(thresholds) == 0:
            return self.default_threshold

        # Align precision/recall arrays (which have len == len(thresholds) + 1)
        prec_aligned = precision[:-1]
        rec_aligned = recall[:-1]

        # F1 list
        denom = prec_aligned + rec_aligned
        f1 = np.zeros_like(thresholds)
        mask = denom > 0
        f1[mask] = 2 * prec_aligned[mask] * rec_aligned[mask] / denom[mask]

        if self.strategy == "MAX_F1":
            best_idx = np.argmax(f1)
            return float(thresholds[best_idx])

        elif self.strategy in ("MAX_RECALL", "MIN_FP"):
            # Highest threshold yielding recall >= target_recall (minimises FPs)
            candidates = np.where(rec_aligned >= self.target_recall)[0]
            if len(candidates) > 0:
                best_idx = candidates[-1]  # rightmost index has highest threshold
                return float(thresholds[best_idx])
            # Fallback to maximizing recall
            return float(thresholds[np.argmax(rec_aligned)])

        elif self.strategy == "MAX_PRECISION":
            # Lowest threshold yielding precision >= target_precision (maximises recall)
            candidates = np.where(prec_aligned >= self.target_precision)[0]
            if len(candidates) > 0:
                best_idx = candidates[0]  # leftmost index has lowest threshold
                return float(thresholds[best_idx])
            # Fallback to maximizing precision
            return float(thresholds[np.argmax(prec_aligned)])

        else:
            logger.warning(
                "Unknown strategy '%s' — falling back to default threshold %.2f.",
                self.strategy,
                self.default_threshold,
            )
            return self.default_threshold

    @staticmethod
    def _compute_metrics_at(
        y_true: np.ndarray,
        y_score: np.ndarray,
        threshold: float,
    ) -> tuple[float, float, float]:
        """Calculates precision, recall, and F1 score at a specific threshold."""
        y_pred = (y_score >= threshold).astype(int)

        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        denom = precision + recall
        f1 = float(2 * precision * recall / denom) if denom > 0 else 0.0

        return precision, recall, f1
