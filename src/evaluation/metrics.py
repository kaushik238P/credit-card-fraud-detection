"""
Stateless evaluation metrics functions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes Area Under the Receiver Operating Characteristic Curve (ROC AUC)."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def compute_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes Area Under the Precision-Recall Curve (PR AUC / Average Precision)."""
    return float(average_precision_score(y_true, y_score))


def compute_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Recall (Sensitivity)."""
    return float(recall_score(y_true, y_pred, zero_division=0.0))


def compute_sensitivity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Sensitivity (same as Recall)."""
    return compute_recall(y_true, y_pred)


def compute_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Precision."""
    return float(precision_score(y_true, y_pred, zero_division=0.0))


def compute_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Specificity (True Negative Rate)."""
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes F1 Score."""
    return float(f1_score(y_true, y_pred, zero_division=0.0))


def compute_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Balanced Accuracy."""
    sens = compute_sensitivity(y_true, y_pred)
    spec = compute_specificity(y_true, y_pred)
    return float((sens + spec) / 2.0)


def compute_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Matthews Correlation Coefficient (MCC)."""
    return float(matthews_corrcoef(y_true, y_pred))


def compute_log_loss(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes Log Loss."""
    return float(log_loss(y_true, y_score))


def compute_brier_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes Brier Score Loss."""
    return float(brier_score_loss(y_true, y_score))


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    """Computes raw confusion matrix values."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def compute_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Computes classification report as a dictionary."""
    return classification_report(y_true, y_pred, output_dict=True, zero_division=0.0)


# ── Business & Operational Metrics ──────────────────────────────────────────


def compute_fraud_detection_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Fraud Detection Rate (FDR), same as Recall."""
    return compute_recall(y_true, y_pred)


def compute_false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes False Positive Rate (FPR)."""
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0


def compute_false_negative_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes False Negative Rate (FNR)."""
    _, _, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0


def compute_fraud_capture_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    amounts: np.ndarray | None = None,
) -> float:
    """
    Computes Fraud Capture Rate (value of caught fraud / value of total fraud).

    If amounts is None, falls back to count-based recall.
    """
    if amounts is None:
        return compute_recall(y_true, y_pred)

    total_fraud_value = float(amounts[y_true == 1].sum())
    if total_fraud_value <= 0:
        return 0.0

    caught_mask = (y_true == 1) & (y_pred == 1)
    caught_fraud_value = float(amounts[caught_mask].sum())
    return caught_fraud_value / total_fraud_value


def compute_transactions_flagged(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes the percentage of overall transactions flagged as fraud."""
    total = len(y_true)
    if total <= 0:
        return 0.0
    flagged = int(y_pred.sum())
    return flagged / total


def compute_precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """
    Computes Precision@K after sorting predictions by probability score.
    """
    if k <= 0 or len(y_true) == 0:
        return 0.0
    actual_k = min(k, len(y_true))
    idx = np.argsort(y_score)[::-1]
    top_k_y_true = y_true[idx[:actual_k]]
    return float(top_k_y_true.sum() / actual_k)


def compute_potential_fraud_amount_detected(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    amounts: np.ndarray | None = None,
) -> float:
    """
    Computes Potential Fraud Amount Detected.

    Definition: Sum of transaction amounts for correctly detected fraud transactions.
    Do NOT imply actual monetary savings.
    """
    if amounts is None:
        return 0.0
    tp_mask = (y_true == 1) & (y_pred == 1)
    return float(amounts[tp_mask].sum())