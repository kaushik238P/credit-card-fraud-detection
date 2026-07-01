"""
Stateless plot generators for the Evaluation Layer.

Always uses matplotlib Agg backend.
Generates:
    - ROC Curve
    - Precision-Recall Curve
    - Confusion Matrix Heatmap
    - Feature Importance Bar Chart
    - Probability Distribution Histograms
    - Calibration Curve
    - Lift Curve
    - Cumulative Gain Curve
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Must be called before pyplot import

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from src.evaluation.exceptions import PlotError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Plot Generator Functions
# ---------------------------------------------------------------------------


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    save_path: str,
    dpi: int = 100,
) -> str:
    """Generates and saves the Receiver Operating Characteristic (ROC) Curve."""
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC Curve")
        plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Baseline")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(True, linestyle="--", alpha=0.6)
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
        plt.close()
        return str(Path(save_path).resolve())
    except Exception as exc:
        raise PlotError("ROC_CURVE", str(exc), {"save_path": save_path}) from exc


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    save_path: str,
    dpi: int = 100,
) -> str:
    """Generates and saves the Precision-Recall (PR) Curve."""
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, color="blue", lw=2, label="PR Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")
        plt.grid(True, linestyle="--", alpha=0.6)
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
        plt.close()
        return str(Path(save_path).resolve())
    except Exception as exc:
        raise PlotError("PR_CURVE", str(exc), {"save_path": save_path}) from exc


def plot_confusion_matrix(
    conf_matrix_dict: dict[str, int],
    save_path: str,
    dpi: int = 100,
) -> str:
    """Generates and saves a Confusion Matrix heatmap representation."""
    try:
        tn = conf_matrix_dict["tn"]
        fp = conf_matrix_dict["fp"]
        fn = conf_matrix_dict["fn"]
        tp = conf_matrix_dict["tp"]
        
        matrix = np.array([[tn, fp], [fn, tp]])
        
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
        
        # Labels and ticks
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted Legitimate", "Predicted Fraud"])
        ax.set_yticklabels(["Actual Legitimate", "Actual Fraud"])
        
        # Display explicit numbers in grid
        for i in range(2):
            for j in range(2):
                color = "white" if matrix[i, j] > (matrix.max() / 2) else "black"
                ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center", color=color, fontweight="bold")
                
        plt.title("Confusion Matrix")
        fig.colorbar(im, ax=ax)
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
        plt.close()
        return str(Path(save_path).resolve())
    except Exception as exc:
        raise PlotError("CONFUSION_MATRIX", str(exc), {"save_path": save_path}) from exc


def plot_feature_importance(
    importance_dict: dict[str, float],
    save_path: str,
    top_n: int = 15,
    dpi: int = 100,
) -> str:
    """Generates a bar chart of top N features sorted by importance."""
    try:
        if not importance_dict:
            raise ValueError("Empty importance dictionary provided.")
            
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features, values = zip(*reversed(sorted_importance))
        
        plt.figure(figsize=(8, 6))
        plt.barh(features, values, color="skyblue")
        plt.xlabel("Importance Score")
        plt.title(f"Top {top_n} Feature Importances")
        plt.grid(axis="x", linestyle="--", alpha=0.6)
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
        plt.close()
        return str(Path(save_path).resolve())
    except Exception as exc:
        raise PlotError("FEATURE_IMPORTANCE", str(exc), {"save_path": save_path}) from exc


def plot_probability_distribution(
    y_true: np.ndarray,
    y_score: np.ndarray,
    save_path: str,
    dpi: int = 100,
) -> str:
    """Plots probability density distribution histograms for classes 0 and 1."""
    try:
        plt.figure(figsize=(7, 5))
        
        # Plot Class 0 (Legitimate)
        plt.hist(
            y_score[y_true == 0],
            bins=50,
            density=True,
            alpha=0.5,
            color="g",
            label="Legitimate (Class 0)",
        )
        
        # Plot Class 1 (Fraud)
        plt.hist(
            y_score[y_true == 1],
            bins=50,
            density=True,
            alpha=0.5,
            color="r",
            label="Fraud (Class 1)",
        )
        
        plt.xlabel("Predicted Probability")
        plt.ylabel("Density")
        plt.title("Probability Distribution by Target Class")
        plt.legend(loc="upper right")
        plt.grid(True, linestyle="--", alpha=0.5)
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
        plt.close()
        return str(Path(save_path).resolve())
    except Exception as exc:
        raise PlotError("PROBABILITY_DISTRIBUTION", str(exc), {"save_path": save_path}) from exc


def plot_calibration_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    save_path: str,
    dpi: int = 100,
) -> str:
    """Plots the calibration curve showing mean predicted value vs fraction of positives."""
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10)
        
        plt.figure(figsize=(6, 5))
        plt.plot(prob_pred, prob_true, marker="o", linewidth=1.5, color="darkorange", label="Model Calibration")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("Calibration Curve (Reliability Diagram)")
        plt.legend(loc="upper left")
        plt.grid(True, linestyle="--", alpha=0.5)
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
        plt.close()
        return str(Path(save_path).resolve())
    except Exception as exc:
        raise PlotError("CALIBRATION_CURVE", str(exc), {"save_path": save_path}) from exc


def plot_lift_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    save_path: str,
    dpi: int = 100,
) -> str:
    """Generates and saves the Lift Curve representation."""
    try:
        idx = np.argsort(y_score)[::-1]
        y_true_sorted = y_true[idx]
        
        total_positives = int(y_true.sum())
        if total_positives == 0:
            raise ValueError("No positive instances found in target array.")
            
        cum_positives = np.cumsum(y_true_sorted)
        cum_recall = cum_positives / total_positives
        
        percentage_population = np.arange(1, len(y_true) + 1) / len(y_true)
        lift = cum_recall / percentage_population
        
        plt.figure(figsize=(6, 5))
        plt.plot(percentage_population, lift, color="purple", lw=2, label="Model Lift")
        plt.axhline(y=1.0, color="navy", linestyle="--", label="Random Baseline (1.0)")
        plt.xlabel("Fraction of Population")
        plt.ylabel("Lift Factor")
        plt.title("Lift Curve")
        plt.legend(loc="upper right")
        plt.grid(True, linestyle="--", alpha=0.5)
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
        plt.close()
        return str(Path(save_path).resolve())
    except Exception as exc:
        raise PlotError("LIFT_CURVE", str(exc), {"save_path": save_path}) from exc


def plot_gain_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    save_path: str,
    dpi: int = 100,
) -> str:
    """Generates and saves the Cumulative Gains Curve."""
    try:
        idx = np.argsort(y_score)[::-1]
        y_true_sorted = y_true[idx]
        
        total_positives = int(y_true.sum())
        if total_positives == 0:
            raise ValueError("No positive instances found in target array.")
            
        cum_positives = np.cumsum(y_true_sorted)
        cum_recall = cum_positives / total_positives
        
        percentage_population = np.arange(1, len(y_true) + 1) / len(y_true)
        
        plt.figure(figsize=(6, 5))
        plt.plot(percentage_population, cum_recall, color="teal", lw=2, label="Model Gains")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Baseline (Random)")
        plt.xlabel("Fraction of Population")
        plt.ylabel("Cumulative Recall (Gain)")
        plt.title("Cumulative Gains Curve")
        plt.legend(loc="lower right")
        plt.grid(True, linestyle="--", alpha=0.5)
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
        plt.close()
        return str(Path(save_path).resolve())
    except Exception as exc:
        raise PlotError("GAIN_CURVE", str(exc), {"save_path": save_path}) from exc
