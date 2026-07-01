"""
EvaluationReport generator for compiling JSON/Markdown reports.

Responsible for taking an EvaluationReport dataclass and generating a
human-readable Markdown report layout with tables and plot links.
"""

from __future__ import annotations

import logging
from typing import Any

from src.evaluation.models import EvaluationReport

logger = logging.getLogger(__name__)


class EvaluationReportGenerator:
    """
    Compiles an EvaluationReport dataclass into printable Markdown format.
    """

    @staticmethod
    def generate_markdown(report: EvaluationReport) -> str:
        """
        Generates a human-readable Markdown summary report.

        Args:
            report: Populated EvaluationReport instance.

        Returns:
            str: Markdown string document.
        """
        md = []

        # Title & Header
        md.append("# Model Evaluation Report")
        md.append(f"**Generated At:** `{report.generated_at}`")
        md.append("")

        # ── 1. Metadata ──────────────────────────────────────────────────
        md.append("## 1. System Metadata & Lineage")
        md.append("| Attribute | Value |")
        md.append("|---|---|")
        md.append(f"| **Model Version (Run ID)** | `{report.model_version}` |")
        md.append(f"| **Preprocessing Version** | `{report.preprocessing_version}` |")
        md.append(f"| **Dataset SHA-256 Hash** | `{report.dataset_hash or 'N/A'}` |")
        md.append(f"| **Feature Schema Hash** | `{report.feature_schema_hash}` |")
        md.append(f"| **Evaluation Duration** | `{report.evaluation_duration:.0f} ms` |")
        md.append("")

        # ── 1b. Fold Statistics (Refinement 1) ───────────────────────────
        md.append("## 1b. Fold Statistics")
        md.append("Data split summary across training, validation, and testing partitions.")
        md.append("")
        md.append("| Fold | Rows | Legitimate | Fraud | Fraud % |")
        md.append("|---|---|---|---|---|")
        for fold_name, stats in report.fold_statistics.items():
            md.append(
                f"| **{fold_name}** | {stats['rows']:,} | {stats['legitimate_count']:,} | "
                f"{stats['fraud_count']:,} | {stats['fraud_rate']:.4f}% |"
            )
        md.append("")

        # ── 2. Decision Threshold ─────────────────────────────────────────
        tr = report.threshold_result
        md.append("## 2. Decision Boundary Optimization")
        md.append(f"- **Strategy applied:** `{tr.optimization_strategy}`")
        md.append(f"- **Default Threshold:** `{tr.default_threshold}`")
        md.append(f"- **Optimized Decision Threshold:** `{tr.optimized_threshold:.4f}`")
        md.append("")
        md.append("### Validation Performance at Optimized Threshold")
        md.append("| Metric | Value |")
        md.append("|---|---|")
        md.append(f"| Precision | `{tr.validation_precision:.6f}` |")
        md.append(f"| Recall | `{tr.validation_recall:.6f}` |")
        md.append(f"| F1 Score | `{tr.validation_f1:.6f}` |")
        md.append("")

        # ── 2b. Threshold Comparison (Refinement 2) ───────────────────────
        md.append("### Decision Boundary Threshold Comparison (Test Set)")
        md.append("Performance comparison between default boundary and optimized decision threshold.")
        md.append("")
        md.append("| Metric | Default (0.5) | Optimized |")
        md.append("|---|---|---|")
        comp = report.threshold_comparison
        def_metrics = comp.get("default_threshold_0.5", {})
        opt_metrics = comp.get("optimized_threshold", {})
        
        comparison_keys = [
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("f1", "F1 Score"),
            ("specificity", "Specificity"),
            ("false_positive_rate", "False Positive Rate"),
            ("false_negative_rate", "False Negative Rate"),
            ("fraud_detection_rate", "Fraud Detection Rate"),
        ]
        for key_name, label in comparison_keys:
            def_val = def_metrics.get(key_name, 0.0)
            opt_val = opt_metrics.get(key_name, 0.0)
            md.append(f"| {label} | `{def_val:.6f}` | `{opt_val:.6f}` |")
        md.append("")

        # ── 2c. Probability Calibration Note (Refinement 6) ─────────────
        md.append("> [!NOTE]")
        md.append(f"> {report.calibration_note}")
        md.append("")

        # ── 3. Performance Metrics ───────────────────────────────────────
        md.append("## 3. Core Model Performance")
        md.append("Standard machine learning performance metrics computed across folds.")
        md.append("")
        md.append("| Metric | Validation Fold | Test Fold |")
        md.append("|---|---|---|")

        # Gather keys from both validation and test metrics, ignoring nested dictionaries like Confusion Matrix or Classification Report
        all_metric_keys = sorted(
            list(set(report.validation_metrics.keys()) | set(report.test_metrics.keys()))
        )

        for key in all_metric_keys:
            val_val = report.validation_metrics.get(key, "N/A")
            test_val = report.test_metrics.get(key, "N/A")

            # Skip complex objects in tabular overview
            if isinstance(val_val, dict) or isinstance(test_val, dict):
                continue

            val_str = f"{val_val:.6f}" if isinstance(val_val, (int, float)) else str(val_val)
            test_str = f"{test_val:.6f}" if isinstance(test_val, (int, float)) else str(test_val)

            md.append(f"| {key} | `{val_str}` | `{test_str}` |")
        md.append("")

        # ── 4. Business Metrics ──────────────────────────────────────────
        md.append("## 4. Business & Financial Impact Metrics")
        md.append("Operational metrics to estimate business outcomes on the Test set.")
        md.append("")
        md.append("| Business Metric | Score | Rationale |")
        md.append("|---|---|---|")
        for key, details in sorted(report.business_metrics.items()):
            if isinstance(details, dict):
                score = details.get("value", 0.0)
                desc = details.get("description", "")
            else:
                score = details
                desc = "Calculated business metric."
            
            score_str = f"{score:.6f}" if isinstance(score, (int, float)) else str(score)
            if "Amount" in key or "Saved" in key:
                score_str = f"${score:,.2f}" if isinstance(score, (int, float)) else str(score)
            
            md.append(f"| **{key}** | `{score_str}` | {desc} |")
        md.append("")

        # Precision@K Context display
        pk_metric = report.business_metrics.get("Precision_At_K", {})
        if pk_metric and isinstance(pk_metric, dict) and "metadata" in pk_metric:
            meta = pk_metric["metadata"]
            md.append("### Precision@K Context Details")
            md.append(f"- **Top K Selected:** `{meta.get('k')}`")
            md.append(f"- **Fraud cases inside Top K:** `{meta.get('fraud_cases_inside_top_k')}`")
            md.append(f"- **Legitimate cases inside Top K:** `{meta.get('legitimate_cases_inside_top_k')}`")
            md.append(f"- **Total fraud cases in Test Fold:** `{meta.get('total_frauds_in_test_fold')}`")
            md.append("")

        # ── 5. Feature Importance ─────────────────────────────────────────
        md.append("## 5. Model-Native Feature Importance")
        md.append("| Rank | Feature Name | Importance Score |")
        md.append("|---|---|---|")
        sorted_importance = sorted(
            report.feature_importance.items(), key=lambda x: x[1], reverse=True
        )
        for rank, (name, val) in enumerate(sorted_importance, 1):
            md.append(f"| {rank} | `{name}` | `{val:.6f}` |")
        md.append("")

        # ── 6. Visualizations ────────────────────────────────────────────
        md.append("## 6. Diagnostic Visualizations")
        md.append("")
        for p in report.plots:
            md.append(f"### {p.plot_name}")
            md.append(f"![{p.plot_name}](plots/{p.file_name})")
            md.append("")

        # ── 7. Operational Interpretation (Refinement 8) ─────────────────
        md.append("## 7. Operational Interpretation")
        md.append("Recommended runtime usage constraints and business guidelines:")
        
        # Calculate rates/values for bullet points
        stats_test = report.fold_statistics.get("Test", {})
        obs_fraud_rate = stats_test.get("fraud_rate", 0.0)
        
        opt_thresh = report.threshold_result.optimized_threshold
        alert_metrics = report.business_metrics.get("Alert_Rate", {})
        alert_rate_val = alert_metrics.get("value", 0.0) if isinstance(alert_metrics, dict) else 0.0
        
        md.append(f"- **Estimated Alert Volume:** The model will flag approximately `{alert_rate_val * 100:.2f}%` of incoming transactions.")
        md.append(f"- **Observed Fraud Rate:** The historical base rate of fraud observed in the test partition is `{obs_fraud_rate:.4f}%`.")
        md.append(f"- **Recommended Review Threshold:** Operational screening queues should apply the optimized decision threshold of `{opt_thresh:.4f}`.")
        md.append("- **Fraud Screening Suitability:** High precision on top scores makes the model highly suitable for low-latency automated transaction blocking.")
        md.append("- **Continuous Tuning Requirement:** Decision boundaries should be dynamically retuned when operational alert queues undergo scale changes.")
        md.append("")

        # ── 8. Warnings ──────────────────────────────────────────────────
        if report.warnings:
            md.append("## 8. Execution Warnings")
            md.append("")
            for w in report.warnings:
                md.append(f"- ⚠️ {w}")
            md.append("")

        return "\n".join(md)