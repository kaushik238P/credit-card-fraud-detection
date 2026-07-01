"""
ReportExporter: serialises an EDAReport to JSON or Markdown.

Design rules:
    - No analysis logic.
    - No DataFrame access.
    - No I/O except writing the output file.
    - Returns the content string regardless of whether a path is provided.
    - Raises ReportGenerationError (not Exception) on failure.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.eda.exceptions import ReportGenerationError
from src.eda.models import EDAReport, EDARecommendation


class ReportExporter:
    """
    Serialises an EDAReport to JSON or Markdown format.

    Example:
        >>> exporter = ReportExporter(report)
        >>> md = exporter.to_markdown("reports/eda/reports/report.md")
        >>> js = exporter.to_json("reports/eda/reports/report.json")

    Args:
        report: The fully built EDAReport to export.
    """

    def __init__(self, report: EDAReport) -> None:
        if not isinstance(report, EDAReport):
            raise TypeError(
                f"report must be an EDAReport instance, got {type(report).__name__}."
            )
        self._report = report

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def to_json(self, path: str | Path | None = None) -> str:
        """
        Serialises the EDAReport to a formatted JSON string.

        Args:
            path: Optional file path to write to. Parent dirs are created
                automatically. Relative path is stored in report_paths.

        Returns:
            str: Complete formatted JSON string.

        Raises:
            ReportGenerationError: If serialisation or file write fails.
        """
        try:
            output = json.dumps(
                self._report.to_dict(), indent=2, ensure_ascii=False, default=str
            )
        except (TypeError, ValueError) as exc:
            raise ReportGenerationError(
                "Failed to serialise EDAReport to JSON.",
                {"error": str(exc), "report_id": self._report.report_id},
            ) from exc

        if path is not None:
            self._write(output, Path(path))

        return output

    def to_markdown(self, path: str | Path | None = None) -> str:
        """
        Renders the EDAReport as a human-readable Markdown document.

        Structure:
            1. Header and metadata
            2. Dataset Overview table
            3. Target Analysis
            4. Numerical Analysis summary
            5. Categorical Analysis summary
            6. Temporal Analysis
            7. Geographic Analysis
            8. Correlation Analysis
            9. Recommendations (sorted HIGH → MEDIUM → LOW)
            10. Figures gallery (image links)
            11. Warnings

        Args:
            path: Optional file path to write to.

        Returns:
            str: Complete Markdown document.

        Raises:
            ReportGenerationError: If file write fails.
        """
        try:
            lines = self._render_markdown()
            output = "\n".join(lines)
        except Exception as exc:
            raise ReportGenerationError(
                "Failed to render EDAReport to Markdown.",
                {"error": str(exc), "report_id": self._report.report_id},
            ) from exc

        if path is not None:
            self._write(output, Path(path))

        return output

    # ------------------------------------------------------------------
    # Private rendering helpers
    # ------------------------------------------------------------------

    def _render_markdown(self) -> list[str]:
        """Builds the Markdown document line by line."""
        r = self._report
        lines: list[str] = []

        # ── Header ───────────────────────────────────────────────────────
        meta = r.report_metadata
        lines += [
            f"# EDA Report: `{r.dataset_name}`",
            "",
            f"| Field | Value |",
            f"|-------|-------|",
            f"| **Report ID** | `{r.report_id}` |",
            f"| **Generated At** | `{r.generated_at}` |",
            f"| **Dataset Hash** | `{r.dataset_hash or 'N/A'}` |",
            f"| **Duration** | `{r.analysis_duration_ms:.1f} ms` |",
            f"| **Figures** | {len(r.figure_paths)} |",
            f"| **Recommendations** | {len(r.recommendations)} |",
            f"| **EDA Version** | `{meta.get('eda_version', 'N/A')}` |",
            f"| **Project Version** | `{meta.get('project_version', 'N/A')}` |",
            f"| **Python Version** | `{meta.get('python_version', 'N/A')}` |",
            f"| **Pandas Version** | `{meta.get('pandas_version', 'N/A')}` |",
            "",
        ]

        # ── Feature Inventory ─────────────────────────────────────────────
        inv = r.feature_inventory
        lines += [
            "## Feature Inventory",
            "",
            "### Identifier Columns",
        ]
        for col in inv.identifiers:
            lines.append(f"- {col}")
        lines += [
            "",
            "### PII Columns",
        ]
        for col in inv.pii:
            lines.append(f"- {col}")
        lines += [
            "",
            "### Numeric Features",
        ]
        for col in inv.numeric:
            lines.append(f"- {col}")
        lines += [
            "",
            "### Categorical Features",
        ]
        for col in inv.categorical:
            lines.append(f"- {col}")
        lines += [
            "",
            "### Temporal Features",
        ]
        for col in inv.temporal:
            lines.append(f"- {col}")
        lines += [
            "",
            "### Geographic Features",
        ]
        for col in inv.geographic:
            lines.append(f"- {col}")
        lines += [
            "",
            "### Target",
        ]
        for col in inv.target:
            lines.append(f"- {col}")
        lines += [
            "",
            "### Drop Before Training",
        ]
        for col in inv.drop_before_training:
            lines.append(f"- {col}")
        lines.append("")

        # ── Feature Engineering Blueprint ───────────────────────────────
        bp = r.feature_engineering_blueprint
        lines += [
            "## Feature Engineering Blueprint",
            "",
            "### Candidate Features",
        ]
        for col in bp.candidate_features:
            lines.append(f"- {col}")
        lines += [
            "",
            "### Drop Features",
        ]
        for col in bp.drop_features:
            lines.append(f"- {col}")
        lines += [
            "",
            "### Transform Features",
        ]
        for col in bp.transform_features:
            lines.append(f"- {col}")
        lines += [
            "",
            "### Encode Features",
        ]
        for col in bp.encode_features:
            lines.append(f"- {col}")
        lines += [
            "",
            "### Scale Features",
        ]
        for col in bp.scale_features:
            lines.append(f"- {col}")
        lines.append("")

        # ── Dataset Overview ──────────────────────────────────────────────
        ov = r.overview
        lines += [
            "## Dataset Overview",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Rows | {ov.num_rows:,} |",
            f"| Columns | {ov.num_columns} |",
            f"| Memory Usage | {ov.memory_usage_mb:.1f} MB |",
            f"| Numeric Columns | {ov.num_numeric_columns} |",
            f"| Categorical Columns | {ov.num_categorical_columns} |",
            f"| Datetime Columns | {ov.num_datetime_columns} |",
            f"| Missing Value Columns | {len(ov.missing_values_summary)} |",
            "",
        ]

        # ── Target Analysis ───────────────────────────────────────────────
        ta = r.target_analysis
        lines += [
            "## Target Analysis",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Target Column | `{ta.target_column}` |",
            f"| Fraud Count | {ta.fraud_count:,} ({ta.fraud_pct:.4f}%) |",
            f"| Legitimate Count | {ta.legitimate_count:,} ({ta.legitimate_pct:.4f}%) |",
            f"| Imbalance Ratio | {ta.imbalance_ratio:.1f}:1 |",
            "",
        ]

        # ── DOB / Age Demographics ────────────────────────────────────────
        if r.dob_analysis is not None:
            dob = r.dob_analysis
            lines += ["## Age Demographics (DOB Analysis)", ""]
            if dob.get("parsed"):
                stats = dob.get("statistics", {})
                lines += [
                    f"DOB column parsed successfully: `{dob.get('column_used')}`.",
                    "",
                    "### Age Statistics",
                    f"- **Mean Age:** {stats.get('mean'):.2f} years",
                    f"- **Median Age:** {stats.get('median'):.2f} years",
                    f"- **Min Age:** {stats.get('min'):.2f} years",
                    f"- **Max Age:** {stats.get('max'):.2f} years",
                    f"- **Std Dev:** {stats.get('std'):.2f} years",
                    "",
                    "### Age Groups Distribution",
                    "| Age Group | Count |",
                    "|-----------|-------|",
                ]
                for grp, cnt in dob.get("age_groups", {}).items():
                    lines.append(f"| {grp} | {cnt:,} |")
            else:
                lines.append(f"DOB parsing failed on column `{dob.get('column_used')}`. Error: {dob.get('error')}")
            lines.append("")

        # ── Numerical Analysis ────────────────────────────────────────────
        if r.numerical_analysis is not None:
            na = r.numerical_analysis
            lines += ["## Numerical Analysis", ""]
            lines += [
                "| Column | Mean | Median | Std | Skewness | Outlier % |",
                "|--------|------|--------|-----|----------|-----------|",
            ]
            for col, stats in na.summaries.items():
                lines.append(
                    f"| `{col}` "
                    f"| {stats.get('mean', 0):.4f} "
                    f"| {stats.get('median', 0):.4f} "
                    f"| {stats.get('std', 0):.4f} "
                    f"| {stats.get('skewness', 0):.4f} "
                    f"| {stats.get('outlier_pct', 0):.2f}% |"
                )
            if na.high_skewness_columns:
                lines += [
                    "",
                    f"> **High Skewness:** {', '.join(f'`{c}`' for c in na.high_skewness_columns)}",
                ]
            if na.high_outlier_columns:
                lines += [
                    f"> **High Outlier %:** {', '.join(f'`{c}`' for c in na.high_outlier_columns)}",
                ]
            lines.append("")

        # ── Categorical Analysis ──────────────────────────────────────────
        if r.categorical_analysis is not None:
            ca = r.categorical_analysis
            lines += ["## Categorical Analysis", ""]
            lines += [
                "| Column | Unique Values | High Cardinality | Top Fraud Category | Fraud Rate |",
                "|--------|--------------|------------------|--------------------|------------|",
            ]
            for col, stats in ca.summaries.items():
                hc = "Yes ⚠️" if stats.get("is_high_cardinality") else "No"
                top_fraud = stats.get("highest_fraud_category", "N/A")
                fraud_rate = stats.get("highest_fraud_rate", 0)
                lines.append(
                    f"| `{col}` "
                    f"| {stats.get('unique_count', 0):,} "
                    f"| {hc} "
                    f"| {top_fraud} "
                    f"| {fraud_rate * 100:.4f}% |"
                )
            lines.append("")

        # ── Temporal Analysis ─────────────────────────────────────────────
        if r.temporal_analysis is not None:
            te = r.temporal_analysis
            lines += [
                "## Temporal Analysis",
                "",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Column Parsed | `{te.datetime_column_used}` |",
                f"| Peak Fraud Hour | `{te.peak_fraud_hour}:00` |",
                f"| Peak Fraud Weekday | `{te.peak_fraud_weekday}` |",
                f"| Peak Fraud Month | `{te.peak_fraud_month}` |",
                "",
            ]

        # ── Geographic Analysis ───────────────────────────────────────────
        if r.geographic_analysis is not None:
            ge = r.geographic_analysis
            lines += [
                "## Geographic Analysis",
                "",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Total States | {ge.total_states} |",
                f"| Total Cities | {ge.total_cities:,} |",
                f"| Top Fraud States | {', '.join(f'`{s}`' for s in ge.top_fraud_states[:5])} |",
                f"| Top Fraud Cities | {', '.join(f'`{c}`' for c in ge.top_fraud_cities[:5])} |",
                "",
            ]

        # ── Correlation Analysis ──────────────────────────────────────────
        if r.correlation_analysis is not None:
            co = r.correlation_analysis
            lines += ["## Correlation Analysis", ""]
            
            pearson_corr = co.target_correlations.get("pearson", {})
            pb_corr = co.target_correlations.get("point_biserial", {})

            if pearson_corr or pb_corr:
                lines += [
                    "| Feature | Pearson Correlation | Point-Biserial Correlation |",
                    "|---------|---------------------|----------------------------|",
                ]
                for col in sorted(list(set(list(pearson_corr.keys()) + list(pb_corr.keys())))):
                    p_val = pearson_corr.get(col, 0.0)
                    pb_val = pb_corr.get(col, 0.0)
                    lines.append(f"| `{col}` | {p_val:+.4f} | {pb_val:+.4f} |")
                lines.append("")
                
            if co.high_correlation_pairs:
                lines += [
                    f"> **Highly Correlated Pairs** (|r| > threshold): "
                    f"{len(co.high_correlation_pairs)} pair(s) detected.",
                    "",
                ]

        # ── Recommendations ───────────────────────────────────────────────
        lines += ["## Recommendations", ""]
        if not r.recommendations:
            lines += ["> No recommendations generated.", ""]
        else:
            priority_groups: dict[str, list[EDARecommendation]] = {
                "HIGH": [], "MEDIUM": [], "LOW": []
            }
            for rec in r.recommendations:
                priority_groups.setdefault(rec.priority, []).append(rec)

            icons = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
            for priority in ("HIGH", "MEDIUM", "LOW"):
                recs = priority_groups.get(priority, [])
                if not recs:
                    continue
                lines += [f"### {icons[priority]} {priority} Priority", ""]
                for rec in recs:
                    lines += [
                        f"**{rec.category}** &nbsp;·&nbsp; Stage: `{'`, `'.join(rec.affects_stage)}`",
                        "",
                        f"> 📊 **Finding:** {rec.finding}",
                        "",
                        f"> 💡 **Recommendation:** {rec.recommendation}",
                        "",
                    ]

        # ── Figures ───────────────────────────────────────────────────────
        if r.figure_paths:
            lines += ["## Figures", ""]
            for name, rel_path in r.figure_paths.items():
                display_name = name.replace("_", " ").title()
                lines += [
                    f"### {display_name}",
                    "",
                    f"![{display_name}]({rel_path})",
                    "",
                ]

        # ── Warnings ──────────────────────────────────────────────────────
        if r.warnings:
            lines += ["## Warnings", ""]
            for w in r.warnings:
                lines.append(f"> ⚠️ {w}")
            lines.append("")

        return lines

    def _write(self, content: str, path: Path) -> None:
        """Writes content to a file, creating parent directories as needed."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        except OSError as exc:
            raise ReportGenerationError(
                f"Failed to write report to '{path}'.",
                {"path": str(path), "error": str(exc)},
            ) from exc
