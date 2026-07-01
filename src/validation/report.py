"""
ReportExporter: formats a ValidationReport into human-readable outputs.

Responsibilities (and ONLY these):
    - Serialise a ValidationReport to JSON.
    - Render a ValidationReport to Markdown.
    - Optionally write the output to a file.

What ReportExporter intentionally does NOT do:
    - Perform any validation logic.
    - Access the DataFrame.
    - Call any check function.
    - Log anything (the validator already logged the run; the report is output).

Design:
    - ReportExporter is constructed with a ValidationReport and exposes
      two public methods: to_json() and to_markdown().
    - Both methods accept an optional path parameter. When provided, the
      output is written to that file in addition to being returned as a string.
      This keeps ReportExporter fully testable with no filesystem dependency
      (just call without path to get the string).
    - to_html() is reserved for a future implementation. The method stub
      exists in the interface so callers do not need to change when it lands.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.validation.models import ValidationReport, ValidationResult


class ReportExporter:
    """
    Serialises a ValidationReport to JSON or Markdown.

    Example:
        >>> exporter = ReportExporter(report)
        >>> md_str = exporter.to_markdown()                     # string only
        >>> json_str = exporter.to_json("reports/run.json")    # string + file

    Args:
        report: The ValidationReport to export. Must be fully built by
            DatasetValidator._build_report() before passing here.
    """

    def __init__(self, report: ValidationReport) -> None:
        if not isinstance(report, ValidationReport):
            raise TypeError(
                f"report must be a ValidationReport instance, "
                f"got {type(report).__name__}."
            )
        self._report = report

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def to_json(self, path: str | Path | None = None) -> str:
        """
        Serialises the ValidationReport to a formatted JSON string.

        Uses ValidationReport.to_dict() which recursively serialises all
        nested models. The output is always valid JSON and can be parsed back
        with json.loads() without any custom decoders.

        Args:
            path: Optional file path to write the JSON to. If None, no file
                is written. Parent directories are created automatically.

        Returns:
            str: The complete, formatted JSON string (2-space indented).
        """
        output = json.dumps(self._report.to_dict(), indent=2, ensure_ascii=False)

        if path is not None:
            resolved = Path(path)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(output, encoding="utf-8")

        return output

    def to_markdown(self, path: str | Path | None = None) -> str:
        """
        Renders the ValidationReport as a human-readable Markdown document.

        The Markdown output is structured for GitHub / GitLab rendering and
        is also readable as plain text. It includes:
            - Report header with key metadata
            - Summary table (overall status, counts, duration)
            - Full check results table
            - Issues section (errors and warnings only, with details)
            - Metadata snapshot (JSON code block)

        Args:
            path: Optional file path to write the Markdown to. If None, no
                file is written. Parent directories are created automatically.

        Returns:
            str: The complete Markdown document as a single string.
        """
        lines: list[str] = []
        r = self._report
        s = r.summary

        # ── Header ───────────────────────────────────────────────────────
        overall_badge = "✅ PASSED" if r.passed else "❌ FAILED"
        lines += [
            f"# Validation Report: `{r.dataset_name}`",
            "",
            f"| Field | Value |",
            f"|-------|-------|",
            f"| **Status** | {overall_badge} |",
            f"| Report ID | `{r.report_id}` |",
            f"| Generated | `{r.generated_at.strftime('%Y-%m-%dT%H:%M:%SZ')}` |",
            f"| Schema | `{r.schema_name}` v`{r.schema_version}` |",
            f"| Dataset Hash | `{r.dataset_hash or 'N/A'}` |",
            f"| Duration | `{r.validation_duration_ms:.1f} ms` |",
            "",
        ]

        # ── Summary table ─────────────────────────────────────────────────
        lines += [
            "## Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Checks | {s.total_checks} |",
            f"| Passed | {s.passed} |",
            f"| Warnings | {s.warnings} |",
            f"| Errors | {s.errors} |",
            f"| Overall | {overall_badge} |",
            "",
        ]

        # ── Check results table ───────────────────────────────────────────
        lines += [
            "## Check Results",
            "",
            "| # | Check | Category | Status | Severity | Time (ms) | Message |",
            "|---|-------|----------|--------|----------|-----------|---------|",
        ]

        status_icons = {
            "PASSED": "✅",
            "WARNING": "⚠️",
            "FAILED": "❌",
        }
        severity_icons = {
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "ERROR": "🔴",
        }

        for i, res in enumerate(r.results, start=1):
            s_icon = status_icons.get(res.status, res.status)
            sev_icon = severity_icons.get(res.severity, res.severity)
            # Truncate long messages in the table cell
            msg_short = res.message if len(res.message) <= 80 else res.message[:77] + "..."
            lines.append(
                f"| {i} "
                f"| `{res.check_name}` "
                f"| {res.category} "
                f"| {s_icon} {res.status} "
                f"| {sev_icon} {res.severity} "
                f"| {res.execution_time_ms:.1f} "
                f"| {msg_short} |"
            )

        lines.append("")

        # ── Issues section (errors + warnings only) ───────────────────────
        errors = r.errors()
        warnings_list = r.warnings()

        if errors or warnings_list:
            lines += ["## Issues", ""]

        if errors:
            lines += [f"### ❌ Errors ({len(errors)})", ""]
            for res in errors:
                lines += self._format_issue_block(res)

        if warnings_list:
            lines += [f"### ⚠️ Warnings ({len(warnings_list)})", ""]
            for res in warnings_list:
                lines += self._format_issue_block(res)

        if not errors and not warnings_list:
            lines += [
                "## Issues",
                "",
                "> No errors or warnings. All checks passed cleanly.",
                "",
            ]

        # ── Metadata snapshot ─────────────────────────────────────────────
        lines += [
            "## Metadata Snapshot",
            "",
            "```json",
            json.dumps(r.metadata_snapshot, indent=2, default=str),
            "```",
            "",
        ]

        output = "\n".join(lines)

        if path is not None:
            resolved = Path(path)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(output, encoding="utf-8")

        return output

    def to_html(self, path: str | Path | None = None) -> str:
        """
        Renders the ValidationReport as an HTML document.

        Note:
            Not implemented in the current version. Reserved for a future
            release. Raises NotImplementedError if called.

        Args:
            path: Reserved. Not used.

        Raises:
            NotImplementedError: Always, until this method is implemented.
        """
        raise NotImplementedError(
            "HTML export is planned for a future release. "
            "Use to_json() or to_markdown() instead."
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _format_issue_block(self, result: ValidationResult) -> list[str]:
        """
        Formats a single ValidationResult as a Markdown detail block.

        Args:
            result: The ValidationResult to format.

        Returns:
            list[str]: Lines of Markdown for this result's block.
        """
        lines: list[str] = [
            f"**`{result.check_name}`** &nbsp;·&nbsp; {result.category}",
            "",
            f"> {result.message}",
            "",
        ]

        if result.details:
            lines += [
                "<details>",
                "<summary>Details</summary>",
                "",
                "```json",
                json.dumps(result.details, indent=2, default=str),
                "```",
                "",
                "</details>",
                "",
            ]

        return lines
