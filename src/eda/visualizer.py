"""
EDAVisualizer: generates and saves figures for the EDA Layer.

Design rules:
    - Each plot method is independently callable with no shared state.
    - Every method saves its figure before returning and calls plt.close().
    - Every method raises VisualizationError (not Exception) on failure.
    - No statistical computations — data comes from pre-computed models or
      raw DataFrame columns only when distribution plots require raw data.
    - Returns relative paths (from project root) not absolute paths.
    - Figure filenames are fixed and prefixed with a two-digit sort index.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")  # Non-interactive backend safe for Docker/CI

from config.settings import settings
from src.eda.exceptions import VisualizationError
from src.eda.models import (
    CategoricalAnalysis,
    CorrelationAnalysis,
    GeographicAnalysis,
    TargetAnalysis,
    TemporalAnalysis,
)

# Suppress seaborn FutureWarnings from matplotlib internals
warnings.filterwarnings("ignore", category=FutureWarning)


class EDAVisualizer:
    """
    Generates and saves the 7 standard EDA figures.

    Example:
        >>> viz = EDAVisualizer(figures_dir="reports/eda/figures")
        >>> paths = viz.generate_all(df, target, categorical, temporal, geographic, correlation)

    Args:
        figures_dir: Directory to save figures. Parent dirs are created
            automatically. Default from settings.eda.output_dir.
        dpi: Figure resolution. Default from settings.eda.figure_dpi.
        style: Matplotlib style name. Default from settings.eda.figure_style.
        top_n: Number of top categories to show in bar charts.
    """

    def __init__(
        self,
        figures_dir: str | Path | None = None,
        dpi: int | None = None,
        style: str | None = None,
        top_n: int | None = None,
    ) -> None:
        cfg = settings.eda
        self._figures_dir = Path(figures_dir) if figures_dir is not None else Path(cfg.output_dir) / "figures"
        self._dpi = dpi or cfg.figure_dpi
        self._style = style or cfg.figure_style
        self._top_n = top_n or cfg.top_n_categories
        self._figures_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_all(
        self,
        df: pd.DataFrame,
        target: TargetAnalysis,
        categorical: CategoricalAnalysis | None,
        temporal: TemporalAnalysis | None,
        geographic: GeographicAnalysis | None,
        correlation: CorrelationAnalysis | None,
    ) -> dict[str, str]:
        """
        Generates all standard EDA figures.

        Each individual plot failure is caught and skipped; the rest
        continue. Returns a dict of {figure_name: relative_path}.

        Args:
            df: The original DataFrame (used for distribution plots).
            target: Target analysis result.
            categorical: Categorical analysis result, or None.
            temporal: Temporal analysis result, or None.
            geographic: Geographic analysis result, or None.
            correlation: Correlation analysis result, or None.

        Returns:
            dict[str, str]: Mapping of figure name to relative file path.
        """
        figure_paths: dict[str, str] = {}

        plots = [
            ("class_distribution",  self._plot_class_distribution,  (target,)),
            ("amount_distribution",  self._plot_amount_distribution, (df,)),
            ("fraud_by_category",    self._plot_fraud_by_category,   (categorical,)),
            ("fraud_by_hour",        self._plot_fraud_by_hour,       (temporal,)),
            ("fraud_by_state",       self._plot_fraud_by_state,      (geographic,)),
            ("correlation_heatmap",  self._plot_correlation_heatmap, (df, correlation)),
            ("missing_values_heatmap", self._plot_missing_values_heatmap, (df,)),
        ]

        for name, fn, args in plots:
            try:
                path = fn(*args)
                if path is not None:
                    figure_paths[name] = path
            except VisualizationError:
                pass  # Caller (EDAAnalyzer) logs the warning

        return figure_paths

    # ------------------------------------------------------------------
    # Individual plot methods
    # ------------------------------------------------------------------

    def _plot_class_distribution(self, target: TargetAnalysis) -> str:
        """Bar chart showing fraud vs legitimate counts and percentages."""
        try:
            self._apply_style()
            fig, ax = plt.subplots(figsize=(7, 5))

            labels = ["Legitimate", "Fraud"]
            counts = [target.legitimate_count, target.fraud_count]
            colors = ["#2ecc71", "#e74c3c"]

            bars = ax.bar(labels, counts, color=colors, edgecolor="white", linewidth=1.2)
            ax.set_title("Class Distribution: Fraud vs Legitimate", fontsize=14, fontweight="bold", pad=14)
            ax.set_ylabel("Transaction Count", fontsize=11)
            ax.set_xlabel("Class", fontsize=11)

            for bar, count, pct in zip(bars, counts, [target.legitimate_pct, target.fraud_pct]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{count:,}\n({pct:.2f}%)",
                    ha="center", va="bottom", fontsize=10,
                )

            ax.yaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
            )
            plt.tight_layout()
            return self._save(fig, "01_class_distribution.png")
        except Exception as exc:
            raise VisualizationError(
                "Failed to plot class distribution.", {"error": str(exc)}
            ) from exc

    def _plot_amount_distribution(self, df: pd.DataFrame) -> str | None:
        """Histogram of transaction amounts (log scale) split by fraud."""
        if "amt" not in df.columns or "is_fraud" not in df.columns:
            raise VisualizationError(
                "Columns 'amt' or 'is_fraud' not found for amount distribution plot.",
                {"columns": list(df.columns)},
            )
        try:
            self._apply_style()
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            legit = df[df["is_fraud"] == 0]["amt"]
            fraud = df[df["is_fraud"] == 1]["amt"]

            # Left: overlapping distributions
            axes[0].hist(legit, bins=60, alpha=0.6, color="#2ecc71", label="Legitimate", density=True)
            axes[0].hist(fraud, bins=60, alpha=0.6, color="#e74c3c", label="Fraud", density=True)
            axes[0].set_xlabel("Transaction Amount ($)", fontsize=11)
            axes[0].set_ylabel("Density", fontsize=11)
            axes[0].set_title("Amount Distribution by Class", fontsize=13, fontweight="bold")
            axes[0].legend()
            axes[0].set_yscale("log")

            # Right: log-transformed amount
            import numpy as np
            axes[1].hist(
                np.log1p(legit), bins=60, alpha=0.6, color="#2ecc71",
                label="Legitimate", density=True,
            )
            axes[1].hist(
                np.log1p(fraud), bins=60, alpha=0.6, color="#e74c3c",
                label="Fraud", density=True,
            )
            axes[1].set_xlabel("log(Amount + 1)", fontsize=11)
            axes[1].set_ylabel("Density", fontsize=11)
            axes[1].set_title("Log-Transformed Amount Distribution", fontsize=13, fontweight="bold")
            axes[1].legend()

            plt.suptitle("Transaction Amount Analysis", fontsize=14, fontweight="bold", y=1.01)
            plt.tight_layout()
            return self._save(fig, "02_amount_distribution.png")
        except VisualizationError:
            raise
        except Exception as exc:
            raise VisualizationError(
                "Failed to plot amount distribution.", {"error": str(exc)}
            ) from exc

    def _plot_fraud_by_category(self, categorical: CategoricalAnalysis | None) -> str | None:
        """Horizontal bar chart of fraud rate by merchant category."""
        if categorical is None or "category" not in categorical.summaries:
            raise VisualizationError(
                "CategoricalAnalysis missing 'category' column data.",
                {},
            )
        try:
            self._apply_style()
            summary = categorical.summaries["category"]
            top_cats = summary.get("top_categories", [])
            if not top_cats:
                raise VisualizationError("No category data available.", {})

            # Sort by fraud rate descending
            sorted_cats = sorted(top_cats, key=lambda x: x.get("fraud_rate", 0), reverse=True)
            n = min(self._top_n, len(sorted_cats))
            cats = [c["value"] for c in sorted_cats[:n]]
            rates = [round(c.get("fraud_rate", 0) * 100, 4) for c in sorted_cats[:n]]

            fig, ax = plt.subplots(figsize=(10, max(5, n * 0.45)))
            colors = ["#e74c3c" if r > 1.0 else "#e67e22" if r > 0.5 else "#3498db" for r in rates]
            bars = ax.barh(cats, rates, color=colors, edgecolor="white", linewidth=0.8)

            for bar, rate in zip(bars, rates):
                ax.text(
                    bar.get_width() + 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f"{rate:.3f}%",
                    va="center", fontsize=9,
                )

            ax.set_xlabel("Fraud Rate (%)", fontsize=11)
            ax.set_title("Fraud Rate by Merchant Category", fontsize=13, fontweight="bold")
            ax.invert_yaxis()
            plt.tight_layout()
            return self._save(fig, "03_fraud_by_category.png")
        except VisualizationError:
            raise
        except Exception as exc:
            raise VisualizationError(
                "Failed to plot fraud by category.", {"error": str(exc)}
            ) from exc

    def _plot_fraud_by_hour(self, temporal: TemporalAnalysis | None) -> str | None:
        """Line chart of fraud rate by hour of day."""
        if temporal is None:
            raise VisualizationError("TemporalAnalysis is None. Skipping hourly plot.", {})
        try:
            self._apply_style()
            hours = list(range(24))
            rates = [
                temporal.fraud_by_hour.get(str(h), 0) * 100 for h in hours
            ]
            volumes = [
                temporal.transaction_volume_by_hour.get(str(h), 0) for h in hours
            ]

            fig, ax1 = plt.subplots(figsize=(12, 5))
            ax2 = ax1.twinx()

            ax1.plot(hours, rates, color="#e74c3c", linewidth=2.5, marker="o",
                     markersize=5, label="Fraud Rate (%)", zorder=3)
            ax1.fill_between(hours, rates, alpha=0.15, color="#e74c3c")
            ax2.bar(hours, volumes, alpha=0.25, color="#3498db", label="Transaction Volume", zorder=1)

            ax1.set_xlabel("Hour of Day", fontsize=11)
            ax1.set_ylabel("Fraud Rate (%)", color="#e74c3c", fontsize=11)
            ax2.set_ylabel("Transaction Volume", color="#3498db", fontsize=11)
            ax1.set_xticks(hours)
            ax1.set_title("Fraud Rate by Hour of Day", fontsize=13, fontweight="bold")

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

            ax1.axvline(x=temporal.peak_fraud_hour, color="#e74c3c",
                        linestyle="--", alpha=0.6, label=f"Peak: {temporal.peak_fraud_hour}:00")
            plt.tight_layout()
            return self._save(fig, "04_fraud_by_hour.png")
        except VisualizationError:
            raise
        except Exception as exc:
            raise VisualizationError(
                "Failed to plot fraud by hour.", {"error": str(exc)}
            ) from exc

    def _plot_fraud_by_state(self, geographic: GeographicAnalysis | None) -> str | None:
        """Horizontal bar chart of top states by fraud rate."""
        if geographic is None or not geographic.fraud_by_state:
            raise VisualizationError("GeographicAnalysis is None or empty. Skipping state plot.", {})
        try:
            self._apply_style()
            # Sort states by fraud rate
            sorted_states = sorted(
                geographic.fraud_by_state.items(), key=lambda x: x[1], reverse=True
            )
            n = min(self._top_n, len(sorted_states))
            states = [s[0] for s in sorted_states[:n]]
            rates = [round(s[1] * 100, 4) for s in sorted_states[:n]]

            fig, ax = plt.subplots(figsize=(10, max(5, n * 0.4)))
            colors = plt.cm.RdYlGn_r(  # type: ignore[attr-defined]
                [i / max(n - 1, 1) for i in range(n)]
            )
            ax.barh(states, rates, color=colors, edgecolor="white", linewidth=0.8)

            for i, (state, rate) in enumerate(zip(states, rates)):
                ax.text(rate + 0.005, i, f"{rate:.3f}%", va="center", fontsize=8)

            ax.set_xlabel("Fraud Rate (%)", fontsize=11)
            ax.set_title(f"Top {n} States by Fraud Rate", fontsize=13, fontweight="bold")
            ax.invert_yaxis()
            plt.tight_layout()
            return self._save(fig, "05_fraud_by_state.png")
        except VisualizationError:
            raise
        except Exception as exc:
            raise VisualizationError(
                "Failed to plot fraud by state.", {"error": str(exc)}
            ) from exc

    def _plot_correlation_heatmap(
        self,
        df: pd.DataFrame,
        correlation: CorrelationAnalysis | None,
    ) -> str | None:
        """Seaborn heatmap of the numeric feature correlation matrix."""
        if correlation is None or not correlation.numeric_columns_used:
            raise VisualizationError("CorrelationAnalysis is None. Skipping heatmap.", {})
        try:
            self._apply_style()
            cols = [c for c in correlation.numeric_columns_used if c in df.columns]
            if len(cols) < 2:
                raise VisualizationError("Fewer than 2 numeric columns for heatmap.", {})

            corr_matrix = df[cols].corr()
            n = len(cols)
            fig_size = max(8, n * 0.7)
            fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

            mask = None
            import numpy as np
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

            sns.heatmap(
                corr_matrix,
                mask=mask,
                annot=True,
                fmt=".2f",
                cmap="RdYlGn",
                center=0,
                vmin=-1,
                vmax=1,
                linewidths=0.5,
                linecolor="white",
                square=True,
                ax=ax,
                annot_kws={"size": max(6, 10 - n // 3)},
            )
            ax.set_title("Feature Correlation Heatmap (Pearson)", fontsize=13, fontweight="bold")
            plt.tight_layout()
            return self._save(fig, "06_correlation_heatmap.png")
        except VisualizationError:
            raise
        except Exception as exc:
            raise VisualizationError(
                "Failed to plot correlation heatmap.", {"error": str(exc)}
            ) from exc

    def _plot_missing_values_heatmap(self, df: pd.DataFrame) -> str:
        """Heatmap showing presence/absence of null values per column."""
        try:
            self._apply_style()
            missing_cols = [col for col in df.columns if df[col].isna().any()]

            if not missing_cols:
                # No missing values — show a simple summary figure
                fig, ax = plt.subplots(figsize=(7, 3))
                ax.text(
                    0.5, 0.5,
                    "No missing values detected\nacross all columns",
                    transform=ax.transAxes,
                    ha="center", va="center",
                    fontsize=14, color="#2ecc71",
                    fontweight="bold",
                )
                ax.axis("off")
                ax.set_title("Missing Values Heatmap", fontsize=13, fontweight="bold")
                plt.tight_layout()
                return self._save(fig, "07_missing_values_heatmap.png")

            # Sample up to 5000 rows for performance
            sample = df[missing_cols].sample(
                n=min(5000, len(df)), random_state=42
            ).isnull()

            fig, ax = plt.subplots(figsize=(max(8, len(missing_cols) * 0.8), 6))
            sns.heatmap(
                sample,
                cbar=False,
                yticklabels=False,
                cmap=["#2ecc71", "#e74c3c"],
                ax=ax,
            )
            ax.set_title(
                f"Missing Values Heatmap ({len(missing_cols)} column(s) with nulls)",
                fontsize=13, fontweight="bold",
            )
            ax.set_xlabel("Column", fontsize=11)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            return self._save(fig, "07_missing_values_heatmap.png")
        except Exception as exc:
            raise VisualizationError(
                "Failed to plot missing values heatmap.", {"error": str(exc)}
            ) from exc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_style(self) -> None:
        """Applies the configured matplotlib style."""
        try:
            plt.style.use(self._style)
        except OSError:
            plt.style.use("default")

    def _save(self, fig: plt.Figure, filename: str) -> str:
        """
        Saves a figure to disk and closes it to free memory.

        Args:
            fig: The matplotlib Figure to save.
            filename: Filename (e.g., "01_class_distribution.png").

        Returns:
            str: Relative path from project root to the saved file.
        """
        abs_path = self._figures_dir / filename
        fig.savefig(abs_path, dpi=self._dpi, bbox_inches="tight")
        plt.close(fig)

        # Return relative path from the current working directory
        try:
            return str(abs_path.relative_to(Path.cwd()))
        except ValueError:
            # abs_path is not under cwd (e.g., absolute output_dir was given)
            return str(abs_path)
