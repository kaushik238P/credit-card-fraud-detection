"""
Insight Engine for the EDA Layer.

Converts analytical findings from EDA model objects into actionable,
implementation-independent engineering recommendations.

Design rules:
    - Stateless: no mutable state between calls.
    - No model-specific recommendations (no XGBoost, no SMOTE library names).
    - Thresholds are injected at construction time from config.settings.
    - Each _*_recommendations() method is independently testable.
"""

from __future__ import annotations

from config.settings import settings
from src.eda.models import (
    CategoricalAnalysis,
    CorrelationAnalysis,
    EDARecommendation,
    GeographicAnalysis,
    NumericalAnalysis,
    TargetAnalysis,
    TemporalAnalysis,
    FeatureInventory,
)

_PRIORITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


class InsightEngine:
    """
    Rule-based engine that converts EDA findings into recommendations.

    Example:
        >>> engine = InsightEngine()
        >>> recs = engine.generate(target, numerical, categorical, temporal,
        ...                        geographic, correlation)

    Args:
        imbalance_severe_threshold: Imbalance ratio above which recommendations
            escalate to HIGH priority. Default from settings.eda.
        high_cardinality_threshold: Unique-value count above which a column
            gets encoding recommendations. Default from settings.eda.
        skewness_threshold: Absolute skewness above which log-transform is
            recommended. Default from settings.eda.
        outlier_pct_threshold: Outlier percentage above which capping is
            recommended. Default from settings.eda.
        high_correlation_threshold: Pearson r above which multicollinearity
            is flagged. Default from settings.eda.
    """

    def __init__(
        self,
        imbalance_severe_threshold: float | None = None,
        high_cardinality_threshold: int | None = None,
        skewness_threshold: float | None = None,
        outlier_pct_threshold: float | None = None,
        high_correlation_threshold: float | None = None,
    ) -> None:
        cfg = settings.eda
        self._imbalance_severe = imbalance_severe_threshold or cfg.imbalance_severe_threshold
        self._high_cardinality = high_cardinality_threshold or cfg.high_cardinality_threshold
        self._skewness = skewness_threshold or cfg.skewness_threshold
        self._outlier_pct = outlier_pct_threshold or cfg.outlier_pct_threshold
        self._high_corr = high_correlation_threshold or cfg.high_correlation_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        target: TargetAnalysis,
        numerical: NumericalAnalysis | None,
        categorical: CategoricalAnalysis | None,
        temporal: TemporalAnalysis | None,
        geographic: GeographicAnalysis | None,
        correlation: CorrelationAnalysis | None,
        feature_inventory: FeatureInventory | None = None,
        dob_analysis: dict | None = None,
    ) -> tuple[EDARecommendation, ...]:
        """
        Generates all recommendations from the complete set of EDA analyses.

        Deduplicates by (category, recommendation) text and sorts by priority.

        Args:
            target: Target column analysis result.
            numerical: Numerical features analysis, or None.
            categorical: Categorical features analysis, or None.
            temporal: Temporal analysis, or None.
            geographic: Geographic analysis, or None.
            correlation: Correlation analysis, or None.
            feature_inventory: Optional feature inventory structure.
            dob_analysis: Optional parsed DOB results.

        Returns:
            tuple[EDARecommendation, ...]: Sorted recommendations (HIGH first).
        """
        all_recs: list[EDARecommendation] = []
        all_recs.extend(self._imbalance_recommendations(target))
        all_recs.extend(self._numerical_recommendations(numerical))
        all_recs.extend(self._categorical_recommendations(categorical))
        all_recs.extend(self._temporal_recommendations(temporal))
        all_recs.extend(self._geographic_recommendations(geographic))
        all_recs.extend(self._correlation_recommendations(correlation))

        # Add inventory-based recommendations
        if feature_inventory is not None:
            if feature_inventory.drop_before_training:
                all_recs.append(
                    EDARecommendation(
                        category="PREPROCESSING",
                        priority="HIGH",
                        finding=f"Identified columns to drop before training: {list(feature_inventory.drop_before_training)}.",
                        recommendation="Remove identifier and PII columns before training to avoid overfitting, data leakage, and privacy violations.",
                        affects_stage=("PREPROCESSING", "FEATURE_ENGINEERING"),
                    )
                )
            if feature_inventory.temporal:
                all_recs.append(
                    EDARecommendation(
                        category="FEATURE_ENGINEERING",
                        priority="HIGH",
                        finding=f"Temporal columns present: {list(feature_inventory.temporal)}.",
                        recommendation="Extract temporal features (e.g. hour, weekday, weekend indicator, month) from date/time columns.",
                        affects_stage=("FEATURE_ENGINEERING",),
                    )
                )
            if feature_inventory.geographic:
                all_recs.append(
                    EDARecommendation(
                        category="FEATURE_ENGINEERING",
                        priority="HIGH",
                        finding=f"Geographic features present: {list(feature_inventory.geographic)}.",
                        recommendation="Compute spatial metrics such as Haversine distance between customer coordinates and merchant coordinates.",
                        affects_stage=("FEATURE_ENGINEERING",),
                    )
                )
            # High cardinality categorical checks
            high_cards = [
                col for col in feature_inventory.categorical
                if categorical and col in categorical.high_cardinality_columns
            ]
            if high_cards:
                all_recs.append(
                    EDARecommendation(
                        category="ENCODING",
                        priority="HIGH",
                        finding=f"High cardinality categorical columns detected: {high_cards}.",
                        recommendation="Use Target Encoding or Frequency Encoding to represent high-cardinality categories.",
                        affects_stage=("FEATURE_ENGINEERING",),
                    )
                )

        if dob_analysis is not None and dob_analysis.get("parsed", False):
            age_stats = dob_analysis.get("statistics", {})
            mean_age = age_stats.get("mean", 0.0)
            all_recs.append(
                EDARecommendation(
                    category="FEATURE_ENGINEERING",
                    priority="MEDIUM",
                    finding=f"DOB parsed successfully. Average customer age is {mean_age:.1f} years.",
                    recommendation="Engineer age-based features and classify customers into age groups to capture demographic-related risk factors.",
                    affects_stage=("FEATURE_ENGINEERING",),
                )
            )

        # Deduplicate by (category, recommendation) text
        seen: set[tuple[str, str]] = set()
        unique: list[EDARecommendation] = []
        for rec in all_recs:
            key = (rec.category, rec.recommendation)
            if key not in seen:
                seen.add(key)
                unique.append(rec)

        # Sort: HIGH → MEDIUM → LOW, then alphabetically within priority
        unique.sort(key=lambda r: (_PRIORITY_ORDER.get(r.priority, 9), r.category))
        return tuple(unique)

    # ------------------------------------------------------------------
    # Private recommendation generators
    # ------------------------------------------------------------------

    def _imbalance_recommendations(
        self, target: TargetAnalysis
    ) -> list[EDARecommendation]:
        """Generates recommendations based on class distribution."""
        recs: list[EDARecommendation] = []
        ratio = target.imbalance_ratio
        fraud_pct = target.fraud_pct

        # Always recommend metric awareness for any imbalanced dataset
        recs.append(
            EDARecommendation(
                category="MODEL_SELECTION",
                priority="HIGH",
                finding=(
                    f"Dataset is imbalanced: fraud rate = {fraud_pct:.4f}%, "
                    f"imbalance ratio = {ratio:.1f}:1."
                ),
                recommendation=(
                    "Evaluate using Precision-Recall AUC and F1-Score "
                    "instead of accuracy. Accuracy is misleading on imbalanced datasets."
                ),
                affects_stage=("TRAINING",),
            )
        )

        recs.append(
            EDARecommendation(
                category="MODEL_SELECTION",
                priority="HIGH",
                finding=(
                    f"Minority class (fraud) represents {fraud_pct:.4f}% of data."
                ),
                recommendation=(
                    "Use class weights during training to penalise "
                    "misclassification of the minority class."
                ),
                affects_stage=("TRAINING",),
            )
        )

        recs.append(
            EDARecommendation(
                category="MODEL_SELECTION",
                priority="HIGH",
                finding=f"Fraud rate = {fraud_pct:.4f}%.",
                recommendation=(
                    "Use stratified k-fold cross-validation to preserve "
                    "class distribution across all folds."
                ),
                affects_stage=("TRAINING",),
            )
        )

        recs.append(
            EDARecommendation(
                category="MODEL_SELECTION",
                priority="MEDIUM",
                finding=f"Imbalance ratio = {ratio:.1f}:1.",
                recommendation=(
                    "Tune the classification decision threshold on the "
                    "validation set using the precision-recall curve."
                ),
                affects_stage=("TRAINING",),
            )
        )

        if ratio >= self._imbalance_severe:
            recs.append(
                EDARecommendation(
                    category="PREPROCESSING",
                    priority="HIGH",
                    finding=(
                        f"Severe class imbalance detected: ratio = {ratio:.1f}:1 "
                        f"(threshold: {self._imbalance_severe:.0f}:1)."
                    ),
                    recommendation=(
                        "Apply oversampling of the minority class "
                        "(e.g., SMOTE or ADASYN) on the training set only. "
                        "Never apply oversampling to the validation or test set."
                    ),
                    affects_stage=("PREPROCESSING", "TRAINING"),
                )
            )

        return recs

    def _numerical_recommendations(
        self, numerical: NumericalAnalysis | None
    ) -> list[EDARecommendation]:
        """Generates recommendations based on skewness and outliers."""
        if numerical is None:
            return []

        recs: list[EDARecommendation] = []

        if numerical.high_skewness_columns:
            cols = list(numerical.high_skewness_columns)
            recs.append(
                EDARecommendation(
                    category="PREPROCESSING",
                    priority="HIGH",
                    finding=(
                        f"{len(cols)} column(s) have high skewness "
                        f"(|skewness| > {self._skewness}): {cols}."
                    ),
                    recommendation=(
                        "Apply log or power transformation to reduce skewness "
                        "before feeding into distance-based or linear models."
                    ),
                    affects_stage=("PREPROCESSING",),
                )
            )

        if numerical.high_outlier_columns:
            cols = list(numerical.high_outlier_columns)
            recs.append(
                EDARecommendation(
                    category="PREPROCESSING",
                    priority="MEDIUM",
                    finding=(
                        f"{len(cols)} column(s) have outlier rate > "
                        f"{self._outlier_pct}%: {cols}."
                    ),
                    recommendation=(
                        "Apply Winsorization or clipping to cap extreme values. "
                        "Use RobustScaler (median/IQR) instead of StandardScaler."
                    ),
                    affects_stage=("PREPROCESSING",),
                )
            )

        if numerical.columns_analysed:
            recs.append(
                EDARecommendation(
                    category="PREPROCESSING",
                    priority="MEDIUM",
                    finding=f"{len(numerical.columns_analysed)} numeric column(s) at different scales.",
                    recommendation=(
                        "Scale numeric features before training. Prefer "
                        "RobustScaler over StandardScaler when outliers are present."
                    ),
                    affects_stage=("PREPROCESSING",),
                )
            )

        # Check for amt-specific finding (transaction amount is a key fraud signal)
        if "amt" in numerical.summaries:
            amt = numerical.summaries["amt"]
            fraud_mean = amt.get("fraud_mean")
            legit_mean = amt.get("legit_mean")
            if fraud_mean is not None and legit_mean is not None and legit_mean > 0:
                ratio = round(fraud_mean / legit_mean, 2)
                if abs(ratio - 1.0) > 0.2:
                    recs.append(
                        EDARecommendation(
                            category="FEATURE_ENGINEERING",
                            priority="HIGH",
                            finding=(
                                f"Transaction amount differs between fraud "
                                f"(mean={fraud_mean:.2f}) and legit (mean={legit_mean:.2f}), "
                                f"ratio={ratio:.2f}."
                            ),
                            recommendation=(
                                "Transaction amount is a strong signal. "
                                "Engineer amount-based features: log(amt+1), "
                                "amount percentile rank, and per-cardholder rolling "
                                "amount statistics."
                            ),
                            affects_stage=("FEATURE_ENGINEERING",),
                        )
                    )

        return recs

    def _categorical_recommendations(
        self, categorical: CategoricalAnalysis | None
    ) -> list[EDARecommendation]:
        """Generates recommendations based on cardinality."""
        if categorical is None:
            return []

        recs: list[EDARecommendation] = []
        high_card = list(categorical.high_cardinality_columns)

        if high_card:
            recs.append(
                EDARecommendation(
                    category="ENCODING",
                    priority="HIGH",
                    finding=(
                        f"{len(high_card)} high-cardinality column(s) detected "
                        f"(> {self._high_cardinality} unique values): {high_card}."
                    ),
                    recommendation=(
                        "Use Target Encoding or Frequency Encoding for high-cardinality "
                        "columns. One-Hot Encoding will produce too many sparse columns."
                    ),
                    affects_stage=("FEATURE_ENGINEERING",),
                )
            )

        # Find low-cardinality columns (suitable for one-hot)
        low_card = [
            col for col in categorical.columns_analysed
            if col not in categorical.high_cardinality_columns
            and categorical.summaries.get(col, {}).get("unique_count", 0) <= 20
        ]
        if low_card:
            recs.append(
                EDARecommendation(
                    category="ENCODING",
                    priority="LOW",
                    finding=(
                        f"{len(low_card)} low-cardinality column(s) "
                        f"(<= 20 unique values): {low_card}."
                    ),
                    recommendation=(
                        "One-Hot Encoding is appropriate for low-cardinality columns."
                    ),
                    affects_stage=("FEATURE_ENGINEERING",),
                )
            )

        # Columns that are likely PII with no predictive value
        drop_candidates = [
            col for col in categorical.columns_analysed
            if col in ("first", "last", "street", "trans_num")
        ]
        if drop_candidates:
            recs.append(
                EDARecommendation(
                    category="FEATURE_ENGINEERING",
                    priority="HIGH",
                    finding=(
                        f"PII columns with no predictive value detected: "
                        f"{drop_candidates}."
                    ),
                    recommendation=(
                        "Drop PII columns (first name, last name, street, "
                        "transaction UUID) before training. They add noise and raise "
                        "privacy concerns."
                    ),
                    affects_stage=("FEATURE_ENGINEERING",),
                )
            )

        return recs

    def _temporal_recommendations(
        self, temporal: TemporalAnalysis | None
    ) -> list[EDARecommendation]:
        """Generates recommendations based on temporal patterns."""
        if temporal is None:
            return []

        recs: list[EDARecommendation] = []

        recs.append(
            EDARecommendation(
                category="FEATURE_ENGINEERING",
                priority="HIGH",
                finding=(
                    f"Datetime column '{temporal.datetime_column_used}' is available. "
                    f"Peak fraud hour: {temporal.peak_fraud_hour}:00, "
                    f"peak fraud day: {temporal.peak_fraud_weekday}."
                ),
                recommendation=(
                    "Engineer time-based features: hour_of_day, day_of_week, "
                    "month, is_weekend, is_night_hour (22:00-06:00). "
                    "These encode the temporal fraud pattern directly."
                ),
                affects_stage=("FEATURE_ENGINEERING",),
            )
        )

        recs.append(
            EDARecommendation(
                category="FEATURE_ENGINEERING",
                priority="MEDIUM",
                finding="Unix timestamp and datetime column available.",
                recommendation=(
                    "Engineer recency features per cardholder: days since "
                    "first transaction, days since last transaction, "
                    "transaction velocity over rolling windows."
                ),
                affects_stage=("FEATURE_ENGINEERING",),
            )
        )

        return recs

    def _geographic_recommendations(
        self, geographic: GeographicAnalysis | None
    ) -> list[EDARecommendation]:
        """Generates recommendations based on geographic patterns."""
        if geographic is None:
            return []

        recs: list[EDARecommendation] = []
        top_states = geographic.top_fraud_states[:3] if geographic.top_fraud_states else []

        recs.append(
            EDARecommendation(
                category="FEATURE_ENGINEERING",
                priority="HIGH",
                finding=(
                    f"Geographic fraud variation detected across "
                    f"{geographic.total_states} states. "
                    f"Top fraud states: {top_states}."
                ),
                recommendation=(
                    "Engineer geographic features: "
                    "state fraud rate (target-encoded), "
                    "cardholder-to-merchant distance (Haversine), "
                    "city population (already a column). "
                    "Distance from home is a strong fraud signal."
                ),
                affects_stage=("FEATURE_ENGINEERING",),
            )
        )

        return recs

    def _correlation_recommendations(
        self, correlation: CorrelationAnalysis | None
    ) -> list[EDARecommendation]:
        """Generates recommendations based on feature correlations."""
        if correlation is None:
            return []

        recs: list[EDARecommendation] = []
        high_pairs = correlation.high_correlation_pairs

        if high_pairs:
            pair_names = [(p["col_a"], p["col_b"]) for p in high_pairs[:5]]
            recs.append(
                EDARecommendation(
                    category="FEATURE_ENGINEERING",
                    priority="MEDIUM",
                    finding=(
                        f"{len(high_pairs)} highly correlated feature pair(s) "
                        f"detected (|r| > {self._high_corr}): {pair_names}."
                    ),
                    recommendation=(
                        "Consider dropping one feature from each highly correlated "
                        "pair to reduce multicollinearity and model complexity."
                    ),
                    affects_stage=("FEATURE_ENGINEERING", "PREPROCESSING"),
                )
            )

        target_corrs = correlation.target_correlations
        if isinstance(target_corrs, dict) and "pearson" in target_corrs:
            target_corrs = target_corrs["pearson"]

        strong_target = [
            col for col, r in target_corrs.items()
            if abs(r) > 0.1 and col != "is_fraud"
        ]
        if strong_target:
            recs.append(
                EDARecommendation(
                    category="FEATURE_ENGINEERING",
                    priority="HIGH",
                    finding=(
                        f"{len(strong_target)} feature(s) have meaningful "
                        f"linear correlation with the target (|r| > 0.1): {strong_target}."
                    ),
                    recommendation=(
                        "Prioritise these features in the initial model feature set. "
                        "They show direct linear signal for fraud detection."
                    ),
                    affects_stage=("FEATURE_ENGINEERING",),
                )
            )

        return recs
