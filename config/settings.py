"""
Application-wide settings for the Credit Card Fraud Detection System.

Settings are loaded from environment variables with sane defaults, making
this configuration Docker-friendly and suitable for CI/CD pipelines.

Environment variables are prefixed with FRAUD_ to avoid collisions.

Usage:
    from config.settings import settings

    encoding = settings.ingestion.encoding
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Ingestion sub-settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IngestionSettings:
    """
    Configuration for the data ingestion layer.

    All values are read from environment variables at import time.
    Defaults are chosen to be safe for fraud detection workloads.

    Attributes:
        encoding: Character encoding used when reading text-based files.
            Override with FRAUD_INGESTION_ENCODING.
        low_memory: When False, pandas infers dtypes using the entire column
            rather than per-chunk, which is safer for mixed-type fraud data.
            Override with FRAUD_INGESTION_LOW_MEMORY (0 or 1).
        supported_extensions: Comma-separated list of allowed file extensions.
            Override with FRAUD_INGESTION_SUPPORTED_EXTENSIONS.
        hash_chunk_size_bytes: Chunk size used when computing SHA-256 hashes
            for large files without loading them fully into memory.
            Override with FRAUD_INGESTION_HASH_CHUNK_SIZE.
        loaded_by: Optional identifier stamped into DatasetMetadata.
            Useful for audit trails in multi-user or multi-service deployments.
            Override with FRAUD_INGESTION_LOADED_BY.
        target_column: Default target/label column name.
            Override with FRAUD_INGESTION_TARGET_COLUMN.
        zip_csv_target: Optional inner filename to extract from a ZIP archive.
            When a ZIP contains multiple CSV files, this setting pins which one
            to load. When None the loader auto-selects the single CSV found,
            or raises an error if more than one exists.
            Override with FRAUD_INGESTION_ZIP_CSV_TARGET.
    """

    encoding: str = field(
        default_factory=lambda: os.environ.get("FRAUD_INGESTION_ENCODING", "utf-8")
    )
    low_memory: bool = field(
        default_factory=lambda: os.environ.get(
            "FRAUD_INGESTION_LOW_MEMORY", "0"
        ) == "1"
    )
    supported_extensions: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            ext.strip()
            for ext in os.environ.get(
                "FRAUD_INGESTION_SUPPORTED_EXTENSIONS", ".csv,.zip"
            ).split(",")
        )
    )
    zip_csv_target: str | None = field(
        default_factory=lambda: os.environ.get("FRAUD_INGESTION_ZIP_CSV_TARGET") or None
    )
    hash_chunk_size_bytes: int = field(
        default_factory=lambda: int(
            os.environ.get("FRAUD_INGESTION_HASH_CHUNK_SIZE", str(8 * 1024 * 1024))
        )
    )
    loaded_by: str | None = field(
        default_factory=lambda: os.environ.get("FRAUD_INGESTION_LOADED_BY") or None
    )
    target_column: str | None = field(
        default_factory=lambda: os.environ.get("FRAUD_INGESTION_TARGET_COLUMN") or None
    )


# ---------------------------------------------------------------------------
# Logging sub-settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoggingSettings:
    """
    Configuration for the centralized logging system.

    Attributes:
        level: Root log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            Override with FRAUD_LOG_LEVEL.
        format: Log format string (text or json).
            Override with FRAUD_LOG_FORMAT (choices: "text", "json").
        date_format: Timestamp format for text-based log lines.
            Override with FRAUD_LOG_DATE_FORMAT.
    """

    level: str = field(
        default_factory=lambda: os.environ.get("FRAUD_LOG_LEVEL", "INFO").upper()
    )
    format: str = field(
        default_factory=lambda: os.environ.get("FRAUD_LOG_FORMAT", "text").lower()
    )
    date_format: str = field(
        default_factory=lambda: os.environ.get(
            "FRAUD_LOG_DATE_FORMAT", "%Y-%m-%dT%H:%M:%S"
        )
    )


# ---------------------------------------------------------------------------
# EDA sub-settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EDASettings:
    """
    Configuration for the Exploratory Data Analysis layer.

    All values are read from environment variables at import time.

    Attributes:
        high_cardinality_threshold: Columns with more unique values than this
            are flagged as high-cardinality and get encoding recommendations.
            Override with FRAUD_EDA_HIGH_CARDINALITY_THRESHOLD.
        outlier_pct_threshold: Percentage of outlier rows above which a column
            gets a capping/Winsorization recommendation.
            Override with FRAUD_EDA_OUTLIER_THRESHOLD.
        skewness_threshold: Absolute skewness above which a column gets a
            log-transform recommendation.
            Override with FRAUD_EDA_SKEWNESS_THRESHOLD.
        high_correlation_threshold: Absolute Pearson r above which a feature
            pair is flagged as highly correlated.
            Override with FRAUD_EDA_HIGH_CORRELATION_THRESHOLD.
        imbalance_severe_threshold: Imbalance ratio (majority/minority) above
            which class imbalance is classified as severe.
            Override with FRAUD_EDA_IMBALANCE_SEVERE_THRESHOLD.
        figure_dpi: Resolution in DPI for saved figures.
            Override with FRAUD_EDA_FIGURE_DPI.
        figure_style: Matplotlib style name used for all plots.
            Override with FRAUD_EDA_FIGURE_STYLE.
        top_n_categories: Number of top categories to show in bar charts.
            Override with FRAUD_EDA_TOP_N_CATEGORIES.
        output_dir: Root output directory for EDA reports and figures.
            Override with FRAUD_EDA_OUTPUT_DIR.
    """

    high_cardinality_threshold: int = field(
        default_factory=lambda: int(
            os.environ.get("FRAUD_EDA_HIGH_CARDINALITY_THRESHOLD", "50")
        )
    )
    outlier_pct_threshold: float = field(
        default_factory=lambda: float(
            os.environ.get("FRAUD_EDA_OUTLIER_THRESHOLD", "5.0")
        )
    )
    skewness_threshold: float = field(
        default_factory=lambda: float(
            os.environ.get("FRAUD_EDA_SKEWNESS_THRESHOLD", "1.0")
        )
    )
    high_correlation_threshold: float = field(
        default_factory=lambda: float(
            os.environ.get("FRAUD_EDA_HIGH_CORRELATION_THRESHOLD", "0.8")
        )
    )
    imbalance_severe_threshold: float = field(
        default_factory=lambda: float(
            os.environ.get("FRAUD_EDA_IMBALANCE_SEVERE_THRESHOLD", "20.0")
        )
    )
    figure_dpi: int = field(
        default_factory=lambda: int(
            os.environ.get("FRAUD_EDA_FIGURE_DPI", "150")
        )
    )
    figure_style: str = field(
        default_factory=lambda: os.environ.get(
            "FRAUD_EDA_FIGURE_STYLE", "seaborn-v0_8-whitegrid"
        )
    )
    top_n_categories: int = field(
        default_factory=lambda: int(
            os.environ.get("FRAUD_EDA_TOP_N_CATEGORIES", "20")
        )
    )
    output_dir: str = field(
        default_factory=lambda: os.environ.get("FRAUD_EDA_OUTPUT_DIR", "reports/eda")
    )
    min_category_support: int = field(
        default_factory=lambda: int(
            os.environ.get("FRAUD_EDA_MIN_CATEGORY_SUPPORT", "100")
        )
    )
    identifier_patterns: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            pat.strip()
            for pat in os.environ.get(
                "FRAUD_EDA_IDENTIFIER_PATTERNS", "Unnamed:, cc_num, trans_num, uuid, id"
            ).split(",")
        )
    )
    pii_columns: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            col.strip()
            for col in os.environ.get(
                "FRAUD_EDA_PII_COLUMNS", "first, last, street, email, phone, trans_num"
            ).split(",")
        )
    )
    dob_patterns: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            pat.strip()
            for pat in os.environ.get(
                "FRAUD_EDA_DOB_PATTERNS", "dob, date_of_birth, birth_date"
            ).split(",")
        )
    )


# ---------------------------------------------------------------------------
# Feature Engineering sub-settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FESettings:
    """
    Configuration for the Feature Engineering Layer.

    All values are read from environment variables at import time.
    No magic numbers exist in feature modules — all thresholds live here.

    Attributes:
        high_value_threshold: Amount above which a transaction is flagged as
            high-value. Override with FRAUD_FE_HIGH_VALUE_THRESHOLD.
        far_transaction_threshold_km: Haversine distance (km) above which a
            transaction is flagged as far. Override with
            FRAUD_FE_FAR_TRANSACTION_THRESHOLD_KM.
        max_one_hot_cardinality: Maximum unique values for ONE_HOT encoding
            recommendation. Override with FRAUD_FE_MAX_ONE_HOT_CARDINALITY.
        enable_behavioral: Whether behavioral features are computed.
            Override with FRAUD_FE_ENABLE_BEHAVIORAL (0 or 1).
        age_bins: Comma-separated bin edges for age_group feature.
            Override with FRAUD_FE_AGE_BINS.
        age_bin_labels: Comma-separated labels for each age_group bin.
            Override with FRAUD_FE_AGE_BIN_LABELS.
        frequency_columns: Comma-separated columns to compute frequency for.
            Override with FRAUD_FE_FREQUENCY_COLUMNS.
        output_dir: Directory for feature engineering reports.
            Override with FRAUD_FE_OUTPUT_DIR.
    """

    high_value_threshold: float = field(
        default_factory=lambda: float(
            os.environ.get("FRAUD_FE_HIGH_VALUE_THRESHOLD", "500.0")
        )
    )
    far_transaction_threshold_km: float = field(
        default_factory=lambda: float(
            os.environ.get("FRAUD_FE_FAR_TRANSACTION_THRESHOLD_KM", "100.0")
        )
    )
    max_one_hot_cardinality: int = field(
        default_factory=lambda: int(
            os.environ.get("FRAUD_FE_MAX_ONE_HOT_CARDINALITY", "10")
        )
    )
    enable_behavioral: bool = field(
        default_factory=lambda: os.environ.get(
            "FRAUD_FE_ENABLE_BEHAVIORAL", "0"
        ) == "1"
    )
    age_bins: tuple[float, ...] = field(
        default_factory=lambda: tuple(
            float(v.strip())
            for v in os.environ.get(
                "FRAUD_FE_AGE_BINS", "0,18,25,35,45,55,65,120"
            ).split(",")
        )
    )
    age_bin_labels: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            v.strip()
            for v in os.environ.get(
                "FRAUD_FE_AGE_BIN_LABELS", "<18,18-25,26-35,36-45,46-55,56-65,66+"
            ).split(",")
        )
    )
    frequency_columns: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            v.strip()
            for v in os.environ.get(
                "FRAUD_FE_FREQUENCY_COLUMNS",
                "merchant,category,job,state,city,gender",
            ).split(",")
        )
    )
    output_dir: str = field(
        default_factory=lambda: os.environ.get(
            "FRAUD_FE_OUTPUT_DIR", "reports/features"
        )
    )


# ---------------------------------------------------------------------------
# Preprocessing sub-settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreprocessingSettings:
    """
    Configuration for the Preprocessing Layer.

    All values are read from environment variables at import time.
    No magic numbers exist in preprocessing modules — all values live here.

    Attributes:
        datetime_column: Transaction timestamp column used for time-based
            splitting and then dropped from feature matrices.
            Override with FRAUD_PP_DATETIME_COLUMN.
        target_column: Binary target column name.
            Override with FRAUD_PP_TARGET_COLUMN.
        train_end_date: Last date (inclusive) for the training fold.
            Override with FRAUD_PP_TRAIN_END_DATE.
        validation_start_date: First date (inclusive) for validation fold.
            Override with FRAUD_PP_VALIDATION_START_DATE.
        validation_end_date: Last date (inclusive) for validation fold.
            Override with FRAUD_PP_VALIDATION_END_DATE.
        test_start_date: First date (inclusive) for test fold.
            Override with FRAUD_PP_TEST_START_DATE.
        test_end_date: Last date (inclusive) for test fold.
            Override with FRAUD_PP_TEST_END_DATE.
        enable_scaling: Whether to apply RobustScaler. False by default
            (tree-first design). Override with FRAUD_PP_ENABLE_SCALING.
        ohe_drop_first: Whether OHE drops the first dummy column.
            Override with FRAUD_PP_OHE_DROP_FIRST.
        ohe_handle_unknown: OHE behaviour for unseen categories.
            Override with FRAUD_PP_OHE_HANDLE_UNKNOWN.
        frequency_unknown_value: Integer assigned to unseen frequency categories.
            Override with FRAUD_PP_FREQ_UNKNOWN_VALUE.
        source_columns_to_drop: Comma-separated source columns to drop before
            training. These have served their purpose in feature engineering.
            Override with FRAUD_PP_SOURCE_COLUMNS_TO_DROP.
        output_dir: Directory for Parquet output files.
            Override with FRAUD_PP_OUTPUT_DIR.
        artifact_dir: Root directory for preprocessing artifacts.
            Override with FRAUD_PP_ARTIFACT_DIR.
        parquet_compression: Parquet compression codec.
            Override with FRAUD_PP_PARQUET_COMPRESSION.
    """

    datetime_column: str = field(
        default_factory=lambda: os.environ.get(
            "FRAUD_PP_DATETIME_COLUMN", "trans_date_trans_time"
        )
    )
    target_column: str = field(
        default_factory=lambda: os.environ.get("FRAUD_PP_TARGET_COLUMN", "is_fraud")
    )
    train_end_date: str = field(
        default_factory=lambda: os.environ.get("FRAUD_PP_TRAIN_END_DATE", "2020-02-29")
    )
    validation_start_date: str = field(
        default_factory=lambda: os.environ.get(
            "FRAUD_PP_VALIDATION_START_DATE", "2020-03-01"
        )
    )
    validation_end_date: str = field(
        default_factory=lambda: os.environ.get(
            "FRAUD_PP_VALIDATION_END_DATE", "2020-04-30"
        )
    )
    test_start_date: str = field(
        default_factory=lambda: os.environ.get("FRAUD_PP_TEST_START_DATE", "2020-05-01")
    )
    test_end_date: str = field(
        default_factory=lambda: os.environ.get("FRAUD_PP_TEST_END_DATE", "2020-06-30")
    )
    enable_scaling: bool = field(
        default_factory=lambda: os.environ.get("FRAUD_PP_ENABLE_SCALING", "0") == "1"
    )
    ohe_drop_first: bool = field(
        default_factory=lambda: os.environ.get("FRAUD_PP_OHE_DROP_FIRST", "0") == "1"
    )
    ohe_handle_unknown: str = field(
        default_factory=lambda: os.environ.get("FRAUD_PP_OHE_HANDLE_UNKNOWN", "ignore")
    )
    frequency_unknown_value: int = field(
        default_factory=lambda: int(
            os.environ.get("FRAUD_PP_FREQ_UNKNOWN_VALUE", "0")
        )
    )
    source_columns_to_drop: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            v.strip()
            for v in os.environ.get(
                "FRAUD_PP_SOURCE_COLUMNS_TO_DROP",
                "dob,trans_date_trans_time,unix_time",
            ).split(",")
        )
    )
    output_dir: str = field(
        default_factory=lambda: os.environ.get(
            "FRAUD_PP_OUTPUT_DIR", "data/preprocessed"
        )
    )
    artifact_dir: str = field(
        default_factory=lambda: os.environ.get(
            "FRAUD_PP_ARTIFACT_DIR", "artifacts/preprocessing"
        )
    )
    parquet_compression: str = field(
        default_factory=lambda: os.environ.get("FRAUD_PP_PARQUET_COMPRESSION", "snappy")
    )
    dataset_schema_filename: str = field(
        default_factory=lambda: os.environ.get(
            "FRAUD_PP_DATASET_SCHEMA_FILENAME", "dataset_schema.json"
        )
    )
    class_distribution_filename: str = field(
        default_factory=lambda: os.environ.get(
            "FRAUD_PP_CLASS_DISTRIBUTION_FILENAME", "class_distribution.json"
        )
    )
    data_integrity_filename: str = field(
        default_factory=lambda: os.environ.get(
            "FRAUD_PP_DATA_INTEGRITY_FILENAME", "data_integrity.json"
        )
    )
    manifest_filename: str = field(
        default_factory=lambda: os.environ.get(
            "FRAUD_PP_MANIFEST_FILENAME", "manifest.json"
        )
    )
    preprocessing_version: str = field(
        default_factory=lambda: os.environ.get("FRAUD_PP_VERSION", "1.0.0")
    )

# ---------------------------------------------------------------------------
# Training sub-settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingSettings:
    """
    Configuration for the Training Layer.

    All values are read from environment variables at import time.
    No magic numbers exist in training modules — all values live here.
    Override any field by setting the corresponding FRAUD_TR_* env var.
    """

    # ── Pipeline control ─────────────────────────────────────────────────
    active_model: str = field(
        default_factory=lambda: os.environ.get("FRAUD_TR_ACTIVE_MODEL", "XGBOOST")
    )
    random_seed: int = field(
        default_factory=lambda: int(os.environ.get("FRAUD_TR_RANDOM_SEED", "42"))
    )
    n_jobs: int = field(
        default_factory=lambda: int(os.environ.get("FRAUD_TR_N_JOBS", "-1"))
    )
    imbalance_strategy: str = field(
        default_factory=lambda: os.environ.get(
            "FRAUD_TR_IMBALANCE_STRATEGY", "SCALE_POS_WEIGHT"
        )
    )
    early_stopping_rounds: int | None = field(
        default_factory=lambda: (
            int(os.environ["FRAUD_TR_EARLY_STOPPING_ROUNDS"])
            if "FRAUD_TR_EARLY_STOPPING_ROUNDS" in os.environ
            else None
        )
    )
    training_version: str = field(
        default_factory=lambda: os.environ.get("FRAUD_TR_VERSION", "1.0.0")
    )
    artifact_dir: str = field(
        default_factory=lambda: os.environ.get(
            "FRAUD_TR_ARTIFACT_DIR", "artifacts/models"
        )
    )

    # ── Logistic Regression ──────────────────────────────────────────────
    lr_C: float = field(
        default_factory=lambda: float(os.environ.get("FRAUD_TR_LR_C", "1.0"))
    )
    lr_penalty: str = field(
        default_factory=lambda: os.environ.get("FRAUD_TR_LR_PENALTY", "l2")
    )
    lr_solver: str = field(
        default_factory=lambda: os.environ.get("FRAUD_TR_LR_SOLVER", "lbfgs")
    )
    lr_max_iter: int = field(
        default_factory=lambda: int(os.environ.get("FRAUD_TR_LR_MAX_ITER", "1000"))
    )

    # ── Random Forest ────────────────────────────────────────────────────
    rf_n_estimators: int = field(
        default_factory=lambda: int(os.environ.get("FRAUD_TR_RF_N_ESTIMATORS", "300"))
    )
    rf_max_depth: int = field(
        default_factory=lambda: int(os.environ.get("FRAUD_TR_RF_MAX_DEPTH", "20"))
    )
    rf_min_samples_leaf: int = field(
        default_factory=lambda: int(
            os.environ.get("FRAUD_TR_RF_MIN_SAMPLES_LEAF", "4")
        )
    )

    # ── XGBoost ──────────────────────────────────────────────────────────
    xgb_n_estimators: int = field(
        default_factory=lambda: int(
            os.environ.get("FRAUD_TR_XGB_N_ESTIMATORS", "500")
        )
    )
    xgb_learning_rate: float = field(
        default_factory=lambda: float(
            os.environ.get("FRAUD_TR_XGB_LEARNING_RATE", "0.05")
        )
    )
    xgb_max_depth: int = field(
        default_factory=lambda: int(os.environ.get("FRAUD_TR_XGB_MAX_DEPTH", "6"))
    )
    xgb_subsample: float = field(
        default_factory=lambda: float(
            os.environ.get("FRAUD_TR_XGB_SUBSAMPLE", "0.8")
        )
    )
    xgb_colsample_bytree: float = field(
        default_factory=lambda: float(
            os.environ.get("FRAUD_TR_XGB_COLSAMPLE_BYTREE", "0.8")
        )
    )
    xgb_tree_method: str = field(
        default_factory=lambda: os.environ.get("FRAUD_TR_XGB_TREE_METHOD", "hist")
    )
    xgb_eval_metric: str = field(
        default_factory=lambda: os.environ.get("FRAUD_TR_XGB_EVAL_METRIC", "aucpr")
    )

    # ── LightGBM ─────────────────────────────────────────────────────────
    lgbm_num_leaves: int = field(
        default_factory=lambda: int(
            os.environ.get("FRAUD_TR_LGBM_NUM_LEAVES", "63")
        )
    )
    lgbm_learning_rate: float = field(
        default_factory=lambda: float(
            os.environ.get("FRAUD_TR_LGBM_LEARNING_RATE", "0.05")
        )
    )
    lgbm_n_estimators: int = field(
        default_factory=lambda: int(
            os.environ.get("FRAUD_TR_LGBM_N_ESTIMATORS", "500")
        )
    )
    lgbm_min_child_samples: int = field(
        default_factory=lambda: int(
            os.environ.get("FRAUD_TR_LGBM_MIN_CHILD_SAMPLES", "20")
        )
    )

    # ── CatBoost ─────────────────────────────────────────────────────────
    cat_iterations: int = field(
        default_factory=lambda: int(
            os.environ.get("FRAUD_TR_CAT_ITERATIONS", "500")
        )
    )
    cat_depth: int = field(
        default_factory=lambda: int(os.environ.get("FRAUD_TR_CAT_DEPTH", "6"))
    )
    cat_learning_rate: float = field(
        default_factory=lambda: float(
            os.environ.get("FRAUD_TR_CAT_LEARNING_RATE", "0.05")
        )
    )
    cat_verbose: int = field(
        default_factory=lambda: int(os.environ.get("FRAUD_TR_CAT_VERBOSE", "0"))
    )


# ---------------------------------------------------------------------------
# Evaluation sub-settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationSettings:
    """
    Configuration for the Evaluation Layer.
    """

    primary_metric: str = field(
        default_factory=lambda: os.environ.get("FRAUD_EV_PRIMARY_METRIC", "PR_AUC")
    )
    default_threshold: float = field(
        default_factory=lambda: float(os.environ.get("FRAUD_EV_DEFAULT_THRESHOLD", "0.5"))
    )
    threshold_strategy: str = field(
        default_factory=lambda: os.environ.get("FRAUD_EV_THRESHOLD_STRATEGY", "MAX_F1")
    )
    top_k: int = field(
        default_factory=lambda: int(os.environ.get("FRAUD_EV_TOP_K", "100"))
    )
    plot_dpi: int = field(
        default_factory=lambda: int(os.environ.get("FRAUD_EV_PLOT_DPI", "100"))
    )
    feature_importance_top_n: int = field(
        default_factory=lambda: int(os.environ.get("FRAUD_EV_FEATURE_IMPORTANCE_TOP_N", "15"))
    )
    output_directory: str = field(
        default_factory=lambda: os.environ.get("FRAUD_EV_OUTPUT_DIRECTORY", "reports/evaluation")
    )
    target_recall: float = field(
        default_factory=lambda: float(os.environ.get("FRAUD_EV_TARGET_RECALL", "0.8"))
    )
    target_precision: float = field(
        default_factory=lambda: float(os.environ.get("FRAUD_EV_TARGET_PRECISION", "0.8"))
    )
    evaluation_version: str = field(
        default_factory=lambda: os.environ.get("FRAUD_EV_VERSION", "1.0.0")
    )


# ---------------------------------------------------------------------------
# Root settings object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    """
    Root settings object for the entire application.

    Access sub-settings via attribute namespacing:
        settings.ingestion.encoding
        settings.logging.level
        settings.eda.figure_dpi
        settings.fe.high_value_threshold
        settings.preprocessing.train_end_date
        settings.training.active_model
        settings.evaluation.primary_metric

    This object is instantiated once at module import and reused everywhere.
    It is immutable (frozen=True) to prevent accidental mutation at runtime.
    """

    ingestion: IngestionSettings = field(default_factory=IngestionSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    eda: EDASettings = field(default_factory=EDASettings)
    fe: FESettings = field(default_factory=FESettings)
    preprocessing: PreprocessingSettings = field(default_factory=PreprocessingSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)


# Module-level singleton — import this everywhere.
settings: Settings = Settings()


