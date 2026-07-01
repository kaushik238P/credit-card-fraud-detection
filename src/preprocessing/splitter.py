"""
Time-based dataset splitter for the Preprocessing Layer.

Uses trans_date_trans_time to partition the dataset into non-overlapping
train / validation / test folds. All date boundaries are read from
config.settings — no magic numbers anywhere in this module.
"""

from __future__ import annotations

import logging
import time

import pandas as pd

from src.preprocessing.exceptions import SplitError
from src.preprocessing.models import DatasetSplit

logger = logging.getLogger(__name__)


class TimeBasedSplitter:
    """
    Partitions a DataFrame into train / validation / test folds using
    transaction timestamps.

    Why time-based splitting:
        Random splits allow future transactions to contaminate training-fold
        statistics (scaler parameters, frequency maps). Time-based splitting
        ensures every statistic derived from the training fold represents only
        information that existed before the validation period — exactly as in
        production.

    Args:
        train_end_date: Last date (inclusive) for the training fold.
            Format: "YYYY-MM-DD". Read from config.settings.preprocessing.
        validation_start_date: First date (inclusive) for the validation fold.
        validation_end_date: Last date (inclusive) for the validation fold.
        test_start_date: First date (inclusive) for the test fold.
        test_end_date: Last date (inclusive) for the test fold.
        datetime_column: Column name containing transaction timestamps.
        target_column: Name of the binary target column.
    """

    def __init__(
        self,
        train_end_date: str,
        validation_start_date: str,
        validation_end_date: str,
        test_start_date: str,
        test_end_date: str,
        datetime_column: str,
        target_column: str,
    ) -> None:
        self._train_end = pd.Timestamp(train_end_date)
        self._val_start = pd.Timestamp(validation_start_date)
        self._val_end = pd.Timestamp(validation_end_date)
        self._test_start = pd.Timestamp(test_start_date)
        self._test_end = pd.Timestamp(test_end_date)
        self._datetime_col = datetime_column
        self._target_col = target_column

        self._validate_config()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(
        self,
        df: pd.DataFrame,
    ) -> tuple[DatasetSplit, DatasetSplit, DatasetSplit]:
        """
        Splits ``df`` into three temporally ordered, non-overlapping folds.

        Args:
            df: Input DataFrame. Must contain the datetime column and target
                column specified in the constructor.

        Returns:
            tuple: (train_split, validation_split, test_split) as DatasetSplit
                objects.

        Raises:
            SplitError: If the datetime column is absent, unparseable, or if
                any resulting fold is empty.
        """
        _start = time.perf_counter()
        logger.info(
            "Time-based split started | rows=%d | train_end=%s | val=%s→%s | test=%s→%s",
            len(df),
            self._train_end.date(),
            self._val_start.date(),
            self._val_end.date(),
            self._test_start.date(),
            self._test_end.date(),
        )

        dt_series = self._parse_datetime(df)

        train_mask = dt_series <= self._train_end
        val_mask = (dt_series >= self._val_start) & (dt_series <= self._val_end)
        test_mask = (dt_series >= self._test_start) & (dt_series <= self._test_end)

        train_df = df[train_mask].copy()
        val_df = df[val_mask].copy()
        test_df = df[test_mask].copy()

        for fold_name, fold_df, mask in (
            ("train", train_df, train_mask),
            ("validation", val_df, val_mask),
            ("test", test_df, test_mask),
        ):
            if len(fold_df) == 0:
                raise SplitError(
                    detail=f"Fold '{fold_name}' is empty after date filtering.",
                    fold_name=fold_name,
                )

        train_split = self._build_split("train", train_df, dt_series[train_mask])
        val_split = self._build_split("validation", val_df, dt_series[val_mask])
        test_split = self._build_split("test", test_df, dt_series[test_mask])

        duration_ms = (time.perf_counter() - _start) * 1_000
        logger.info(
            "Split complete | train=%d rows | val=%d rows | test=%d rows | duration=%.1fms",
            train_split.num_rows,
            val_split.num_rows,
            test_split.num_rows,
            duration_ms,
        )

        return train_split, val_split, test_split

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        """Validates that configured date boundaries are logically consistent."""
        if self._train_end >= self._val_start:
            raise SplitError(
                detail=(
                    f"train_end_date ({self._train_end.date()}) must be before "
                    f"validation_start_date ({self._val_start.date()})."
                )
            )
        if self._val_end >= self._test_start:
            raise SplitError(
                detail=(
                    f"validation_end_date ({self._val_end.date()}) must be before "
                    f"test_start_date ({self._test_start.date()})."
                )
            )
        if self._val_start > self._val_end:
            raise SplitError(
                detail=(
                    f"validation_start_date ({self._val_start.date()}) must be "
                    f"before or equal to validation_end_date ({self._val_end.date()})."
                )
            )
        if self._test_start > self._test_end:
            raise SplitError(
                detail=(
                    f"test_start_date ({self._test_start.date()}) must be "
                    f"before or equal to test_end_date ({self._test_end.date()})."
                )
            )

    def _parse_datetime(self, df: pd.DataFrame) -> pd.Series:
        """
        Parses the datetime column from ``df``.

        Raises:
            SplitError: If the column is absent or entirely unparseable.
        """
        if self._datetime_col not in df.columns:
            raise SplitError(
                detail=(
                    f"Datetime column '{self._datetime_col}' not found in DataFrame. "
                    f"Available columns: {list(df.columns)[:10]}..."
                )
            )

        dt = pd.to_datetime(df[self._datetime_col], errors="coerce")

        if dt.isna().all():
            raise SplitError(
                detail=(
                    f"Datetime column '{self._datetime_col}' is entirely unparseable."
                )
            )

        unparseable = int(dt.isna().sum())
        if unparseable > 0:
            logger.warning(
                "Splitter: %d row(s) have unparseable datetime values — excluded from all folds.",
                unparseable,
            )

        return dt

    def _build_split(
        self,
        fold_name: str,
        fold_df: pd.DataFrame,
        dt_series: pd.Series,
    ) -> DatasetSplit:
        """Constructs a DatasetSplit from a filtered DataFrame."""
        if self._target_col not in fold_df.columns:
            raise SplitError(
                detail=(
                    f"Target column '{self._target_col}' not found in fold '{fold_name}'."
                ),
                fold_name=fold_name,
            )

        y = fold_df[self._target_col].copy()
        X = fold_df.drop(columns=[self._target_col])

        fraud_count = int(y.sum())
        fraud_pct = float(fraud_count / len(y) * 100) if len(y) > 0 else 0.0

        start_date = dt_series.min().strftime("%Y-%m-%d") if not dt_series.empty else ""
        end_date = dt_series.max().strftime("%Y-%m-%d") if not dt_series.empty else ""

        return DatasetSplit(
            fold_name=fold_name,
            X=X,
            y=y,
            start_date=start_date,
            end_date=end_date,
            num_rows=len(fold_df),
            num_features=len(X.columns),
            fraud_count=fraud_count,
            fraud_pct=fraud_pct,
        )
