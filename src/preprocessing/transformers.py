"""
Transformer interface and concrete implementations for the Preprocessing Layer.

All transformers inherit from BaseTransformer and expose:
    fit(X, y=None)         — fit on training data only
    transform(X)           — apply fitted state; never re-fits
    fit_transform(X, y=None)

The y=None signature future-proofs the interface for TargetEncoder.
Transformers are stateless after fitting — safe for joblib serialisation
and FastAPI inference reuse.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder as SklearnOHE

from src.preprocessing.exceptions import TransformationError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BaseTransformer (abstract interface)
# ---------------------------------------------------------------------------


class BaseTransformer(ABC):
    """
    Abstract base class for all preprocessing transformers.

    All concrete transformers must implement fit(), transform(),
    get_feature_names_out(), and get_params().

    Attributes:
        name: Unique human-readable identifier for this transformer.
        transformer_type: One of "ENCODER", "SCALER", "IDENTITY".
        columns: Columns this transformer operates on.
        is_fitted: True after fit() has been called successfully.
    """

    def __init__(
        self,
        name: str,
        transformer_type: str,
        columns: list[str],
    ) -> None:
        self.name = name
        self.transformer_type = transformer_type
        self.columns = list(columns)
        self.is_fitted: bool = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "BaseTransformer":
        """
        Fits internal state from training data.

        Args:
            X: Training feature DataFrame. Must contain self.columns.
            y: Optional target series. Ignored by most transformers.
                Required by future TargetEncoder implementations.

        Returns:
            self — enables method chaining.
        """

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies fitted transformation to X.

        Args:
            X: Feature DataFrame to transform. Must contain self.columns.

        Returns:
            pd.DataFrame: Transformed DataFrame. Never modifies X in-place.

        Raises:
            TransformationError: If called before fit() or on incompatible input.
        """

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Fits then transforms. Convenience method."""
        self.fit(X, y)
        return self.transform(X)

    @abstractmethod
    def get_feature_names_out(self) -> list[str]:
        """Returns output column names after transformation."""

    @abstractmethod
    def get_params(self) -> dict:
        """Returns serialisable configuration parameters for logging."""

    def _assert_fitted(self) -> None:
        if not self.is_fitted:
            raise TransformationError(
                transformer_name=self.name,
                step_name="unknown",
                detail=f"Transformer '{self.name}' must be fitted before transform() is called.",
            )

    def _assert_columns_present(self, X: pd.DataFrame) -> None:
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise TransformationError(
                transformer_name=self.name,
                step_name="unknown",
                detail=f"Required columns missing from DataFrame: {missing}.",
            )


# ---------------------------------------------------------------------------
# FrequencyEncoder
# ---------------------------------------------------------------------------


class FrequencyEncoder(BaseTransformer):
    """
    Maps each categorical value to its occurrence count from the training fold.

    Replaces original column values in-place (no duplicate columns).
    Unknown categories at inference receive ``unknown_value`` (from config).

    Args:
        columns: Categorical columns to encode.
        unknown_value: Integer assigned to unseen category values. Default 0.
    """

    def __init__(
        self,
        columns: list[str],
        unknown_value: int = 0,
    ) -> None:
        super().__init__(
            name="FrequencyEncoder",
            transformer_type="ENCODER",
            columns=columns,
        )
        self.unknown_value = unknown_value
        self.frequency_maps_: dict[str, dict[str, int]] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FrequencyEncoder":
        """
        Builds frequency maps from training data.

        Args:
            X: Training DataFrame. Must contain all columns in self.columns.
            y: Unused. Present for interface compatibility.

        Returns:
            self
        """
        self._assert_columns_present(X)
        self.frequency_maps_ = {}
        for col in self.columns:
            counts = X[col].value_counts(dropna=False)
            self.frequency_maps_[col] = {str(k): int(v) for k, v in counts.items()}
            logger.debug(
                "FrequencyEncoder: fitted '%s' | unique=%d", col, len(self.frequency_maps_[col])
            )
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Replaces categorical values with their training-fold frequencies.

        Modifies only the columns in self.columns. Returns a copy of X
        with those columns replaced — no new columns created.

        Args:
            X: DataFrame to transform.

        Returns:
            pd.DataFrame: X with frequency-encoded values in self.columns.
        """
        self._assert_fitted()
        self._assert_columns_present(X)

        out = X.copy()
        for col in self.columns:
            out[col] = (
                X[col]
                .map(self.frequency_maps_[col])
                .fillna(self.unknown_value)
                .astype("int32")
            )
            logger.debug("FrequencyEncoder: transformed '%s'", col)

        return out

    def get_feature_names_out(self) -> list[str]:
        """Returns input column names (no expansion)."""
        return list(self.columns)

    def get_params(self) -> dict:
        return {
            "unknown_value": self.unknown_value,
            "columns": self.columns,
            "vocabulary_sizes": {
                col: len(m) for col, m in self.frequency_maps_.items()
            } if self.is_fitted else {},
        }


# ---------------------------------------------------------------------------
# OneHotEncoder
# ---------------------------------------------------------------------------


class OneHotEncoder(BaseTransformer):
    """
    One-hot encodes categorical columns using sklearn internally.

    Categories are locked during fit() on training data only.
    Unseen values at inference receive the all-zeros row (handle_unknown="ignore").
    Original categorical columns are removed; OHE dummy columns are appended.

    Args:
        columns: Categorical columns to encode.
        drop_first: Whether to drop the first dummy column per feature.
        handle_unknown: Behaviour for unseen categories. Always "ignore".
    """

    def __init__(
        self,
        columns: list[str],
        drop_first: bool = False,
        handle_unknown: str = "ignore",
    ) -> None:
        super().__init__(
            name="OneHotEncoder",
            transformer_type="ENCODER",
            columns=columns,
        )
        self.drop_first = drop_first
        self.handle_unknown = handle_unknown
        self._ohe: SklearnOHE | None = None
        self._output_feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "OneHotEncoder":
        """
        Discovers unique categories per column from training data.

        Args:
            X: Training DataFrame. Must contain all columns in self.columns.
            y: Unused. Present for interface compatibility.

        Returns:
            self
        """
        self._assert_columns_present(X)
        drop = "first" if self.drop_first else None
        self._ohe = SklearnOHE(
            drop=drop,
            handle_unknown=self.handle_unknown,
            sparse_output=False,
            dtype=np.float32,
        )
        self._ohe.fit(X[self.columns])
        self._output_feature_names = list(self._ohe.get_feature_names_out(self.columns))
        self.is_fitted = True
        logger.debug(
            "OneHotEncoder: fitted | columns=%s | output_cols=%d",
            self.columns,
            len(self._output_feature_names),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies OHE transformation. Removes original columns; appends dummies.

        Args:
            X: DataFrame to transform.

        Returns:
            pd.DataFrame: X without original OHE columns, with dummy columns appended.
        """
        self._assert_fitted()
        self._assert_columns_present(X)

        ohe_array = self._ohe.transform(X[self.columns])
        ohe_df = pd.DataFrame(
            ohe_array,
            columns=self._output_feature_names,
            index=X.index,
        )

        out = X.drop(columns=self.columns).copy()
        out = pd.concat([out, ohe_df], axis=1)
        logger.debug(
            "OneHotEncoder: transformed | removed=%d cols | added=%d cols",
            len(self.columns),
            len(self._output_feature_names),
        )
        return out

    def get_feature_names_out(self) -> list[str]:
        """Returns OHE output column names."""
        return list(self._output_feature_names)

    def get_params(self) -> dict:
        return {
            "columns": self.columns,
            "drop_first": self.drop_first,
            "handle_unknown": self.handle_unknown,
            "output_features": self._output_feature_names,
        }


# ---------------------------------------------------------------------------
# RobustScalerTransformer
# ---------------------------------------------------------------------------


class RobustScalerTransformer(BaseTransformer):
    """
    Scales numeric features using median and interquartile range (IQR).

    Robust to outliers — appropriate for `amt`, `age`, and distance features
    which have heavy-tailed distributions. Fitted on training data only.

    Args:
        columns: Numeric columns to scale.
        with_centering: If True, subtract the median before scaling.
        with_scaling: If True, divide by the IQR.
    """

    def __init__(
        self,
        columns: list[str],
        with_centering: bool = True,
        with_scaling: bool = True,
    ) -> None:
        super().__init__(
            name="RobustScalerTransformer",
            transformer_type="SCALER",
            columns=columns,
        )
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.center_: dict[str, float] = {}
        self.scale_: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "RobustScalerTransformer":
        """
        Computes median and IQR from training data.

        Args:
            X: Training DataFrame. Must contain all columns in self.columns.
            y: Unused.

        Returns:
            self
        """
        self._assert_columns_present(X)
        for col in self.columns:
            series = X[col].dropna().astype(float)
            self.center_[col] = float(series.median()) if self.with_centering else 0.0
            q75, q25 = float(series.quantile(0.75)), float(series.quantile(0.25))
            iqr = q75 - q25
            self.scale_[col] = iqr if (self.with_scaling and iqr > 0.0) else 1.0
        self.is_fitted = True
        logger.debug("RobustScalerTransformer: fitted | columns=%s", self.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies robust scaling: (X - center_) / scale_.

        Args:
            X: DataFrame to transform.

        Returns:
            pd.DataFrame: X with scaled values in self.columns.
        """
        self._assert_fitted()
        self._assert_columns_present(X)

        out = X.copy()
        for col in self.columns:
            out[col] = (X[col].astype(float) - self.center_[col]) / self.scale_[col]
        logger.debug("RobustScalerTransformer: transformed | columns=%s", self.columns)
        return out

    def get_feature_names_out(self) -> list[str]:
        return list(self.columns)

    def get_params(self) -> dict:
        return {
            "columns": self.columns,
            "with_centering": self.with_centering,
            "with_scaling": self.with_scaling,
            "center": self.center_ if self.is_fitted else {},
            "scale": self.scale_ if self.is_fitted else {},
        }


# ---------------------------------------------------------------------------
# IdentityTransformer
# ---------------------------------------------------------------------------


class IdentityTransformer(BaseTransformer):
    """
    Passes columns through unchanged.

    Used in place of RobustScalerTransformer when scaling is disabled
    (tree-model configurations). Keeps the pipeline loop uniform — no
    branching required in pipeline.py.

    Args:
        columns: Columns to pass through.
    """

    def __init__(self, columns: list[str]) -> None:
        super().__init__(
            name="IdentityTransformer",
            transformer_type="IDENTITY",
            columns=columns,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "IdentityTransformer":
        """No-op. Sets is_fitted=True for interface consistency."""
        self._assert_columns_present(X)
        self.is_fitted = True
        logger.debug("IdentityTransformer: fitted (no-op) | columns=%s", self.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Returns a copy of X unchanged."""
        self._assert_fitted()
        self._assert_columns_present(X)
        return X.copy()

    def get_feature_names_out(self) -> list[str]:
        return list(self.columns)

    def get_params(self) -> dict:
        return {"columns": self.columns}
