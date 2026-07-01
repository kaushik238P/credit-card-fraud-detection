"""
Trainer for the Training Layer.

Fits one configured estimator on X_train / y_train. Stateless — holds
no state between calls. Returns the fitted estimator and duration only.

The Trainer deliberately knows nothing about:
    - Evaluation metrics
    - Plotting
    - Feature importance
    - Threshold optimization
    - Validation / test folds
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from src.training.exceptions import TrainingFailureError
from src.training.models import ModelType

logger = logging.getLogger(__name__)

_REQUIRED_ESTIMATOR_ATTRS = ("fit", "predict", "predict_proba")


class Trainer:
    """
    Fits a single sklearn-compatible estimator on training data.

    Stateless — every call to train() and predict() is independent.
    Safe for use in concurrent / multi-model pipelines.
    """

    @staticmethod
    def train(
        estimator: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_type: ModelType,
        sample_weight: np.ndarray | None = None,
    ) -> tuple[Any, float]:
        """
        Fits the estimator on the training fold.

        Args:
            estimator: Unfitted sklearn-compatible estimator.
            X_train: Training feature matrix. Never modified in place.
            y_train: Training target vector (binary 0/1).
            model_type: ModelType identifier — used for logging and errors only.
            sample_weight: Optional per-sample weight array for SAMPLE_WEIGHT
                imbalance strategy. Shape must equal len(y_train).

        Returns:
            tuple: (fitted_estimator, duration_ms)
                fitted_estimator — same object as input, after fit() call.
                duration_ms — wall-clock training time in milliseconds.

        Raises:
            TrainingFailureError: If fit() raises, or if the fitted estimator
                fails interface validation.
        """
        n_rows = len(X_train)
        n_features = len(X_train.columns)
        fraud_count = int(y_train.sum())

        logger.info(
            "Training started | model=%s | rows=%d | features=%d | fraud=%d (%.4f%%)",
            model_type.value,
            n_rows,
            n_features,
            fraud_count,
            fraud_count / n_rows * 100 if n_rows > 0 else 0.0,
        )

        _start = time.perf_counter()

        try:
            fit_kwargs: dict[str, Any] = {}
            if sample_weight is not None:
                fit_kwargs["sample_weight"] = sample_weight

            estimator.fit(X_train, y_train, **fit_kwargs)
        except Exception as exc:
            raise TrainingFailureError(
                model_type=model_type.value,
                training_rows=n_rows,
                detail=str(exc),
                original_exc=exc,
            ) from exc

        duration_ms = (time.perf_counter() - _start) * 1_000

        # Validate fitted estimator exposes the required interface
        Trainer._validate_interface(estimator, model_type)

        logger.info(
            "Training complete | model=%s | duration=%.1fms",
            model_type.value,
            duration_ms,
        )

        return estimator, duration_ms

    @staticmethod
    def predict(
        estimator: Any,
        X: pd.DataFrame,
        model_type: ModelType,
    ) -> np.ndarray:
        """
        Generates class predictions from a fitted estimator.

        Args:
            estimator: Fitted estimator (must expose predict()).
            X: Feature matrix to predict on.
            model_type: Used for error context only.

        Returns:
            np.ndarray: Integer class predictions (0 or 1).

        Raises:
            TrainingFailureError: If predict() raises.
        """
        try:
            return np.asarray(estimator.predict(X))
        except Exception as exc:
            raise TrainingFailureError(
                model_type=model_type.value,
                training_rows=len(X),
                detail=f"predict() failed: {exc}",
                original_exc=exc,
            ) from exc

    @staticmethod
    def predict_proba(
        estimator: Any,
        X: pd.DataFrame,
        model_type: ModelType,
    ) -> np.ndarray:
        """
        Generates positive-class probability scores from a fitted estimator.

        Args:
            estimator: Fitted estimator (must expose predict_proba()).
            X: Feature matrix to score.
            model_type: Used for error context only.

        Returns:
            np.ndarray: 1-D array of positive-class probabilities (shape: n_rows).

        Raises:
            TrainingFailureError: If predict_proba() raises.
        """
        try:
            proba = estimator.predict_proba(X)
            return np.asarray(proba[:, 1])
        except Exception as exc:
            raise TrainingFailureError(
                model_type=model_type.value,
                training_rows=len(X),
                detail=f"predict_proba() failed: {exc}",
                original_exc=exc,
            ) from exc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_interface(estimator: Any, model_type: ModelType) -> None:
        """
        Validates that the fitted estimator exposes the required interface.

        Raises:
            TrainingFailureError: If any required attribute is missing.
        """
        missing = [
            attr for attr in _REQUIRED_ESTIMATOR_ATTRS
            if not hasattr(estimator, attr)
        ]
        if missing:
            raise TrainingFailureError(
                model_type=model_type.value,
                training_rows=0,
                detail=(
                    f"Fitted estimator is missing required attributes: {missing}. "
                    f"Estimator type: {type(estimator).__name__}"
                ),
            )
