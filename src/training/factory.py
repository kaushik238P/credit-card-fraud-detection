"""
ModelFactory for the Training Layer.

Instantiates and configures sklearn-compatible estimators from ModelType
and TrainingSettings. The pipeline never instantiates estimators directly.

Imbalance params are pre-computed by the pipeline and passed in —
the factory never computes ratios itself.
"""

from __future__ import annotations

import logging
from typing import Any

from config.settings import TrainingSettings
from src.training.exceptions import ModelCreationError
from src.training.models import ModelType

logger = logging.getLogger(__name__)


class ModelFactory:
    """
    Instantiates configured estimators from ModelType + TrainingSettings.

    Stateless — every call to create() is independent. No internal state
    is held between calls, making this safe for concurrent use.
    """

    @staticmethod
    def create(
        model_type: ModelType,
        cfg: TrainingSettings,
        class_weight_map: dict[int, float] | None = None,
        scale_pos_weight: float | None = None,
    ) -> Any:
        """
        Creates and returns an unfitted sklearn-compatible estimator.

        Imbalance params are mutually exclusive:
            - class_weight_map → passed as class_weight param (LR, RF, LGBM)
            - scale_pos_weight → passed directly to XGBoost / LGBM
            - Neither          → no imbalance parameter (NONE strategy)

        Args:
            model_type: Target model type from ModelType enum.
            cfg: TrainingSettings — source of all hyperparameters.
            class_weight_map: Pre-computed {0: 1.0, 1: ratio} or None.
            scale_pos_weight: Pre-computed legitimate/fraud ratio or None.

        Returns:
            Unfitted estimator conforming to the sklearn fit/predict interface.

        Raises:
            ModelCreationError: If the library is not installed or params
                are invalid.
        """
        creators = {
            ModelType.LOGISTIC_REGRESSION: ModelFactory._create_logistic_regression,
            ModelType.RANDOM_FOREST: ModelFactory._create_random_forest,
            ModelType.XGBOOST: ModelFactory._create_xgboost,
            ModelType.LIGHTGBM: ModelFactory._create_lightgbm,
            ModelType.CATBOOST: ModelFactory._create_catboost,
        }

        creator = creators.get(model_type)
        if creator is None:
            raise ModelCreationError(
                model_type=model_type.value,
                detail=f"No factory creator registered for '{model_type.value}'.",
            )

        try:
            estimator = creator(cfg, class_weight_map, scale_pos_weight)
            logger.info(
                "Model created | type=%s | imbalance_map=%s | scale_pos_weight=%s",
                model_type.value,
                class_weight_map is not None,
                scale_pos_weight,
            )
            return estimator
        except ModelCreationError:
            raise
        except Exception as exc:
            raise ModelCreationError(
                model_type=model_type.value,
                detail=str(exc),
                original_exc=exc,
            ) from exc

    # ------------------------------------------------------------------
    # Per-model creators (private)
    # ------------------------------------------------------------------

    @staticmethod
    def _create_logistic_regression(
        cfg: TrainingSettings,
        class_weight_map: dict | None,
        scale_pos_weight: float | None,
    ) -> Any:
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError as exc:
            raise ModelCreationError(
                model_type="LOGISTIC_REGRESSION",
                detail="scikit-learn is not installed.",
                original_exc=exc,
            ) from exc

        cw = class_weight_map if class_weight_map is not None else None
        return LogisticRegression(
            C=cfg.lr_C,
            penalty=cfg.lr_penalty,
            solver=cfg.lr_solver,
            max_iter=cfg.lr_max_iter,
            class_weight=cw,
            random_state=cfg.random_seed,
            n_jobs=cfg.n_jobs,
        )

    @staticmethod
    def _create_random_forest(
        cfg: TrainingSettings,
        class_weight_map: dict | None,
        scale_pos_weight: float | None,
    ) -> Any:
        try:
            from sklearn.ensemble import RandomForestClassifier
        except ImportError as exc:
            raise ModelCreationError(
                model_type="RANDOM_FOREST",
                detail="scikit-learn is not installed.",
                original_exc=exc,
            ) from exc

        cw = class_weight_map if class_weight_map is not None else None
        return RandomForestClassifier(
            n_estimators=cfg.rf_n_estimators,
            max_depth=cfg.rf_max_depth,
            min_samples_leaf=cfg.rf_min_samples_leaf,
            class_weight=cw,
            random_state=cfg.random_seed,
            n_jobs=cfg.n_jobs,
        )

    @staticmethod
    def _create_xgboost(
        cfg: TrainingSettings,
        class_weight_map: dict | None,
        scale_pos_weight: float | None,
    ) -> Any:
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ModelCreationError(
                model_type="XGBOOST",
                detail="xgboost is not installed. Run: pip install xgboost",
                original_exc=exc,
            ) from exc

        spw = scale_pos_weight if scale_pos_weight is not None else 1.0
        params: dict[str, Any] = {
            "n_estimators": cfg.xgb_n_estimators,
            "learning_rate": cfg.xgb_learning_rate,
            "max_depth": cfg.xgb_max_depth,
            "subsample": cfg.xgb_subsample,
            "colsample_bytree": cfg.xgb_colsample_bytree,
            "tree_method": cfg.xgb_tree_method,
            "eval_metric": cfg.xgb_eval_metric,
            "scale_pos_weight": spw,
            "random_state": cfg.random_seed,
            "n_jobs": cfg.n_jobs,
            "verbosity": 0,
        }
        # early_stopping_rounds: architecture prepared; not wired until
        # eval_set is available at fit() time (future extension).
        if cfg.early_stopping_rounds is not None:
            params["early_stopping_rounds"] = cfg.early_stopping_rounds

        return XGBClassifier(**params)

    @staticmethod
    def _create_lightgbm(
        cfg: TrainingSettings,
        class_weight_map: dict | None,
        scale_pos_weight: float | None,
    ) -> Any:
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ModelCreationError(
                model_type="LIGHTGBM",
                detail="lightgbm is not installed. Run: pip install lightgbm",
                original_exc=exc,
            ) from exc

        # LGBM accepts either class_weight or scale_pos_weight; prefer
        # scale_pos_weight when provided.
        if scale_pos_weight is not None:
            cw = None
            spw = scale_pos_weight
        elif class_weight_map is not None:
            cw = class_weight_map
            spw = None
        else:
            cw = None
            spw = None

        params: dict[str, Any] = {
            "num_leaves": cfg.lgbm_num_leaves,
            "learning_rate": cfg.lgbm_learning_rate,
            "n_estimators": cfg.lgbm_n_estimators,
            "min_child_samples": cfg.lgbm_min_child_samples,
            "random_state": cfg.random_seed,
            "n_jobs": cfg.n_jobs,
            "verbose": -1,
        }
        if cw is not None:
            params["class_weight"] = cw
        if spw is not None:
            params["scale_pos_weight"] = spw

        return LGBMClassifier(**params)

    @staticmethod
    def _create_catboost(
        cfg: TrainingSettings,
        class_weight_map: dict | None,
        scale_pos_weight: float | None,
    ) -> Any:
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise ModelCreationError(
                model_type="CATBOOST",
                detail="catboost is not installed. Run: pip install catboost",
                original_exc=exc,
            ) from exc

        params: dict[str, Any] = {
            "iterations": cfg.cat_iterations,
            "depth": cfg.cat_depth,
            "learning_rate": cfg.cat_learning_rate,
            "random_seed": cfg.random_seed,
            "verbose": cfg.cat_verbose,
            "allow_writing_files": False,
        }
        if class_weight_map is not None:
            # CatBoost expects class_weights as a list [w_class0, w_class1]
            params["class_weights"] = [
                class_weight_map.get(0, 1.0),
                class_weight_map.get(1, 1.0),
            ]

        return CatBoostClassifier(**params)
