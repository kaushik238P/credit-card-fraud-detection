"""
Inference Pipeline service coordinating request parsing, feature transform, and prediction.
"""

from config.logging import get_logger
import time
from typing import Any

import pandas as pd
import numpy as np

from src.api.exceptions.custom import TransformationError, PredictionError
from src.api.models.inference_context import InferenceContext

logger = get_logger(__name__)






# ---------------------------------------------------------------------------
# InferencePipeline Class
# ---------------------------------------------------------------------------

class InferencePipeline:
    """
    Orchestrates the HTTP inference request processing, data transformation,
    and model prediction using the loaded InferenceContext.
    """

    def __init__(self, context: InferenceContext) -> None:
        self.context = context

    def predict(self, raw_data_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Runs the full prediction pipeline on a list of raw transaction inputs.
        
        Args:
            raw_data_list: Raw transaction input dictionaries.
            
        Returns:
            list[dict[str, Any]]: Output dictionaries containing scores, classes, and thresholds.
            
        Raises:
            TransformationError: If data transformations fail.
            PredictionError: If CatBoost inference fails.
        """
        if not raw_data_list:
            return []

        t_start = time.perf_counter()
        
        try:
            # 1. Convert input to DataFrame
            df_raw = pd.DataFrame(raw_data_list)
            
            # 2. Execute FeatureEngineeringPipeline.transform()
            engineered = self.context.feature_engineering_pipeline.transform(df_raw)
            
            # 3. Execute PreprocessingPipeline.transform()
            preprocessed_df = self.context.preprocessing_pipeline.transform(engineered)
            
            # 4. Column Alignment
            # Sort columns to match training schema feature order exactly
            expected_features = self.context.feature_schema.get("features", [])
            if not expected_features:
                expected_features = self.context.metadata["training"].get("feature_names", [])
                
            missing_cols = [c for c in expected_features if c not in preprocessed_df.columns]
            if missing_cols:
                raise TransformationError(f"Schema mismatch: Preprocessed output is missing expected features: {missing_cols}")
                
            # Align features
            df_aligned = preprocessed_df[expected_features]
            
            # 5. Run CatBoost Prediction Proba
            # Ensure input is float/numeric as expected by CatBoost
            X = df_aligned.to_numpy().astype(np.float32)
            
            try:
                probabilities = self.context.model.predict_proba(X)[:, 1]
            except Exception as pred_exc:
                raise PredictionError(f"CatBoost predict_proba failed: {pred_exc}") from pred_exc
                
            # 6. Apply Decision Threshold Classification
            threshold = self.context.threshold
            decisions = (probabilities >= threshold).astype(int)
            
            latency_total_ms = (time.perf_counter() - t_start) * 1_000
            latency_per_sample_ms = latency_total_ms / len(raw_data_list)
            
            # 7. Format output predictions
            results = []
            for prob, dec in zip(probabilities, decisions):
                results.append({
                    "is_fraud": int(dec),
                    "probability": float(prob),
                    "threshold_used": float(threshold),
                    "model_version": self.context.model_version,
                    "latency_ms": round(latency_per_sample_ms, 4),
                })
                
            return results
        except Exception as e:
            if not isinstance(e, (TransformationError, PredictionError)):
                raise PredictionError(f"Prediction pipeline encountered unhandled exception: {e}") from e
            raise
