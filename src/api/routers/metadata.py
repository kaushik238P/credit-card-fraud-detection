"""
FastAPI router for model metadata and features endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, status

from src.api.dependencies.providers import get_inference_context
from src.api.models.inference_context import InferenceContext
from src.api.schemas.metadata import ModelFeaturesResponse, ModelInfoResponse, MetadataResponse

router = APIRouter(tags=["Model Metadata"])


@router.get(
    "/model",
    response_model=ModelInfoResponse,
    status_code=status.HTTP_200_OK,
    summary="Get active model information",
)
async def get_model_info(
    context: InferenceContext = Depends(get_inference_context),
) -> ModelInfoResponse:
    """
    Returns information about the loaded CatBoost model, training setup, and parameters.
    """
    tr_meta = context.metadata["training"]
    return ModelInfoResponse(
        model_type=tr_meta.get("model_type", "CATBOOST"),
        model_version=context.model_version,
        training_version=context.training_version,
        hyperparameters=tr_meta.get("hyperparameters", {}),
        random_seed=tr_meta.get("random_seed", 42),
        imbalance_strategy=tr_meta.get("imbalance_strategy", "SCALE_POS_WEIGHT"),
        imbalance_ratio=tr_meta.get("imbalance_ratio", 0.0),
        feature_count=tr_meta.get("feature_count", 33),
    )


@router.get(
    "/model/features",
    response_model=ModelFeaturesResponse,
    status_code=status.HTTP_200_OK,
    summary="Get model features definition",
)
async def get_model_features(
    context: InferenceContext = Depends(get_inference_context),
) -> ModelFeaturesResponse:
    """
    Returns the required feature names, execution order, types, and encoder mappings.
    """
    features = context.feature_schema.get("features", [])
    if not features:
        features = context.metadata["training"].get("feature_names", [])

    # Map features to expected types (standard categories vs numeric floats/ints)
    # The categories are: cc_num, zip, lat, long, city_pop, merch_lat, merch_long, amt, age,
    # and all frequency/OHE features.
    types = {}
    for f in features:
        if f in ("gender_F", "gender_M"):
            types[f] = "binary"
        elif f.endswith("_frequency") or f in ("merchant", "category", "job", "state", "city"):
            types[f] = "numeric_encoded"
        else:
            types[f] = "numeric"

    # Encoders configurations
    ohe_out = ["gender_F", "gender_M"]
    encoded_features = {
        "FREQUENCY_ENCODING": ["category", "city", "job", "merchant", "state"],
        "ONE_HOT_ENCODING": ["gender"],
        "ONE_HOT_ENCODED_OUTPUTS": ohe_out,
    }

    return ModelFeaturesResponse(
        features=features,
        count=len(features),
        types=types,
        encoded_features=encoded_features,
    )


@router.get(
    "/metadata",
    response_model=MetadataResponse,
    status_code=status.HTTP_200_OK,
    summary="Get combined pipeline metadata",
)
async def get_metadata(
    context: InferenceContext = Depends(get_inference_context),
) -> MetadataResponse:
    """
    Returns full combined metadata details including hashes, versions, run IDs, and threshold.
    """
    tr_meta = context.metadata["training"]
    return MetadataResponse(
        dataset_hash=context.dataset_hash,
        feature_schema_hash=tr_meta.get("feature_schema_hash", "unknown"),
        preprocessing_version=context.preprocessing_version,
        training_version=context.training_version,
        preprocessing_artifact_version=tr_meta.get("preprocessing_artifact_version", "unknown"),
        mlflow_run_id=context.mlflow_run_id,
        mlflow_model_version=context.mlflow_model_version,
        threshold=context.threshold,
        threshold_strategy=context.metadata.get("threshold_strategy", "MAX_F1"),
    )
