"""
FastAPI router for prediction inference endpoints.
"""

from config.logging import get_logger
import time

from fastapi import APIRouter, Depends, status, HTTPException
from prometheus_client import Counter, Histogram

from src.api.dependencies.providers import get_inference_pipeline, verify_api_key
from src.api.exceptions.custom import TransformationError, PredictionError
from src.api.schemas.prediction import (
    TransactionInput,
    PredictionResponse,
    BatchPredictionResponse,
)
from src.api.services.inference_pipeline import InferencePipeline

logger = get_logger(__name__)

router = APIRouter(tags=["Predictions"], dependencies=[Depends(verify_api_key)])


# ---------------------------------------------------------------------------
# Prometheus metrics definition
# ---------------------------------------------------------------------------

PREDICTION_COUNT = Counter(
    "api_predictions_total",
    "Total count of transactions evaluated for fraud.",
    ["model_version", "outcome"],
)

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total count of API requests served.",
    ["endpoint"],
)

LATENCY_HISTOGRAM = Histogram(
    "api_latency_seconds",
    "Latency distribution of prediction requests.",
    ["endpoint"],
)

ERROR_COUNT = Counter(
    "api_errors_total",
    "Total count of transaction evaluation failures.",
    ["endpoint", "error_type"],
)


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate single credit card transaction for fraud",
)
async def predict_single(
    payload: TransactionInput,
    pipeline: InferencePipeline = Depends(get_inference_pipeline),
) -> PredictionResponse:
    """
    Evaluates a single raw transaction, returning a classification decision (Fraud/Legitimate),
    probability score, active threshold, and processing latency.
    """
    REQUEST_COUNT.labels(endpoint="/predict").inc()
    t_start = time.perf_counter()
    
    # 1. Convert input to dict list
    raw_dict = payload.model_dump()
    
    try:
        results = pipeline.predict([raw_dict])
        if not results:
            raise HTTPException(status_code=500, detail="Prediction pipeline returned empty output.")
            
        res = results[0]
        outcome = "fraud" if res["is_fraud"] == 1 else "legitimate"
        PREDICTION_COUNT.labels(
            model_version=res["model_version"],
            outcome=outcome,
        ).inc()
        
        latency = time.perf_counter() - t_start
        LATENCY_HISTOGRAM.labels(endpoint="/predict").observe(latency)
        
        # Log basic decision details without sensitive cardholder fields
        logger.info(
            "Prediction outcome | is_fraud=%d | probability=%.4f | model_version=%s | duration=%.2fms",
            res["is_fraud"],
            res["probability"],
            res["model_version"],
            latency * 1_000,
        )
        
        return PredictionResponse(**res)
        
    except (TransformationError, PredictionError) as exc:
        ERROR_COUNT.labels(endpoint="/predict", error_type=type(exc).__name__).inc()
        raise
    except Exception as e:
        ERROR_COUNT.labels(endpoint="/predict", error_type="UnhandledException").inc()
        raise PredictionError(f"Prediction failed due to internal error: {e}") from e


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate batch list of credit card transactions for fraud",
)
async def predict_batch(
    payload: list[TransactionInput],
    pipeline: InferencePipeline = Depends(get_inference_pipeline),
) -> BatchPredictionResponse:
    """
    Evaluates a batch of transactions, returning classification decisions and scores for each.
    """
    REQUEST_COUNT.labels(endpoint="/predict/batch").inc()
    t_start = time.perf_counter()
    
    if not payload:
        raise HTTPException(status_code=400, detail="Request batch payload cannot be empty.")
        
    # 1. Convert batch list to dicts
    raw_list = [item.model_dump() for item in payload]
    
    try:
        results = pipeline.predict(raw_list)
        
        # Track statistics
        for res in results:
            outcome = "fraud" if res["is_fraud"] == 1 else "legitimate"
            PREDICTION_COUNT.labels(
                model_version=res["model_version"],
                outcome=outcome,
            ).inc()
            
        latency = time.perf_counter() - t_start
        LATENCY_HISTOGRAM.labels(endpoint="/predict/batch").observe(latency)
        
        logger.info(
            "Batch predictions processed | batch_size=%d | duration=%.2fms",
            len(payload),
            latency * 1_000,
        )
        
        pred_responses = [PredictionResponse(**r) for r in results]
        return BatchPredictionResponse(
            predictions=pred_responses,
            batch_size=len(payload),
            latency_ms=round(latency * 1_000, 4),
        )
        
    except (TransformationError, PredictionError) as exc:
        ERROR_COUNT.labels(endpoint="/predict/batch", error_type=type(exc).__name__).inc()
        raise
    except Exception as e:
        ERROR_COUNT.labels(endpoint="/predict/batch", error_type="UnhandledException").inc()
        raise PredictionError(f"Batch prediction failed due to internal error: {e}") from e
