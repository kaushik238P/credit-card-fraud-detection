"""
FastAPI application lifespan context manager.

Coordinations preloading, validation, and dummy prediction on startup.
"""

from config.logging import get_logger
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from src.api.exceptions.custom import ConfigurationError, PredictionError
from src.api.services.model_provider import ModelProvider
from src.api.services.inference_pipeline import InferencePipeline

logger = get_logger(__name__)

# Complete valid mock transaction payload for dummy validation prediction
DUMMY_PAYLOAD = {
    "trans_date_trans_time": "2020-06-21 12:14:30",
    "cc_num": 123456789012,
    "merchant": "fraud_test_store",
    "category": "grocery_pos",
    "amt": 150.75,
    "first": "John",
    "last": "Doe",
    "gender": "M",
    "street": "123 Test St",
    "city": "Dallas",
    "state": "TX",
    "zip": 75201,
    "lat": 32.7767,
    "long": -96.7970,
    "city_pop": 1300000,
    "job": "Software Engineer",
    "dob": "1988-04-12",
    "trans_num": "dummy_num_1234",
    "unix_time": 1592741670,
    "merch_lat": 32.7800,
    "merch_long": -96.8000,
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI Lifespan Context Manager.
    
    Pre-loads model and encoders into cache, validates schemas and versions,
    and runs a dummy prediction to verify service health before receiving traffic.
    """
    logger.info("Lifespan startup: preloading CatBoost model and pipeline encoders...")
    
    try:
        # 1. Load singleton assets
        provider = ModelProvider()
        context = provider.load_assets()
        
        # 2. Run one dummy prediction to warm up the pipeline and prove end-to-end correctness
        logger.info("Running warm-up validation dummy prediction...")
        pipeline = InferencePipeline(context)
        
        dummy_results = pipeline.predict([DUMMY_PAYLOAD])
        if not dummy_results or len(dummy_results) != 1:
            raise PredictionError("Warm-up dummy prediction returned empty or invalid results.")
            
        dummy_res = dummy_results[0]
        logger.info(
            "Dummy prediction SUCCESS | is_fraud=%d | prob=%.4f | latency=%.2fms",
            dummy_res["is_fraud"],
            dummy_res["probability"],
            dummy_res["latency_ms"],
        )
        
        logger.info("Lifespan startup complete. API is ready to receive requests.")
        
    except Exception as exc:
        logger.critical("CRITICAL: Startup validation failed! Server will shut down. Cause: %s", exc)
        # Propagate exception to fail startup and prevent server from binding ports
        raise ConfigurationError(f"Startup validation failed: {exc}") from exc
        
    yield
    
    logger.info("Lifespan shutdown: cleaning up cached in-memory resources.")
