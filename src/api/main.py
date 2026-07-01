"""
Main entrypoint for the FastAPI Inference Service.
"""

from __future__ import annotations

from config.logging import get_logger
logger = get_logger(__name__)

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app
from typing import Any
from src.api.config import inference_settings
from src.api.exceptions.handlers import register_exception_handlers
from src.api.lifespan import lifespan
from src.api.middleware.logging import register_logging_middleware
from src.api.routers import health, prediction, metadata, version


# 1. Instantiate FastAPI application with lifespan context manager
app = FastAPI(
    title="Credit Card Fraud Detection Inference Service",
    description="Real-time HTTP API for transaction evaluation and fraud classification.",
    version="1.0.0",
    lifespan=lifespan,
)

# 2. Register global exception handlers
register_exception_handlers(app)

# 3. Mount Trusted Host middleware
if "*" not in inference_settings.trusted_hosts:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=inference_settings.trusted_hosts,
    )

# 4. Mount CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=inference_settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 5. Payload size limiting middleware
@app.middleware("http")
async def limit_payload_size(request: Request, call_next) -> Any:
    content_length = request.headers.get("content-length")
    if content_length:
        if int(content_length) > inference_settings.max_content_size_bytes:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={
                    "type": "https://errors.fraud-api.local/payload-too-large",
                    "title": "Request Payload Too Large",
                    "status": 413,
                    "detail": f"Payload size exceeds maximum allowed limit of {inference_settings.max_content_size_bytes} bytes.",
                    "instance": str(request.url),
                },
                headers={"Content-Type": "application/problem+json"},
            )
    return await call_next(request)


# 6. Mount Centralized request logging middleware
register_logging_middleware(app)

# 7. Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# 8. Register routers
app.include_router(health.router)
app.include_router(prediction.router)
app.include_router(metadata.router)
app.include_router(version.router)


@app.get("/", include_in_schema=False)
async def root_redirect() -> JSONResponse:
    """Redirects base path visits to model overview page."""
    return JSONResponse(
        content={
            "service": "Credit Card Fraud Detection API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health/liveness",
        }
    )
