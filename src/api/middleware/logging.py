"""
Centralized request tracing and logging middleware.
"""

from config.logging import get_logger
import time
import uuid
from typing import Callable

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = get_logger("src.api.middleware.logging")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware intercepting incoming HTTP requests to trace ID, track latency,
    and log structured request metrics.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 1. Resolve Request ID
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = str(uuid.uuid4())

        # Start timer
        start_time = time.perf_counter()
        
        # Inject request_id into state for downstream access
        request.state.request_id = request_id
        
        logger.info(
            "Request started | request_id=%s | method=%s | url=%s | client=%s",
            request_id,
            request.method,
            request.url.path,
            request.client.host if request.client else "unknown",
        )

        response = Response("Internal Server Error", status_code=500)
        try:
            response = await call_next(request)
        except Exception as exc:
            logger.exception(
                "Unhandled exception in request lifecycle | request_id=%s | error=%s",
                request_id,
                exc,
            )
            raise
        finally:
            # Calculate duration in milliseconds
            duration_ms = (time.perf_counter() - start_time) * 1_000
            
            # Write Request ID back to client header
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"
            
            logger.info(
                "Request complete | request_id=%s | url=%s | status=%d | duration=%.2fms",
                request_id,
                request.url.path,
                response.status_code,
                duration_ms,
            )
            
        return response


def register_logging_middleware(app: FastAPI) -> None:
    """Registers RequestLoggingMiddleware to the application instance."""
    app.add_middleware(RequestLoggingMiddleware)
