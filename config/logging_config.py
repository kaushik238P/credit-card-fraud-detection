"""
Centralized logging configuration for the Credit Card Fraud Detection System.

Provides a single get_logger() factory consumed by every module in the project.
The logging format and level are driven entirely by config.settings, making this
suitable for Docker, GitHub Actions, and future log-aggregation integrations
(Datadog, CloudWatch, ELK).

Design decisions:
    - Named loggers (per __name__) integrate cleanly with Python's logging
      hierarchy, allowing per-module level tuning without code changes.
    - A single call to _configure_root_logger() is guarded by a module-level
      sentinel so logging is initialized exactly once, regardless of how many
      modules call get_logger().
    - Text and JSON formats are both supported. JSON is the recommended choice
      for Docker / containerized deployments because structured log lines can
      be directly indexed by log aggregators.

Usage:
    from config.logging_config import get_logger

    logger = get_logger(__name__)
    logger.info("Loading dataset", extra={"file": "fraud.csv"})

NOTE: This module is named logging_config.py (not logging.py) to avoid
shadowing Python's built-in logging module during internal imports.
The config package __init__.py re-exports get_logger so callers can use
    from config.logging import get_logger
if they prefer that import path.
"""

from __future__ import annotations

import logging
import sys
from typing import Final

_SENTINEL: Final[str] = "fraud_detection.logging_initialized"


def _build_text_formatter(date_format: str) -> logging.Formatter:
    """
    Constructs a human-readable log formatter.

    Args:
        date_format: strftime-compatible date format string.

    Returns:
        A configured logging.Formatter instance.
    """
    fmt = (
        "[%(asctime)s] "
        "[%(levelname)-8s] "
        "[%(name)s] "
        "%(message)s"
    )
    return logging.Formatter(fmt=fmt, datefmt=date_format)


def _build_json_formatter(date_format: str) -> logging.Formatter:
    """
    Constructs a JSON-structured log formatter.

    Produces single-line JSON records suitable for ingestion by log aggregators
    (Datadog, ELK, CloudWatch). Each record includes level, logger name,
    timestamp, message, and any extra fields passed via the extra= kwarg.

    Args:
        date_format: strftime-compatible date format string.

    Returns:
        A configured logging.Formatter instance.
    """

    class _JsonFormatter(logging.Formatter):
        """Minimal JSON formatter without external dependencies."""

        def format(self, record: logging.LogRecord) -> str:  # noqa: A003
            import json
            from datetime import datetime, timezone

            payload: dict = {
                "timestamp": datetime.now(tz=timezone.utc).strftime(date_format),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }

            # Append any extra fields attached via logger.info(..., extra={...})
            _stdlib_keys = logging.LogRecord(
                "", 0, "", 0, "", (), None
            ).__dict__.keys()
            for key, value in record.__dict__.items():
                if key not in _stdlib_keys and not key.startswith("_"):
                    payload[key] = value

            if record.exc_info:
                payload["exception"] = self.formatException(record.exc_info)

            return json.dumps(payload, default=str)

    return _JsonFormatter()


def _configure_root_logger() -> None:
    """
    Initializes the root logger exactly once for the application lifetime.

    Reads format and level from config.settings. Attaches a single
    StreamHandler (stdout) to the root logger so all child loggers
    inherit the configuration automatically.

    This function is idempotent: repeated calls are no-ops.
    """
    root = logging.getLogger()

    # Guard: configure only once.
    if getattr(root, _SENTINEL, False):
        return

    from config.settings import settings  # local import to break circular ref

    level_str: str = settings.logging.level
    level: int = getattr(logging, level_str, logging.INFO)
    root.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    if settings.logging.format == "json":
        formatter = _build_json_formatter(settings.logging.date_format)
    else:
        formatter = _build_text_formatter(settings.logging.date_format)

    handler.setFormatter(formatter)

    # Remove any default handlers added by basicConfig elsewhere.
    root.handlers.clear()
    root.addHandler(handler)

    # Suppress noisy third-party loggers to WARNING in production.
    for noisy_lib in ("urllib3", "boto3", "botocore", "azure", "google"):
        logging.getLogger(noisy_lib).setLevel(logging.WARNING)

    setattr(root, _SENTINEL, True)


def get_logger(name: str) -> logging.Logger:
    """
    Returns a named logger for the given module.

    Ensures the root logger is configured before returning the child logger.
    This is the single factory function that every module in the project uses.

    Args:
        name: The logger name, conventionally passed as __name__.
              Example: get_logger(__name__) in src/ingestion/loader.py
              produces a logger named 'src.ingestion.loader'.

    Returns:
        A configured logging.Logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Dataset loaded", extra={"rows": 284_807})
    """
    _configure_root_logger()
    return logging.getLogger(name)
