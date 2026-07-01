"""
Compatibility shim: config.logging

This module exists so that callers may use the import path specified in the
project contract:

    from config.logging import get_logger

Internally this delegates to config.logging_config to avoid shadowing
Python's built-in `logging` standard library module within this package.
"""

from config.logging_config import get_logger  # noqa: F401 — re-export

__all__ = ["get_logger"]
