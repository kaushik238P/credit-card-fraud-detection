"""
Centralized configuration package for the Credit Card Fraud Detection System.

Exports:
    settings: Application-wide settings instance.
    get_logger: Factory function for named loggers.
"""

from config.logging_config import get_logger
from config.settings import settings

__all__ = ["get_logger", "settings"]
