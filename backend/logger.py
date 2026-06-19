"""
Structured JSON logging with automatic correlation_id injection using contextvars.
All logs from the same request/websocket message automatically include the correlation_id.
"""

import logging
import sys
from contextvars import ContextVar
from typing import Optional

from backend.config import settings

# Context variable to hold correlation_id for the current async context
correlation_id_var: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


class CorrelationFilter(logging.Filter):
    """Logging filter that adds correlation_id to every log record from the context."""
    def filter(self, record: logging.LogRecord) -> bool:
        cid = correlation_id_var.get()
        setattr(record, "correlation_id", cid if cid is not None else "N/A")
        return True


def setup_logging() -> logging.LoggerAdapter:
    """Configure root logger with JSON formatter and correlation filter."""
    logger = logging.getLogger("refinery")
    logger.setLevel(settings.LOG_LEVEL.upper())

    # Clear existing handlers to avoid duplication
    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '{"time": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", '
        '"message": "%(message)s", "correlation_id": "%(correlation_id)s"}'
    )
    handler.setFormatter(formatter)
    handler.addFilter(CorrelationFilter())
    logger.addHandler(handler)

    # Prevent propagation to root logger
    logger.propagate = False

    # Return a LoggerAdapter (though not strictly needed now, kept for compatibility)
    return logging.LoggerAdapter(logger, {})


logger = setup_logging()


def set_correlation_id(cid: str) -> None:
    """Set the correlation_id for the current async context."""
    correlation_id_var.set(cid)


def get_correlation_id() -> Optional[str]:
    """Get the correlation_id from the current async context."""
    return correlation_id_var.get()