"""
Custom exception classes and unified error handlers for FastAPI.
All application exceptions derive from this module and have a uniform structure.
"""

from fastapi import Request
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger("refinery")


class AppException(Exception):
    """Base application exception with status code and error code."""
    def __init__(self, message: str, code: str = "INTERNAL_ERROR", status_code: int = 500):
        self.message = message
        self.code = code
        self.status_code = status_code
        super().__init__(self.message)


class NotFoundException(AppException):
    """Resource not found."""
    def __init__(self, message: str = "Requested resource not found"):
        super().__init__(message, code="NOT_FOUND", status_code=404)


class UnauthorizedException(AppException):
    """Unauthorized access."""
    def __init__(self, message: str = "Authentication required"):
        super().__init__(message, code="UNAUTHORIZED", status_code=401)


class ForbiddenException(AppException):
    """Forbidden (role level)."""
    def __init__(self, message: str = "You do not have permission"):
        super().__init__(message, code="FORBIDDEN", status_code=403)


class ValidationException(AppException):
    """Input validation error."""
    def __init__(self, message: str = "Invalid input data"):
        super().__init__(message, code="VALIDATION_ERROR", status_code=422)


# FastAPI handlers
async def app_exception_handler(request: Request, exc: AppException):
    logger.error(
        f"AppException: {exc.message}",
        extra={"correlation_id": getattr(request.state, "correlation_id", None)}
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message, "code": exc.code}
    )


async def global_exception_handler(request: Request, exc: Exception):
    logger.critical(
        f"Unhandled exception: {exc}",
        exc_info=True,
        extra={"correlation_id": getattr(request.state, "correlation_id", None)}
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "code": "INTERNAL_ERROR"}
    )