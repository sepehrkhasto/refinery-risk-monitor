"""
FastAPI middleware for correlation_id injection and request logging.
"""

import uuid
import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from backend.logger import logger, set_correlation_id  


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.correlation_id = request_id
        set_correlation_id(request_id)   

        start = time.time()
        response = await call_next(request)
        process_time = time.time() - start

        logger.info(
            f"{request.method} {request.url.path} completed in {process_time:.3f}s",
            extra={"correlation_id": request_id}
        )
        response.headers["X-Request-ID"] = request_id
        return response