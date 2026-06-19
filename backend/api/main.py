"""
Main FastAPI entry point with WebSocket support, background tasks,
and production-grade lifespan management.
"""

import os
from contextlib import asynccontextmanager
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.db.database import engine, Base
from backend.middleware import LoggingMiddleware
from backend.exceptions import AppException, app_exception_handler, global_exception_handler
from backend.api.routes import (
    dashboard, predictions, analysis, root_cause, monitoring,
    auth, reports, simulate, websocket, health
)
from backend.services.live_simulator import LiveSimulator
from backend.services.auto_trainer import get_auto_trainer
from backend.websocket.manager import periodic_risk_broadcast
from backend.logger import logger

# ⚠️ DO NOT use Base.metadata.create_all here! Alembic handles migrations.
# Base.metadata.create_all(bind=engine)  # Commented for production safety

# Singleton services
live_sim = LiveSimulator()
auto_trainer = get_auto_trainer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        live_sim.start()
    except Exception as e:
        logger.error(f"Failed to start live_sim: {e}")
    try:
        auto_trainer.start()
    except Exception as e:
        logger.error(f"Failed to start auto_trainer: {e}")

    broadcast_task = asyncio.create_task(periodic_risk_broadcast())
    yield
    broadcast_task.cancel()
    try:
        await broadcast_task
    except asyncio.CancelledError:
        pass
    auto_trainer.stop()
    live_sim.stop()


app = FastAPI(
    title="Refinery Risk Monitor",
    version="5.1.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Rate limiter is disabled for now (to avoid dependency issues)
# To enable, install slowapi and uncomment the lines below:
# from slowapi.errors import RateLimitExceeded
# from backend.middleware.rate_limiter import limiter, limiter_exceed_handler
# app.state.limiter = limiter
# app.add_exception_handler(RateLimitExceeded, limiter_exceed_handler)

# Middlewares
app.add_middleware(LoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS.split(",") if settings.CORS_ORIGINS != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handlers
app.add_exception_handler(AppException, app_exception_handler)
app.add_exception_handler(Exception, global_exception_handler)

# API routers (health first)
app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(dashboard.router, prefix="/api", tags=["Dashboard"])
app.include_router(predictions.router, prefix="/api", tags=["Predictions"])
app.include_router(analysis.router, prefix="/api", tags=["Analysis"])
app.include_router(root_cause.router, prefix="/api", tags=["Root Cause"])
app.include_router(monitoring.router, prefix="/api", tags=["Monitoring"])
app.include_router(auth.router, prefix="/api", tags=["Authentication"])
app.include_router(reports.router, prefix="/api", tags=["Reports"])
app.include_router(simulate.router, prefix="/api", tags=["Simulation"])
app.include_router(websocket.router, prefix="/api", tags=["WebSocket"])