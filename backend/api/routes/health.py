"""
Health check endpoints for liveness, readiness, and overall status.
All endpoints are async for compatibility with async WebSocket manager.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text   
from backend.api.dependencies import get_db
from backend.config import settings
from backend.websocket.manager import manager

router = APIRouter()


@router.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """
    Basic health check: returns service status and database connectivity.
    Async version to call async manager.get_stats().
    """
    db_status = "connected"
    try:
        db.execute(text("SELECT 1"))  
    except Exception:
        db_status = "disconnected"

    ws_stats = await manager.get_stats()

    return {
        "status": "ok",
        "version": "5.1.0",
        "environment": settings.ENVIRONMENT,
        "database": db_status,
        "websocket_connections": ws_stats["total_connections"]
    }


@router.get("/health/ready")
async def readiness_check(db: Session = Depends(get_db)):
    """
    Readiness probe: checks if the service is ready to accept traffic.
    """
    try:
        db.execute(text("SELECT 1"))   
        db_ready = True
    except Exception:
        db_ready = False

    return {
        "status": "ready" if db_ready else "not_ready",
        "database": "connected" if db_ready else "disconnected"
    }


@router.get("/health/live")
async def liveness_check():
    """
    Liveness probe: simple endpoint to confirm the service is alive.
    """
    return {"status": "alive"}