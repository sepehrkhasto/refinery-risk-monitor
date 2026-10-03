"""
Monitoring and health endpoints: OHLC, timeline, sensor health, correlation.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from backend.api.dependencies import get_db, get_current_user   
from backend.db.crud import (
    get_ohlc_data, get_timeline, get_sensor_health, get_correlation_matrix
)

router = APIRouter()


@router.get("/ohlc/{unit_name}")
def ohlc(
    unit_name: str,
    interval: int = 5,
    limit: int = 100,
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    data = get_ohlc_data(db, unit_name, interval_minutes=interval, limit=limit)
    return {"unit_name": unit_name, "data": data}


@router.get("/timeline")
def timeline(
    hours_back: int = 48,
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    events = get_timeline(db, hours_back)
    return {"events": events}


@router.get("/sensor-health")
def sensor_health(
    db: Session = Depends(get_db),
    user = Depends(get_current_user)  
):
    return {"health_percent": get_sensor_health(db)}


@router.get("/correlation")
def correlation(
    unit_name: str = None,
    db: Session = Depends(get_db),
    user = Depends(get_current_user)  
):
    matrix = get_correlation_matrix(db, unit_name)
    if matrix is None:
        return {"error": "Insufficient data to calculate correlation"}
    return {"matrix": matrix}