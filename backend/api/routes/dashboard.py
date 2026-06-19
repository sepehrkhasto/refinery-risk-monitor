"""
Dashboard management endpoints.
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from backend.api.dependencies import get_db, get_current_user
from backend.db import crud

router = APIRouter()


@router.get("/stats")
def stats(db: Session = Depends(get_db), user = Depends(get_current_user)):
    """Get overall system statistics."""
    return crud.get_dashboard_stats(db)


@router.get("/data")
def historical_data(limit: int = 60, db: Session = Depends(get_db), user = Depends(get_current_user)):
    """Get recent risk records."""
    records = crud.get_recent_risk_assessments(db, limit=limit)
    return [
        {
            "unit_name": r.unit.name,
            "risk_score": r.risk_score,
            "status": r.status,
            "root_cause": r.root_cause,
            "timestamp": r.timestamp.isoformat() + 'Z'
        }
        for r in records
    ]


@router.get("/top-risky")
def top_risky(db: Session = Depends(get_db), user = Depends(get_current_user)):
    """Get units with highest current risk."""
    units = crud.get_top_risky_units(db)
    return [
        {"unit_name": u.unit.name, "risk_score": u.risk_score, "status": u.status}
        for u in units
    ]


@router.get("/status-distribution")
def status_distribution(db: Session = Depends(get_db), user = Depends(get_current_user)):
    """Get distribution of risk statuses (Normal/Warning/Critical)."""
    return crud.get_status_distribution(db)


@router.get("/alerts")
def alerts(limit: int = 50, db: Session = Depends(get_db), user=Depends(get_current_user)):
    """Get recent alerts (risk assessments with status Warning or Critical)."""
    from sqlalchemy import desc
    from backend.db.db_models import RiskAssessment

    records = db.query(RiskAssessment).filter(
        RiskAssessment.status.in_(['Warning', 'Critical'])  
    ).order_by(desc(RiskAssessment.timestamp)).limit(limit).all()
    return [
        {
            "unit_name": r.unit.name,
            "risk_score": r.risk_score,
            "status": r.status,
            "root_cause": r.root_cause,
            "timestamp": r.timestamp.isoformat() + 'Z'
        }
        for r in records
    ]


@router.get("/heatmap")
def heatmap(db: Session = Depends(get_db), user = Depends(get_current_user)):
    """Get heatmap data (unit name, risk score, status)."""
    return crud.get_heatmap_data(db)


@router.get("/units")
def list_units(db: Session = Depends(get_db), user = Depends(get_current_user)):
    """Get list of all refinery units."""
    units = crud.get_all_units(db)
    return [{"id": u.id, "name": u.name} for u in units]


@router.get("/latest-sensor/{unit_name}")
def latest_sensor(unit_name: str, db: Session = Depends(get_db), user = Depends(get_current_user)):
    """Get latest sensor values for a specific unit."""
    features = crud.get_latest_features_for_unit(db, unit_name)
    if not features:
        return {"error": "No data found for this unit"}
    return features