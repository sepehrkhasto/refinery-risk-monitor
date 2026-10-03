"""
Operator reporting endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.api.dependencies import get_db, get_current_user
from backend.db import crud, schemas

router = APIRouter()


@router.post("/operator-report", response_model=schemas.OperatorReportResponse)
def submit_report(
    report: schemas.OperatorReportCreate,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Submit an operator report (authentication required)."""
    unit = crud.get_unit_by_name(db, report.unit_name)
    if not unit:
        raise HTTPException(status_code=404, detail="Unit not found")
    new_report = crud.create_operator_report(db, user.id, unit.id, report)
    return {
        "id": new_report.id,
        "user_id": new_report.user_id,
        "unit_id": new_report.unit_id,
        "unit_name": unit.name,
        "timestamp": new_report.timestamp,
        "description": new_report.description,
        "sensor_name": new_report.sensor_name,
        "reported_issue": new_report.reported_issue,
        "risk_impact": new_report.risk_impact,
        "status": new_report.status,
    }


@router.get("/operator-reports", response_model=list[schemas.OperatorReportResponse])
def get_reports(
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get operator reports.
    Operators see only their own reports; engineers/admins see all.
    """
    user_id = user.id if user.role == "operator" else None
    reports = crud.get_operator_reports(db, user_id)
    result = []
    for r in reports:
        result.append({
            "id": r.id,
            "user_id": r.user_id,
            "unit_id": r.unit_id,
            "unit_name": r.unit.name if r.unit else None,
            "timestamp": r.timestamp,
            "description": r.description,
            "sensor_name": r.sensor_name,
            "reported_issue": r.reported_issue,
            "risk_impact": r.risk_impact,
            "status": r.status,
        })
    return result