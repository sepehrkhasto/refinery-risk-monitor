"""
Root Cause Analysis endpoints.
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from backend.api.dependencies import get_db, get_current_user   
from backend.services.rootcauseanalyze import (
    get_top_root_causes_professional,
    get_cause_trend_professional,
    predict_root_cause_professional,
    get_complete_rca_analysis
)

router = APIRouter()


@router.get("/root-cause/top")
def top_root_causes(
    unit_name: str = None,
    limit: int = 5,
    db: Session = Depends(get_db),
    user = Depends(get_current_user)  
):
    return get_top_root_causes_professional(db, unit_name, limit=limit)


@router.get("/root-cause/trend")
def cause_trend(
    cause: str,
    days: int = 30,
    db: Session = Depends(get_db),
    user = Depends(get_current_user)  
):
    return get_cause_trend_professional(db, cause, days)


@router.get("/root-cause/predict")
def predict_root_cause(
    unit_name: str,
    db: Session = Depends(get_db),
    user = Depends(get_current_user)  
):
    return predict_root_cause_professional(db, unit_name)


@router.get("/root-cause/complete")
def complete_rca(
    unit_name: str = None,
    db: Session = Depends(get_db),
    user = Depends(get_current_user)   
):
    return get_complete_rca_analysis(db, unit_name)