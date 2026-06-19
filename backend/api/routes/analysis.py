"""
Analysis endpoints: anomaly detection, SHAP, association rules.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from backend.api.dependencies import get_db, get_current_user
from backend.services.anomaly_detector import detect_anomalies_professional
from backend.services.shap_analyzer import get_shap_values_professional
from backend.services.apriori_analyzer import get_apriori_rules_professional

router = APIRouter()


@router.get("/anomalies/{unit_name}")
def anomalies(
    unit_name: str,
    limit: int = 60,
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    """Detect anomalies using ensemble model."""
    return detect_anomalies_professional(db, unit_name, limit=limit)


@router.get("/shap/{unit_name}")
def shap(
    unit_name: str,
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    """SHAP values for sensor impact interpretation."""
    result = get_shap_values_professional(db, unit_name)
    return result


@router.get("/apriori")
def apriori(
    min_confidence: float = 0.5,
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    """Association rules (Apriori/FP-Growth)."""
    rules = get_apriori_rules_professional(db, min_confidence)
    return {"rules": rules}