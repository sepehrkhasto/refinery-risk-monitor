"""
Risk prediction endpoints (ML and overall) – final version with correct route order.
Includes simple in-memory cache for latest risk per unit and latest features.
Thread-safe cache with RLock.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import desc
from backend.api.dependencies import get_db, get_current_user
from backend.db.db_models import RiskAssessment, Unit
from backend.db import crud
from backend.services.predictor_ml import predict_risk_quantile_professional
from backend.services.trend_analyzer import analyze_unit_trend_professional
from backend.config import settings
from backend.logger import logger
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import threading

router = APIRouter()

# Simple in-memory cache for latest risk per unit
_latest_risk_cache: Dict[str, Dict[str, Any]] = {}
_latest_features_cache: Dict[str, Dict[str, Any]] = {}
_cache_ttl_seconds = 2  # Cache expires after 2 seconds
_cache_lock = threading.RLock()  # Protect cache access


def _get_cached_latest_risk(unit_id: str, db: Session) -> Optional[Dict[str, Any]]:
    """Get latest risk from cache or DB (thread-safe)."""
    global _latest_risk_cache
    now = datetime.now(timezone.utc)
    with _cache_lock:
        if unit_id in _latest_risk_cache:
            cached = _latest_risk_cache[unit_id]
            if (now - cached["timestamp"]).total_seconds() < _cache_ttl_seconds:
                return cached["data"]
        # Cache miss or expired
        latest = db.query(RiskAssessment).filter(
            RiskAssessment.unit_id == unit_id
        ).order_by(desc(RiskAssessment.timestamp)).first()
        if latest:
            data = {
                "risk_score": latest.risk_score,
                "status": latest.status,
                "root_cause": latest.root_cause,
                "timestamp": latest.timestamp
            }
            _latest_risk_cache[unit_id] = {"data": data, "timestamp": now}
            return data
        return None


def _get_cached_latest_features(unit_name: str, db: Session) -> Optional[Dict[str, float]]:
    """Get latest features from cache or DB (thread-safe)."""
    global _latest_features_cache
    now = datetime.now(timezone.utc)
    with _cache_lock:
        if unit_name in _latest_features_cache:
            cached = _latest_features_cache[unit_name]
            if (now - cached["timestamp"]).total_seconds() < _cache_ttl_seconds:
                return cached["data"]
        features = crud.get_latest_features_for_unit(db, unit_name)
        if features:
            _latest_features_cache[unit_name] = {"data": features, "timestamp": now}
            return features
        return None


@router.get("/predict/overall")
def predict_overall(
    db: Session = Depends(get_db),
    steps: int = 6,
    user = Depends(get_current_user)
):
    """
    Predict average risk for the entire refinery for multiple future steps.
    Uses cached latest risk per unit for performance.
    """
    units = db.query(Unit).all()
    if not units:
        return {"overall_risk": 0, "unit_count": 0, "prediction": {"p10": [], "p50": [], "p90": []}}

    all_p10 = [[] for _ in range(steps)]
    all_p50 = [[] for _ in range(steps)]
    all_p90 = [[] for _ in range(steps)]
    current_risks = []

    for u in units:
        latest = _get_cached_latest_risk(u.id, db)
        if not latest:
            continue
        current_risk = latest["risk_score"]
        current_risks.append(current_risk)

        try:
            pred = predict_risk_quantile_professional(db, u.name, steps=steps)
            if pred and "predictions" in pred:
                for i, step_data in enumerate(pred["predictions"]):
                    all_p10[i].append(step_data["p10"])
                    all_p50[i].append(step_data["p50"])
                    all_p90[i].append(step_data["p90"])
            else:
                for i in range(steps):
                    all_p10[i].append(current_risk)
                    all_p50[i].append(current_risk)
                    all_p90[i].append(current_risk)
        except Exception as e:
            logger.warning(f"Overall multi-step fallback for {u.name}: {e}")
            for i in range(steps):
                all_p10[i].append(current_risk)
                all_p50[i].append(current_risk)
                all_p90[i].append(current_risk)

    if not current_risks:
        return {"overall_risk": 0, "unit_count": 0, "prediction": {"p10": [], "p50": [], "p90": []}}

    overall_current = round(sum(current_risks) / len(current_risks), 2)
    p10_steps = [round(sum(step) / len(step), 2) for step in all_p10]
    p50_steps = [round(sum(step) / len(step), 2) for step in all_p50]
    p90_steps = [round(sum(step) / len(step), 2) for step in all_p90]

    # Ensure P10 <= P50 <= P90 for every step (safety net)
    for i in range(len(p10_steps)):
        a, b, c = p10_steps[i], p50_steps[i], p90_steps[i]
        sorted_vals = sorted([a, b, c])
        p10_steps[i], p50_steps[i], p90_steps[i] = sorted_vals[0], sorted_vals[1], sorted_vals[2]

    step_seconds = getattr(settings, "PREDICTION_STEP_SECONDS", 3)

    return {
        "overall_risk": overall_current,
        "unit_count": len(units),
        "prediction": {
            "p10": p10_steps,
            "p50": p50_steps,
            "p90": p90_steps
        },
        "step_seconds": step_seconds
    }


@router.get("/predict/ml/{unit_name}")
def predict_ml(
    unit_name: str,
    steps: int = 6,
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    """Quantile prediction (P10/P50/P90) for a specific unit."""
    result = predict_risk_quantile_professional(db, unit_name, steps=steps)
    if "predictions" in result and len(result["predictions"]) > 0:
        unit = crud.UnitRepo.get_by_name(db, unit_name)
        if unit:
            first = result["predictions"][0]
            crud.create_prediction_log(db, unit.id, first["p10"], first["p50"], first["p90"])
    return result


@router.get("/predict/{unit_name}")
def predict_linear(
    unit_name: str,
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    """Deprecated endpoint. Redirect to /predict/ml/{unit_name}."""
    return {
        "status": "deprecated",
        "detail": "Use /api/predict/ml/{unit_name} for accurate predictions."
    }


@router.get("/trend/{unit_name}")
def trend(
    unit_name: str,
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    """Statistical trend analysis for a unit."""
    result = analyze_unit_trend_professional(db, unit_name)
    if "error" in result:
        return {"error": result["error"]}
    return result


@router.get("/predict/accuracy/{unit_name}")
def prediction_accuracy(
    unit_name: str,
    limit: int = 50,
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    """MAE and RMSE accuracy of predictions for a unit."""
    logs = crud.get_prediction_accuracy(db, unit_name, limit)
    if not logs:
        return {"error": "No logs available"}
    mae = float(np.mean([l.error_mae for l in logs]))
    rmse = float(np.sqrt(np.mean([l.error_rmse ** 2 for l in logs])))
    return {
        "unit": unit_name,
        "samples": len(logs),
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2)
    }