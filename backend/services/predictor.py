"""
Risk prediction service - Unified interface for risk forecasting.
Uses the professional quantile predictor (predictor_ml) as primary engine,
with legacy LSTM and simple ensemble fallbacks for backward compatibility.
All methods are thread-safe and use the fixed 39-feature column set.

This module respects PREDICTION_STEP_SECONDS from config for time interpretation
(in the returned metadata, not affecting ML model internal logic).
"""

import warnings
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
from sqlalchemy.orm import Session

from backend.db.crud import get_unit_risk_history, get_latest_features_for_unit
from backend.services.predictor_ml import (
    predict_risk_quantile_professional as _quantile_predict,
    QuantilePredictionService
)
from backend.config import settings
from backend.logger import logger

# Optional legacy models (will be lazy-loaded if needed)
_legacy_lstm = None
_legacy_ensemble = None


@dataclass
class RiskPredictionReport:
    """Structured prediction output with time scale metadata."""
    timestamp: str
    unit_name: str
    current_risk: float
    quantile_p10: float
    quantile_p50: float
    quantile_p90: float
    ensemble_prediction: Optional[float]
    lstm_prediction: Optional[float]
    recommendations: List[str]
    step_seconds: int  # Time duration per prediction step (from config)


class RiskPredictionService:
    """
    Unified risk prediction service.
    Primary method uses quantile regression (P10/P50/P90).
    Legacy methods (ensemble, LSTM) are available as fallbacks.
    """

    def __init__(self, use_legacy_fallback: bool = True):
        self.use_legacy_fallback = use_legacy_fallback
        self._quantile_service = QuantilePredictionService()
        self._step_seconds = getattr(settings, "PREDICTION_STEP_SECONDS", 3)

    def predict_quantile(
        self,
        db: Session,
        unit_name: str,
        steps: int = 6,
        features: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Primary prediction method using quantile regression.
        Returns P10, P50, P90 for each step.
        """
        return _quantile_predict(db, unit_name, steps=steps)

    def predict_with_report(
        self,
        db: Session,
        unit_name: str,
        steps: int = 6
    ) -> RiskPredictionReport:
        """
        Generate a comprehensive prediction report including quantiles
        and (optionally) legacy ensemble/LSTM predictions for comparison.
        """
        quantile_result = _quantile_predict(db, unit_name, steps=steps)
        if "error" in quantile_result:
            # Fallback: use current risk as constant prediction
            risk_history = get_unit_risk_history(db, unit_name, limit=1)
            current_risk = risk_history[0].risk_score if risk_history else 50.0
            return RiskPredictionReport(
                timestamp=datetime.now(timezone.utc).isoformat(),
                unit_name=unit_name,
                current_risk=current_risk,
                quantile_p10=current_risk,
                quantile_p50=current_risk,
                quantile_p90=current_risk,
                ensemble_prediction=None,
                lstm_prediction=None,
                recommendations=["Quantile model unavailable - using current risk"],
                step_seconds=self._step_seconds
            )

        predictions = quantile_result.get("predictions", [])
        if predictions:
            first_step = predictions[0]
            p10 = first_step.get("p10", 50.0)
            p50 = first_step.get("p50", 50.0)
            p90 = first_step.get("p90", 50.0)
        else:
            p10 = p50 = p90 = quantile_result.get("current_risk", 50.0)

        # Optionally compute legacy predictions
        ensemble_pred = None
        lstm_pred = None
        if self.use_legacy_fallback:
            try:
                ensemble_pred = self._ensemble_predict_fallback(db, unit_name)
            except Exception as e:
                logger.debug(f"Ensemble fallback failed: {e}")
            try:
                lstm_pred = self._lstm_predict_fallback(db, unit_name)
            except Exception as e:
                logger.debug(f"LSTM fallback failed: {e}")

        recommendations = quantile_result.get("recommendations", [])
        if not recommendations and p50 > 70:
            recommendations.append("High risk predicted - take action")

        return RiskPredictionReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            unit_name=unit_name,
            current_risk=quantile_result.get("current_risk", 50.0),
            quantile_p10=p10,
            quantile_p50=p50,
            quantile_p90=p90,
            ensemble_prediction=ensemble_pred,
            lstm_prediction=lstm_pred,
            recommendations=recommendations,
            step_seconds=self._step_seconds
        )

    def _ensemble_predict_fallback(self, db: Session, unit_name: str) -> Optional[float]:
        """Simple ensemble fallback using historical average."""
        risk_history = get_unit_risk_history(db, unit_name, limit=20)
        if len(risk_history) < 5:
            return None
        recent_risks = [r.risk_score for r in risk_history[-10:]]
        return float(np.mean(recent_risks))

    def _lstm_predict_fallback(self, db: Session, unit_name: str) -> Optional[float]:
        """LSTM fallback using simple moving average (placeholder)."""
        risk_history = get_unit_risk_history(db, unit_name, limit=10)
        if len(risk_history) < 3:
            return None
        if len(risk_history) >= 2:
            last_two = [risk_history[-2].risk_score, risk_history[-1].risk_score]
            trend = last_two[1] - last_two[0]
            return last_two[1] + trend
        return risk_history[-1].risk_score


# Singleton for backward compatibility
_prediction_service = RiskPredictionService()


def predict_unit_risk_professional(
    db: Session,
    unit_name: str,
    limit: int = 50,
    steps: int = 6
) -> Dict[str, Any]:
    """
    Backward-compatible function for routes that expect the old interface.
    Uses quantile predictor for consistency.
    """
    return _prediction_service.predict_quantile(db, unit_name, steps=steps)


def get_risk_prediction_report(
    db: Session,
    unit_name: str,
    steps: int = 6
) -> Dict[str, Any]:
    """Return a detailed report as dict (for API endpoints)."""
    report = _prediction_service.predict_with_report(db, unit_name, steps)
    return {
        "timestamp": report.timestamp,
        "unit_name": report.unit_name,
        "current_risk": report.current_risk,
        "quantile": {
            "p10": report.quantile_p10,
            "p50": report.quantile_p50,
            "p90": report.quantile_p90
        },
        "ensemble_prediction": report.ensemble_prediction,
        "lstm_prediction": report.lstm_prediction,
        "recommendations": report.recommendations,
        "step_seconds": report.step_seconds   # New field for frontend
    }