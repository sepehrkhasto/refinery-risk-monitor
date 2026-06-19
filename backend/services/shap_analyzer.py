"""
Professional SHAP analysis service for model interpretability.
Uses the LightGBM P50 quantile model (or a fallback RandomForest) to explain
which features contribute most to risk predictions.
Thread-safe and compatible with fixed feature columns.
"""

import os
import sys
import warnings
import threading
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import joblib
import shap
from sqlalchemy.orm import Session

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
import lightgbm as lgb

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)
from backend.db.crud import get_latest_features_for_unit
from backend.logger import logger

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
QUANTILE_MODEL_DIR = os.path.join(MODEL_DIR, 'quantile')

# Import fixed feature columns from predictor_ml
try:
    from backend.services.predictor_ml import FIXED_FEATURE_COLUMNS
except ImportError:
    FIXED_FEATURE_COLUMNS = [
        'temp_in', 'temp_out', 'press_in', 'press_out', 'flow', 'level', 'vib',
        'bearing', 'rpm', 'current', 'torque', 'seal', 'valve1', 'valve2', 'gas',
        'conduct', 'ph', 'coil_temp', 'steam', 'oil_level', 'cooling_temp',
        'ambient_temp', 'humidity', 'wind', 'atm', 'rain', 'solar',
        'last_service_days', 'comp_age', 'repl_count', 'mtbf', 'mttr', 'repair_cost',
        'last_incident_days', 'safety_score', 'risk_lag1', 'risk_lag2',
        'vib_roll_mean', 'temp_out_roll_std'
    ]


# ============================================================================
# Model Loader (thread-safe)
# ============================================================================
class SHAPModelLoader:
    """
    Loads a trained LightGBM model (p50 quantile) for SHAP explanations.
    Falls back to a RandomForest if no quantile model exists.
    """

    def __init__(self):
        self._model = None
        self._scaler = None
        self._explainer = None
        self._feature_names = FIXED_FEATURE_COLUMNS
        self._lock = threading.RLock()

    def load_or_train_fallback(self, db: Session, unit_name: str) -> bool:
        with self._lock:
            unit_path = os.path.join(QUANTILE_MODEL_DIR, unit_name.replace(' ', '_'))
            lgb_path = os.path.join(unit_path, 'lgb_q50.pkl')
            scaler_path = os.path.join(unit_path, 'scaler.pkl')

            if os.path.exists(lgb_path) and os.path.exists(scaler_path):
                try:
                    self._model = joblib.load(lgb_path)
                    self._scaler = joblib.load(scaler_path)
                    self._explainer = shap.TreeExplainer(self._model)
                    logger.info(f"SHAP model loaded from {lgb_path}")
                    return True
                except Exception as e:
                    logger.warning(f"Failed to load LightGBM for SHAP: {e}, falling back")

            logger.info(f"Training fallback RandomForest for SHAP on {unit_name}")
            try:
                from backend.db.crud import get_aligned_features_and_risk
                features_list, risk_scores = get_aligned_features_and_risk(db, unit_name, limit=300)
                if len(features_list) < 30 or len(risk_scores) < 30:
                    logger.warning("Not enough data for fallback SHAP model")
                    return False


                min_len = min(len(features_list), len(risk_scores))
                X = []
                y = []
                for i in range(min_len):
                    vec = [features_list[i].get(col, 0.0) for col in self._feature_names]
                    vec = [0.0 if np.isnan(v) or np.isinf(v) else v for v in vec]
                    X.append(vec)
                    y.append(risk_scores[i]) 
                X = np.array(X)
                y = np.array(y)

                self._scaler = RobustScaler()
                X_scaled = self._scaler.fit_transform(X)

                self._model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
                self._model.fit(X_scaled, y)
                self._explainer = shap.TreeExplainer(self._model)
                logger.info(f"Fallback RandomForest trained for SHAP on {unit_name}")
                return True
            except Exception as e:
                logger.error(f"Fallback SHAP model training failed: {e}")
                return False

    def get_explainer(self):
        return self._explainer

    def get_model(self):
        return self._model

    def get_scaler(self):
        return self._scaler

    def get_feature_names(self):
        return self._feature_names


# ============================================================================
# SHAP Analysis Service (per-unit, thread-safe)
# ============================================================================
class SHAPAnalysisService:
    """
    Provides SHAP explanations for risk predictions.
    Maintains a separate loader per unit.
    """

    def __init__(self):
        self._loaders: Dict[str, SHAPModelLoader] = {}
        self._locks: Dict[str, threading.RLock] = {}
        self._global_lock = threading.RLock()

    def _get_loader(self, unit_name: str, db: Optional[Session] = None) -> Optional[SHAPModelLoader]:
        """Get or create a model loader for the unit."""
        with self._global_lock:
            if unit_name not in self._loaders:
                self._loaders[unit_name] = SHAPModelLoader()
                self._locks[unit_name] = threading.RLock()
            loader = self._loaders[unit_name]
            # If model not loaded yet and db is provided, try to load/train
            if loader.get_explainer() is None and db is not None:
                with self._locks[unit_name]:
                    if loader.get_explainer() is None:
                        loader.load_or_train_fallback(db, unit_name)
            return loader

    def explain(
        self,
        db: Session,
        unit_name: str,
        features_dict: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanation for the latest (or provided) feature vector.
        Returns feature impacts, recommendations, and a text visualization.
        """
        loader = self._get_loader(unit_name, db)
        if loader is None or loader.get_explainer() is None:
            return {
                "success": False,
                "error": "SHAP model not available for this unit",
                "recommendations": ["Train the quantile model first or ensure data availability"]
            }

        # Get features if not provided
        if features_dict is None:
            features_dict = get_latest_features_for_unit(db, unit_name)
            if not features_dict:
                return {
                    "success": False,
                    "error": "No feature data available for unit",
                    "unit_name": unit_name
                }

        # Build feature vector in fixed order
        feature_names = loader.get_feature_names()
        feature_values = []
        for col in feature_names:
            val = features_dict.get(col, 0.0)
            if val is None or np.isnan(val) or np.isinf(val):
                val = 0.0
            feature_values.append(float(val))
        X = np.array(feature_values).reshape(1, -1)

        # Scale if scaler exists
        scaler = loader.get_scaler()
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X

        # Get SHAP values
        explainer = loader.get_explainer()
        model = loader.get_model()
        shap_values = explainer.shap_values(X_scaled)

        # For TreeExplainer, shap_values shape depends on output dimension
        if isinstance(shap_values, list):
            # For multi-output, take first output
            shap_values = shap_values[0]
        if len(shap_values.shape) == 3:
            shap_values = shap_values[0, :, 0]  # [samples, features, outputs]
        elif len(shap_values.shape) == 2:
            shap_values = shap_values[0]  # [samples, features]
        elif len(shap_values.shape) == 1:
            shap_values = shap_values  # already 1D

        # Get base value (expected value)
        base_value = float(explainer.expected_value)
        if isinstance(base_value, (list, np.ndarray)):
            base_value = float(base_value[0])

        # Get model prediction
        predicted_risk = float(model.predict(X_scaled)[0])

        # Sort features by absolute SHAP value
        feature_impacts = []
        for i, name in enumerate(feature_names):
            if i >= len(shap_values):
                break
            shap_val = float(shap_values[i])
            abs_shap = abs(shap_val)
            feature_impacts.append({
                "feature": name,
                "shap_value": round(shap_val, 4),
                "abs_shap": round(abs_shap, 4),
                "current_value": round(feature_values[i], 2),
                "impact_direction": "INCREASES" if shap_val > 0 else "DECREASES",
                "impact_level": self._impact_level(abs_shap)
            })
        # Sort by absolute SHAP descending
        feature_impacts.sort(key=lambda x: x['abs_shap'], reverse=True)

        # Top positive and negative
        top_positive = [f for f in feature_impacts if f['shap_value'] > 0][:5]
        top_negative = [f for f in feature_impacts if f['shap_value'] < 0][:5]

        # Generate recommendations
        recommendations = self._generate_recommendations(top_positive, predicted_risk)

        # Generate text visualization (like waterfall summary)
        text_viz = self._generate_text_viz(predicted_risk, base_value, top_positive, top_negative)

        return {
            "success": True,
            "unit_name": unit_name,
            "timestamp": datetime.now(timezone.utc).isoformat(), 
            "predicted_risk": round(predicted_risk, 2),
            "base_value": round(base_value, 2),
            "feature_impacts": feature_impacts[:15],  # top 15 features
            "top_positive_features": top_positive,
            "top_negative_features": top_negative,
            "recommendations": recommendations,
            "text_visualization": text_viz
        }

    def _impact_level(self, abs_shap: float) -> str:
        if abs_shap > 10:
            return "VERY_HIGH"
        elif abs_shap > 5:
            return "HIGH"
        elif abs_shap > 2:
            return "MEDIUM"
        elif abs_shap > 0.5:
            return "LOW"
        else:
            return "NEGLIGIBLE"

    def _generate_recommendations(self, top_positive: List[Dict], predicted_risk: float) -> List[str]:
        recs = []
        if predicted_risk > 75:
            recs.append("CRITICAL: Risk level is very high - immediate action required")
        elif predicted_risk > 60:
            recs.append("HIGH risk - schedule maintenance and investigate top contributing features")
        elif predicted_risk > 45:
            recs.append("ELEVATED risk - monitor closely")

        feature_rec_map = {
            'temp_out': "High outlet temperature - check cooling system",
            'vib': "Excessive vibration - inspect rotating equipment",
            'pressure_out': "Pressure deviation - check valves and pumps",
            'flow': "Flow rate anomaly - verify pump operation",
            'bearing': "Bearing temperature high - lubrication or wear issue",
            'gas': "Gas concentration abnormal - check seals and ventilation",
            'current': "Motor current high - possible mechanical overload",
            'vib_roll_mean': "Vibration trend increasing - schedule vibration analysis",
            'temp_out_roll_std': "Temperature instability - control loop may need tuning"
        }
        for f in top_positive[:3]:
            if f['feature'] in feature_rec_map:
                recs.append(feature_rec_map[f['feature']])
        return recs[:5]

    def _generate_text_viz(self, predicted_risk, base_value, top_pos, top_neg):
        lines = []
        lines.append("=" * 60)
        lines.append(f"SHAP EXPLANATION - Predicted Risk: {predicted_risk:.1f} (Base: {base_value:.1f})")
        lines.append("=" * 60)
        lines.append("\n[INCREASING RISK]")
        for f in top_pos[:5]:
            lines.append(f"  {f['feature']:25s} +{f['shap_value']:6.2f}  (value={f['current_value']:.1f})")
        lines.append("\n[DECREASING RISK]")
        for f in top_neg[:5]:
            lines.append(f"  {f['feature']:25s} {f['shap_value']:6.2f}  (value={f['current_value']:.1f})")
        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================================
# Singleton service and backward-compatible function
# ============================================================================
_shap_service = SHAPAnalysisService()


def get_shap_values_professional(
    db: Session,
    unit_name: str,
    use_ensemble: bool = True,
    return_detailed: bool = True
) -> Dict[str, Any]:
    """
    Backward-compatible function for routes.
    Returns SHAP explanation for the given unit.
    """
    result = _shap_service.explain(db, unit_name)
    if not result.get("success"):
        return {"error": result.get("error", "SHAP analysis failed")}
    if return_detailed:
        return result
    else:
        # Simplified output (list of feature, shap_value)
        return [
            {"feature": f['feature'], "shap_value": f['shap_value']}
            for f in result.get('feature_impacts', [])
        ]