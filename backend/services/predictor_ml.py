"""
Quantile prediction service (P10, P50, P90) with adaptive drift learning,
safety net using weighted blending, and thread-safe model storage.
NO on-the-fly training in predict() – models must be pre-trained.
"""

import os
import sys
import pickle
import warnings
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque

import numpy as np
import joblib
from sqlalchemy.orm import Session

from sklearn.preprocessing import RobustScaler
import lightgbm as lgb

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)
from backend.db.crud import get_latest_features_for_unit, get_unit_risk_history, get_aligned_features_and_risk
from backend.config import settings
from backend.logger import logger

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'quantile')

FIXED_FEATURE_COLUMNS = [
    'temp_in', 'temp_out', 'press_in', 'press_out', 'flow', 'level', 'vib',
    'bearing', 'rpm', 'current', 'torque', 'seal', 'valve1', 'valve2', 'gas',
    'conduct', 'ph', 'coil_temp', 'steam', 'oil_level', 'cooling_temp',
    'ambient_temp', 'humidity', 'wind', 'atm', 'rain', 'solar',
    'last_service_days', 'comp_age', 'repl_count', 'mtbf', 'mttr', 'repair_cost',
    'last_incident_days', 'safety_score', 'risk_lag1', 'risk_lag2',
    'vib_roll_mean', 'temp_out_roll_std'
]


class QuantileFeatureEngineer:
    def __init__(self, lookback_window: int = 10):
        self.lookback_window = lookback_window

    def create_features(
        self,
        current_features: Dict[str, float],
        risk_history: List[float],
        sensor_history: Optional[List[Dict]] = None
    ) -> np.ndarray:
        vec = [current_features.get(col, 0.0) for col in FIXED_FEATURE_COLUMNS]
        if len(risk_history) >= 1:
            vec[FIXED_FEATURE_COLUMNS.index('risk_lag1')] = float(risk_history[-1])
        if len(risk_history) >= 2:
            vec[FIXED_FEATURE_COLUMNS.index('risk_lag2')] = float(risk_history[-2])
        vib_vals = [current_features.get('vib', 0.0)]
        if sensor_history:
            for h in sensor_history[-4:]:
                vib_vals.append(h.get('vib', 0.0))
        vec[FIXED_FEATURE_COLUMNS.index('vib_roll_mean')] = float(np.mean(vib_vals))
        temp_vals = [current_features.get('temp_out', 0.0)]
        if sensor_history:
            for h in sensor_history[-4:]:
                temp_vals.append(h.get('temp_out', 0.0))
        vec[FIXED_FEATURE_COLUMNS.index('temp_out_roll_std')] = float(np.std(temp_vals)) if len(temp_vals) > 1 else 0.0
        vec = [0.0 if (np.isnan(v) or np.isinf(v)) else v for v in vec]
        return np.array(vec, dtype=np.float32)


class AdaptiveDriftLearner:
    """
    Robust drift learner using Theil-Sen estimator (more resilient to outliers than polyfit).
    """
    def __init__(self, smoothing_factor: float = 0.3, max_history: int = 100):
        self.smoothing_factor = smoothing_factor
        self.drift_history = deque(maxlen=max_history)
        self.learned_drift = 0.0

    def update(self, risk_history: List[float]) -> None:
        if len(risk_history) < 5:
            return
        recent = risk_history[-10:]
        x = np.arange(len(recent))
        from scipy import stats
        slope = stats.theilslopes(recent, x)[0] if len(recent) >= 4 else 0.0
        self.learned_drift = (self.smoothing_factor * slope +
                              (1 - self.smoothing_factor) * self.learned_drift)
        self.drift_history.append(slope)

    def predict_drift(self, steps: int) -> np.ndarray:
        if steps <= 0:
            return np.array([])
        return np.full(steps, self.learned_drift)


class QuantileRegressor:
    def __init__(self, quantiles: Tuple[float, float, float] = (0.1, 0.5, 0.9)):
        self.quantiles = quantiles
        self.models: Dict[float, lgb.LGBMRegressor] = {}
        self.scaler = RobustScaler()
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'QuantileRegressor':
        if X.shape[0] < 20:
            raise ValueError(f"Not enough samples for training: {X.shape[0]}")
        X_scaled = self.scaler.fit_transform(X)
        for q in self.quantiles:
            params = {
                'n_estimators': 300,
                'max_depth': 6,
                'learning_rate': 0.03 if q in (0.1, 0.9) else 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': 42,
                'verbose': -1,
                'objective': 'quantile',
                'alpha': q
            }
            model = lgb.LGBMRegressor(**params)
            model.fit(X_scaled, y, eval_set=[(X_scaled, y)], callbacks=[lgb.early_stopping(20, verbose=False)])
            self.models[q] = model
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> Dict[float, np.ndarray]:
        if not self.is_fitted:
            raise ValueError("QuantileRegressor not fitted")
        X_scaled = self.scaler.transform(X)
        results = {}
        for q in self.quantiles:
            pred = self.models[q].predict(X_scaled)
            results[q] = np.clip(pred, 0.0, 100.0)
        p10 = results[0.1]
        p50 = results[0.5]
        p90 = results[0.9]

        for i in range(len(p10)):
            if p10[i] > p50[i]:
                p10[i], p50[i] = p50[i], p10[i]
            if p50[i] > p90[i]:
                p50[i], p90[i] = p90[i], p50[i]
            if p10[i] > p50[i]:
                p10[i], p50[i] = p50[i], p10[i]

        for i in range(len(p10)):
            if abs(p10[i] - p50[i]) < 0.01:
                p50[i] = p10[i] + 0.05
                if p50[i] > p90[i]:
                    p90[i] = p50[i] + 0.05
            if p90[i] <= p50[i]:
                p90[i] = p50[i] + 0.1

        results[0.1] = p10
        results[0.5] = p50
        results[0.9] = p90

        return results

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        for q, model in self.models.items():
            joblib.dump(model, os.path.join(path, f'lgb_q{int(q*100)}.pkl'))
        joblib.dump(self.scaler, os.path.join(path, 'scaler.pkl'))
        with open(os.path.join(path, 'quantiles.pkl'), 'wb') as f:
            pickle.dump(self.quantiles, f)
        logger.info(f"QuantileRegressor saved to {path}")

    @classmethod
    def load(cls, path: str) -> Optional['QuantileRegressor']:
        quantiles_path = os.path.join(path, 'quantiles.pkl')
        if not os.path.exists(quantiles_path):
            return None
        with open(quantiles_path, 'rb') as f:
            quantiles = pickle.load(f)
        instance = cls(quantiles)
        scaler_path = os.path.join(path, 'scaler.pkl')
        if os.path.exists(scaler_path):
            instance.scaler = joblib.load(scaler_path)
        for q in quantiles:
            model_path = os.path.join(path, f'lgb_q{int(q*100)}.pkl')
            if os.path.exists(model_path):
                instance.models[q] = joblib.load(model_path)
        if len(instance.models) == len(quantiles):
            instance.is_fitted = True
        return instance


@dataclass
class QuantilePredictionReport:
    timestamp: str
    unit_name: str
    current_risk: float
    predictions: Dict[str, List[float]]
    recommendations: List[str]


class QuantilePredictionService:
    """
    Per-unit quantile prediction service. NO on-the-fly training.
    If model not found, returns fallback (constant risk) and logs error.
    """
    def __init__(self, base_model_dir: str = MODEL_DIR):
        self.base_model_dir = base_model_dir
        self._models: Dict[str, QuantileRegressor] = {}
        self._drift_learners: Dict[str, AdaptiveDriftLearner] = {}
        self._engineers: Dict[str, QuantileFeatureEngineer] = {}
        self._locks: Dict[str, threading.RLock] = {}
        self._global_lock = threading.RLock()
        self._sensor_history_buffer: Dict[str, deque] = {}
        self._buffer_locks: Dict[str, threading.RLock] = {}
        self._global_buffer_lock = threading.RLock()

    def _get_unit_path(self, unit_name: str) -> str:
        safe_name = unit_name.replace(' ', '_').replace('/', '_')
        return os.path.join(self.base_model_dir, safe_name)

    def _get_lock(self, unit_name: str) -> threading.RLock:
        with self._global_lock:
            if unit_name not in self._locks:
                self._locks[unit_name] = threading.RLock()
            return self._locks[unit_name]

    def _get_sensor_history(self, unit_name: str, current_features: Dict[str, float]) -> List[Dict[str, float]]:
        """
        Thread-safe rolling window (maxlen=10) of sensor readings for a unit.
        Used to compute rolling statistics like vib_roll_mean and temp_out_roll_std.
        """
        with self._global_buffer_lock:
            if unit_name not in self._buffer_locks:
                self._buffer_locks[unit_name] = threading.RLock()
            lock = self._buffer_locks[unit_name]

        with lock:
            if unit_name not in self._sensor_history_buffer:
                self._sensor_history_buffer[unit_name] = deque(maxlen=10)
            self._sensor_history_buffer[unit_name].append(current_features)
            return list(self._sensor_history_buffer[unit_name])

    def _load_unit_model(self, unit_name: str) -> bool:
        unit_path = self._get_unit_path(unit_name)
        model = QuantileRegressor.load(unit_path)
        if model is not None:
            self._models[unit_name] = model
            self._drift_learners[unit_name] = AdaptiveDriftLearner()
            self._engineers[unit_name] = QuantileFeatureEngineer()
            return True
        return False

    def train(self, db: Session, unit_name: str, force_retrain: bool = False) -> bool:
        """
        Train quantile model for a unit. Should be called from offline scripts only.
        """
        lock = self._get_lock(unit_name)
        with lock:
            unit_path = self._get_unit_path(unit_name)
            if not force_retrain and os.path.exists(os.path.join(unit_path, 'quantiles.pkl')):
                return self._load_unit_model(unit_name)

            features_list, risk_scores = get_aligned_features_and_risk(db, unit_name, limit=500)

            if len(features_list) < 30:
                logger.warning(f"Not enough aligned data for {unit_name}: {len(features_list)}")
                return False

            engineer = QuantileFeatureEngineer()
            X_list = []

            for i, current_feats in enumerate(features_list):
                hist = risk_scores[:i + 1]
                sensor_hist = features_list[:i + 1] 
                feat_vec = engineer.create_features(current_feats, hist, sensor_hist)
                X_list.append(feat_vec)

            X = np.array(X_list)
            y = np.array(risk_scores[:len(X)]) 

            try:
                quantile_model = QuantileRegressor()
                quantile_model.fit(X, y)
                quantile_model.save(unit_path)
                self._models[unit_name] = quantile_model
                self._drift_learners[unit_name] = AdaptiveDriftLearner()
                self._engineers[unit_name] = engineer
                self._drift_learners[unit_name].update(risk_scores)
                logger.info(f"Quantile model trained for {unit_name} on {len(X)} samples")
                return True
            except Exception as e:
                logger.error(f"Quantile training failed for {unit_name}: {e}")
                return False

    def predict(
        self,
        db: Session,
        unit_name: str,
        steps: int = 6,
        current_features: Optional[Dict[str, float]] = None
    ) -> QuantilePredictionReport:
        """
        Generate quantile predictions. If model not loaded, returns fallback
        (current risk) immediately – no training on the fly.
        """
        lock = self._get_lock(unit_name)
        with lock:
            if unit_name not in self._models:
                if not self._load_unit_model(unit_name):
                    # Fallback: return current risk as constant prediction
                    current_risk = 50.0
                    try:
                        latest = get_unit_risk_history(db, unit_name, limit=1)
                        if latest:
                            current_risk = latest[0].risk_score
                    except Exception as e:
                        logger.error(f"Failed to get latest risk for fallback: {e}")
                    logger.warning(f"Quantile model not available for {unit_name}, using fallback constant risk")
                    return QuantilePredictionReport(
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        unit_name=unit_name,
                        current_risk=current_risk,
                        predictions={
                            'p10': [current_risk] * steps,
                            'p50': [current_risk] * steps,
                            'p90': [current_risk] * steps
                        },
                        recommendations=["Quantile model not available - using current risk"]
                    )

            model = self._models[unit_name]
            drift_learner = self._drift_learners[unit_name]
            engineer = self._engineers[unit_name]

            if current_features is None:
                current_features = get_latest_features_for_unit(db, unit_name)
                if not current_features:
                    raise ValueError(f"Cannot get features for {unit_name}")

            risk_records = get_unit_risk_history(db, unit_name, limit=100)
            risk_scores = [r.risk_score for r in sorted(risk_records, key=lambda x: x.timestamp)]
            if not risk_scores:
                risk_scores = [50.0]
            current_risk = risk_scores[-1]

            drift_learner.update(risk_scores)

            sensor_history = self._get_sensor_history(unit_name, current_features)
            feat_vec = engineer.create_features(current_features, risk_scores, sensor_history)

            X = feat_vec.reshape(1, -1)

            base_pred = model.predict(X)
            model_p10 = float(base_pred[0.1][0])
            model_p50 = float(base_pred[0.5][0])
            model_p90 = float(base_pred[0.9][0])

            diff = abs(current_risk - model_p50)
            if diff > 20:
                blend_weight = min(0.8, diff / 100.0)
                blended_p50 = (blend_weight * current_risk + (1 - blend_weight) * model_p50)
                blended_p10 = (blend_weight * current_risk + (1 - blend_weight) * model_p10)
                blended_p90 = (blend_weight * current_risk + (1 - blend_weight) * model_p90)
            else:
                blended_p50 = model_p50
                blended_p10 = model_p10
                blended_p90 = model_p90

            step_drifts = drift_learner.predict_drift(steps)
            predictions = {'p10': [], 'p50': [], 'p90': []}
            for step in range(1, steps + 1):
                drift = step_drifts[step - 1] * step
                p10 = np.clip(blended_p10 + drift * 0.7, 0, 100)
                p50 = np.clip(blended_p50 + drift * 1.0, 0, 100)
                p90 = np.clip(blended_p90 + drift * 1.3, 0, 100)
                if p10 > p50:
                    p10, p50 = p50, p10
                if p50 > p90:
                    p50, p90 = p90, p50
                predictions['p10'].append(round(p10, 2))
                predictions['p50'].append(round(p50, 2))
                predictions['p90'].append(round(p90, 2))

            recommendations = []
            if current_risk > 80:
                recommendations.append("CRITICAL: Immediate action required")
            elif current_risk > 65:
                recommendations.append("HIGH risk - schedule maintenance")
            elif current_risk > 50:
                recommendations.append("Elevated risk - monitor closely")
            if predictions['p90'][-1] > 80:
                recommendations.append("Forecast indicates critical risk within next steps")
            if diff > 25:
                recommendations.append("Model adjusted due to sudden risk change")

            return QuantilePredictionReport(
                timestamp=datetime.now(timezone.utc).isoformat(),
                unit_name=unit_name,
                current_risk=round(current_risk, 2),
                predictions=predictions,
                recommendations=recommendations
            )


_quantile_service = QuantilePredictionService()


def predict_risk_quantile_professional(
    db: Session,
    unit_name: str,
    steps: int = 6
) -> Dict[str, Any]:
    try:
        report = _quantile_service.predict(db, unit_name, steps=steps)
        predictions_list = []
        for i in range(steps):
            predictions_list.append({
                "step": i + 1,
                "p10": report.predictions['p10'][i],
                "p50": report.predictions['p50'][i],
                "p90": report.predictions['p90'][i]
            })
        step_seconds = getattr(settings, "PREDICTION_STEP_SECONDS", 3)
        return {
            "unit_name": unit_name,
            "timestamp": report.timestamp,
            "current_risk": report.current_risk,
            "predictions": predictions_list,
            "recommendations": report.recommendations,
            "step_seconds": step_seconds
        }
    except Exception as e:
        logger.error(f"Quantile prediction failed for {unit_name}: {e}")
        step_seconds = getattr(settings, "PREDICTION_STEP_SECONDS", 3)
        return {
            "unit_name": unit_name,
            "error": str(e),
            "current_risk": 0,
            "predictions": [{"step": i+1, "p10": 50, "p50": 50, "p90": 50} for i in range(steps)],
            "recommendations": ["Prediction service error - using fallback"],
            "step_seconds": step_seconds
        }