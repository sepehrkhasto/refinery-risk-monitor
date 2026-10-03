"""
Anomaly detection service using ensemble of multiple algorithms.
Supports per-unit model storage, rich feature set (39+ columns), and thread-safe
training/prediction. Outputs include normalized anomaly score and severity level.
"""

import os
import sys
import warnings
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import numpy as np
import joblib
from sqlalchemy.orm import Session

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.base import BaseEstimator, OutlierMixin

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)
from backend.db.crud import get_all_features_for_unit
from backend.logger import logger

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'anomaly')

ANOMALY_FEATURE_COLUMNS = [
    'temp_in', 'temp_out', 'press_in', 'press_out', 'flow', 'level', 'vib',
    'bearing', 'rpm', 'current', 'torque', 'seal', 'valve1', 'valve2', 'gas',
    'conduct', 'ph', 'coil_temp', 'steam', 'oil_level', 'cooling_temp',
    'ambient_temp', 'humidity', 'wind', 'atm', 'rain', 'solar',
    'last_service_days', 'comp_age', 'repl_count', 'mtbf', 'mttr', 'repair_cost',
    'last_incident_days', 'safety_score', 'risk_lag1', 'risk_lag2',
    'vib_roll_mean', 'temp_out_roll_std'
]


class EnsembleAnomalyDetector(BaseEstimator, OutlierMixin):
    """
    Weighted ensemble of multiple anomaly detection algorithms.
    Uses fixed normalization parameters fitted on training data to avoid batch-dependent scaling.
    """

    def __init__(
        self,
        contamination: float = 0.1,
        random_state: int = 42,
        algorithms: Optional[List[str]] = None
    ):
        self.contamination = contamination
        self.random_state = random_state
        self.algorithms = algorithms or [
            'isolation_forest', 'one_class_svm', 'lof', 'elliptic_envelope'
        ]
        self.models: Dict[str, Any] = {}
        self.weights: Dict[str, float] = {}
        self.thresholds: Dict[str, float] = {}
        self._fitted = False
        self._norm_params: Dict[str, Tuple[float, float]] = {}

    def _init_model(self, name: str) -> Optional[Any]:
        if name == 'isolation_forest':
            return IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
                n_estimators=200,
                max_samples='auto',
                bootstrap=True
            )
        elif name == 'one_class_svm':
            return OneClassSVM(
                nu=self.contamination,
                kernel='rbf',
                gamma='auto'
            )
        elif name == 'lof':
            return LocalOutlierFactor(
                contamination=self.contamination,
                n_neighbors=20,
                novelty=True
            )
        elif name == 'elliptic_envelope':
            return EllipticEnvelope(
                contamination=self.contamination,
                random_state=self.random_state,
                support_fraction=0.7
            )
        else:
            return None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'EnsembleAnomalyDetector':
        if X.shape[0] < 20:
            logger.warning(f"EnsembleAnomalyDetector.fit: insufficient samples ({X.shape[0]})")
            return self

        self.models = {}
        for name in self.algorithms:
            model = self._init_model(name)
            if model is None:
                continue
            try:
                model.fit(X)
                self.models[name] = model
                logger.debug(f"Trained {name} on {X.shape[0]} samples")
            except Exception as e:
                logger.warning(f"Failed to train {name}: {e}")

        if not self.models:
            raise ValueError("No anomaly detection algorithm could be trained")

        n_models = len(self.models)
        self.weights = {name: 1.0 / n_models for name in self.models}

        for name, model in self.models.items():
            try:
                if hasattr(model, 'score_samples'):
                    scores = model.score_samples(X)
                elif hasattr(model, 'decision_function'):
                    scores = model.decision_function(X)
                else:
                    scores = -model.predict(X)
                self._norm_params[name] = (float(np.min(scores)), float(np.max(scores)))
                self.thresholds[name] = np.percentile(scores, self.contamination * 100)
            except Exception as e:
                logger.warning(f"Could not compute params for {name}: {e}")
                self._norm_params[name] = (0.0, 1.0)
                self.thresholds[name] = 0.0

        self._fitted = True
        self.threshold_optimal = np.percentile(self.decision_function(X), (1 - self.contamination) * 100)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Weighted average of normalized decision functions using fixed training parameters.
        Higher score = more normal, lower score = more anomalous.
        """
        if not self._fitted:
            raise ValueError("EnsembleAnomalyDetector not fitted yet")

        all_scores = []
        for name, model in self.models.items():
            try:
                if hasattr(model, 'score_samples'):
                    scores = model.score_samples(X)
                elif hasattr(model, 'decision_function'):
                    scores = model.decision_function(X)
                else:
                    scores = -model.predict(X)

                min_s, max_s = self._norm_params[name]
                if max_s - min_s > 1e-8:
                    scores_norm = (scores - min_s) / (max_s - min_s)
                else:
                    scores_norm = np.ones_like(scores) * 0.5
                all_scores.append(self.weights[name] * scores_norm)
            except Exception as e:
                logger.debug(f"Decision function failed for {name}: {e}")
                continue

        if not all_scores:
            return np.zeros(X.shape[0])

        combined = np.sum(all_scores, axis=0)
        return np.clip(combined, 0.0, 1.0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        threshold = getattr(self, 'threshold_optimal', 0.5)
        return np.where(scores < threshold, -1, 1)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        return 1.0 - scores

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        joblib.dump(self, os.path.join(path, 'ensemble_model.pkl'))
        logger.info(f"EnsembleAnomalyDetector saved to {path}")

    @classmethod
    def load(cls, path: str) -> Optional['EnsembleAnomalyDetector']:
        model_path = os.path.join(path, 'ensemble_model.pkl')
        if os.path.exists(model_path):
            try:
                return joblib.load(model_path)
            except Exception as e:
                logger.error(f"Failed to load ensemble model: {e}")
        return None


@dataclass
class AnomalyReport:
    timestamp: str
    unit_name: str
    total_anomalies: int
    anomaly_rate: float
    critical_anomalies: int
    warning_anomalies: int
    details: List[Dict[str, Any]]
    summary: Dict[str, Any]
    recommendations: List[str]


class AnomalyDetectionService:
    def __init__(self, base_model_dir: str = MODEL_DIR):
        self.base_model_dir = base_model_dir
        self._models: Dict[str, EnsembleAnomalyDetector] = {}
        self._scalers: Dict[str, RobustScaler] = {}
        self._locks: Dict[str, threading.RLock] = {}
        self._global_lock = threading.RLock()

    def _get_unit_path(self, unit_name: str) -> str:
        safe_name = unit_name.replace(' ', '_').replace('/', '_')
        return os.path.join(self.base_model_dir, safe_name)

    def _get_lock(self, unit_name: str) -> threading.RLock:
        with self._global_lock:
            if unit_name not in self._locks:
                self._locks[unit_name] = threading.RLock()
            return self._locks[unit_name]

    def train(self, db: Session, unit_name: str, force_retrain: bool = False) -> bool:
        lock = self._get_lock(unit_name)
        with lock:
            unit_path = self._get_unit_path(unit_name)
            if not force_retrain and os.path.exists(os.path.join(unit_path, 'ensemble_model.pkl')):
                logger.info(f"Anomaly model for {unit_name} already exists, loading...")
                self._load_unit_model(unit_name)
                return self._models.get(unit_name) is not None

            features_list = get_all_features_for_unit(db, unit_name, limit=500)
            if len(features_list) < 30:
                logger.warning(f"Not enough data for {unit_name}: {len(features_list)} samples")
                return False

            X = []
            for row in features_list:
                vec = [row.get(col, 0.0) for col in ANOMALY_FEATURE_COLUMNS]
                vec = [0.0 if (v is None or np.isnan(v) or np.isinf(v)) else float(v) for v in vec]
                X.append(vec)
            X = np.array(X, dtype=np.float32)

            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)

            detector = EnsembleAnomalyDetector(contamination=0.1, random_state=42)
            try:
                detector.fit(X_scaled)
                os.makedirs(unit_path, exist_ok=True)
                detector.save(unit_path)
                joblib.dump(scaler, os.path.join(unit_path, 'scaler.pkl'))
                self._models[unit_name] = detector
                self._scalers[unit_name] = scaler
                logger.info(f"Anomaly detection model trained for {unit_name} on {len(X)} samples")
                return True
            except Exception as e:
                logger.error(f"Failed to train anomaly model for {unit_name}: {e}")
                return False

    def _load_unit_model(self, unit_name: str) -> bool:
        unit_path = self._get_unit_path(unit_name)
        detector = EnsembleAnomalyDetector.load(unit_path)
        if detector is None:
            return False
        scaler_path = os.path.join(unit_path, 'scaler.pkl')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        else:
            scaler = RobustScaler()
        self._models[unit_name] = detector
        self._scalers[unit_name] = scaler
        return True

    def detect_single(
        self,
        db: Session,
        unit_name: str,
        features_dict: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        lock = self._get_lock(unit_name)
        with lock:
            if unit_name not in self._models:
                if not self._load_unit_model(unit_name):
                    success = self.train(db, unit_name)
                    if not success:
                        return {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "unit_name": unit_name,
                            "is_anomaly": 0,
                            "anomaly_score": 0.5,
                            "severity": "UNKNOWN",
                            "error": "Model not available"
                        }

            detector = self._models.get(unit_name)
            scaler = self._scalers.get(unit_name)
            if detector is None or scaler is None:
                return {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "unit_name": unit_name,
                    "is_anomaly": 0,
                    "anomaly_score": 0.5,
                    "severity": "UNKNOWN",
                    "error": "Model not loaded"
                }

            if features_dict is None:
                from backend.db.crud import get_latest_features_for_unit
                features_dict = get_latest_features_for_unit(db, unit_name)
                if not features_dict:
                    return {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "unit_name": unit_name,
                        "is_anomaly": 0,
                        "anomaly_score": 0.5,
                        "severity": "UNKNOWN",
                        "error": "No features available"
                    }

            vec = [features_dict.get(col, 0.0) for col in ANOMALY_FEATURE_COLUMNS]
            vec = [0.0 if (v is None or np.isnan(v) or np.isinf(v)) else float(v) for v in vec]
            X = np.array(vec, dtype=np.float32).reshape(1, -1)
            X_scaled = scaler.transform(X)

            prediction = detector.predict(X_scaled)[0]
            anomaly_score_raw = detector.score_samples(X_scaled)[0]
            anomaly_score = 1.0 - anomaly_score_raw
            anomaly_score = np.clip(anomaly_score, 0.0, 1.0)

            if prediction == -1:
                if anomaly_score > 0.8:
                    severity = "CRITICAL"
                elif anomaly_score > 0.6:
                    severity = "HIGH"
                elif anomaly_score > 0.4:
                    severity = "MEDIUM"
                else:
                    severity = "LOW"
            else:
                severity = "NORMAL"

            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "unit_name": unit_name,
                "is_anomaly": int(prediction == -1),
                "anomaly_score": float(anomaly_score),
                "severity": severity
            }

    def detect_batch(
        self,
        db: Session,
        unit_name: str,
        limit: int = 50
    ) -> Dict[str, Any]:
        lock = self._get_lock(unit_name)
        with lock:
            if unit_name not in self._models:
                if not self._load_unit_model(unit_name):
                    self.train(db, unit_name)

        if unit_name not in self._models:
            if unit_name not in self._models:
                if not self._load_unit_model(unit_name):
                    if not self.train(db, unit_name):
                        return {
                            "success": False,
                            "error": f"Model for {unit_name} not available and training failed",
                            "unit_name": unit_name,
                            "total_anomalies": 0,
                            "anomaly_rate": 0.0,
                            "critical_anomalies": 0,
                            "warning_anomalies": 0,
                            "recommendations": ["Please run: python scripts/train_advanced_models.py"],
                            "details": []
                        }

        features_list = get_all_features_for_unit(db, unit_name, limit=limit)
        if not features_list:
            return {"error": "No data available"}

        results = []
        for feats in features_list:
            vec = [feats.get(col, 0.0) for col in ANOMALY_FEATURE_COLUMNS]
            vec = [0.0 if (v is None or np.isnan(v) or np.isinf(v)) else float(v) for v in vec]
            X = np.array(vec, dtype=np.float32).reshape(1, -1)
            scaler = self._scalers.get(unit_name)
            if scaler:
                X_scaled = scaler.transform(X)
                detector = self._models[unit_name]
                pred = detector.predict(X_scaled)[0]
                score_raw = detector.score_samples(X_scaled)[0]
                anomaly_score = 1.0 - score_raw
                is_anomaly = pred == -1
                if is_anomaly:
                    if anomaly_score > 0.8:
                        sev = "CRITICAL"
                    elif anomaly_score > 0.6:
                        sev = "HIGH"
                    elif anomaly_score > 0.4:
                        sev = "MEDIUM"
                    else:
                        sev = "LOW"
                else:
                    sev = "NORMAL"
                results.append({
                    "timestamp": feats.get('timestamp',  datetime.now(timezone.utc).isoformat()),
                    "is_anomaly": int(is_anomaly),
                    "anomaly_score": float(anomaly_score),
                    "severity": sev
                })
            else:
                return {"error": f"Scaler not available for unit {unit_name}", "success": False}
        total = len(results)
        anomalies = [r for r in results if r['is_anomaly']]
        critical = [r for r in anomalies if r['severity'] == 'CRITICAL']
        warning = [r for r in anomalies if r['severity'] in ('HIGH', 'MEDIUM')]

        recommendations = []
        if critical:
            recommendations.append(f"Found {len(critical)} critical anomalies - immediate inspection required")
        if len(anomalies) / total > 0.2:
            recommendations.append("High anomaly rate - consider recalibrating model or investigating process")
        if not recommendations:
            recommendations.append("No critical anomalies detected")

        return {
            "success": True,
            "unit_name": unit_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_anomalies": len(anomalies),
            "anomaly_rate": round(len(anomalies) / total, 4) if total else 0,
            "critical_anomalies": len(critical),
            "warning_anomalies": len(warning),
            "recommendations": recommendations,
            "details": results[-20:]
        }


_anomaly_service = AnomalyDetectionService()


def detect_anomalies_professional(
    db: Session,
    unit_name: str,
    limit: int = 60
) -> Dict[str, Any]:
    return _anomaly_service.detect_batch(db, unit_name, limit=limit)