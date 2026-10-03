"""
Root Cause Analysis (RCA) service with weighted cause analysis,
trend detection, and ML-based predictive RCA using rich feature set (39 columns).
Thread-safe and production-ready.
"""

import os
import sys
import pickle
import warnings
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict, Counter

import numpy as np
import joblib
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from sklearn.preprocessing import RobustScaler, LabelEncoder
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)
from backend.db.db_models import RiskAssessment, Unit, EnvironmentData, MaintenanceLog, HSEReport
from backend.logger import logger

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'rca')

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


FIXED_FEATURE_COLUMNS_RCA = [
    'temp_in', 'temp_out', 'press_in', 'press_out', 'flow', 'level', 'vib',
    'bearing', 'rpm', 'current', 'torque', 'seal', 'valve1', 'valve2', 'gas',
    'conduct', 'ph', 'coil_temp', 'steam', 'oil_level', 'cooling_temp',
    'ambient_temp', 'humidity', 'wind', 'atm', 'rain', 'solar',
    'last_service_days', 'comp_age', 'repl_count', 'mtbf', 'mttr', 'repair_cost',
    'last_incident_days', 'safety_score'
]


class CauseSeverity(Enum):
    CRITICAL = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1
    INFO = 0

    @classmethod
    def from_risk_score(cls, risk_score: float):
        if risk_score >= 80:
            return cls.CRITICAL
        if risk_score >= 60:
            return cls.HIGH
        if risk_score >= 40:
            return cls.MEDIUM
        if risk_score >= 20:
            return cls.LOW
        return cls.INFO


@dataclass
class WeightedCause:
    cause: str
    weight: float
    frequency: int
    avg_risk_score: float
    risk_percentage: float


@dataclass
class CauseTrend:
    cause: str
    daily_data: List[Dict[str, Any]]
    trend_direction: str
    trend_strength: float


@dataclass
class PredictiveRCAResult:
    predicted_cause: str
    confidence: float
    top_causes: List[Tuple[str, float]]


@dataclass
class CompleteRCAAnalysis:
    timestamp: str
    unit_name: str
    top_causes: List[Dict[str, Any]]
    cause_trends: Dict[str, Any]
    predictive_rca: Optional[Dict[str, Any]]
    recommendations: List[str]
    summary_statistics: Dict[str, Any]


class WeightedCauseAnalyzer:
    def __init__(self, decay_factor: float = 0.1):
        self.decay_factor = decay_factor
        self.cause_mapping = {
            'cavitation': 'Pump Cavitation',
            'high temp': 'High Temperature',
            'valve stuck': 'Valve Failure',
            'leak': 'Leak Detected',
            'bearing': 'Bearing Failure',
            'vibration': 'High Vibration',
            'pressure': 'Pressure Deviation',
            'flow': 'Flow Reduction',
            'seal': 'Seal Leakage',
            'motor': 'Motor Overheating',
            'catalyst': 'Catalyst Deactivation'
        }

    def _canonicalize(self, cause: str) -> str:
        if not cause:
            return "Unknown"
        cause_lower = cause.lower()
        for key, canonical in self.cause_mapping.items():
            if key in cause_lower:
                return canonical
        return cause

    def _calculate_weight(self, risk_score: float, age_days: float) -> float:
        decay = np.exp(-self.decay_factor * age_days)
        risk_weight = risk_score / 100.0
        return max(0.05, min(1.0, decay * risk_weight))

    def analyze(
        self,
        db: Session,
        unit_name: Optional[str] = None,
        days_back: int = 30,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
        query = db.query(RiskAssessment).filter(
            RiskAssessment.timestamp >= cutoff,
            RiskAssessment.status.in_(['Warning', 'Critical'])
        )
        if unit_name:
            unit = db.query(Unit).filter(Unit.name == unit_name).first()
            if unit:
                query = query.filter(RiskAssessment.unit_id == unit.id)

        records = query.all()
        if not records:
            return []

        cause_weights = defaultdict(float)
        cause_details = defaultdict(lambda: {'count': 0, 'total_risk': 0.0, 'last_ts': None})

        for r in records:
            cause = self._canonicalize(r.root_cause)

            ts = r.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age_days = (datetime.now(timezone.utc) - ts).total_seconds() / 86400.0
            weight = self._calculate_weight(r.risk_score, age_days)
            cause_weights[cause] += weight
            cause_details[cause]['count'] += 1
            cause_details[cause]['total_risk'] += r.risk_score
            if cause_details[cause]['last_ts'] is None or r.timestamp > cause_details[cause]['last_ts']:
                cause_details[cause]['last_ts'] = r.timestamp

        total_weight = sum(cause_weights.values())
        results = []
        for cause, weight in sorted(cause_weights.items(), key=lambda x: x[1], reverse=True)[:limit]:
            details = cause_details[cause]
            results.append({
                "cause": cause,
                "weight": round(weight, 4),
                "frequency": details['count'],
                "avg_risk_score": round(details['total_risk'] / details['count'], 1),
                "risk_percentage": round(weight / total_weight * 100, 1) if total_weight > 0 else 0,
                "last_occurrence": details['last_ts'].isoformat() if details['last_ts'] else None
            })
        return results


class CauseTrendAnalyzer:
    def analyze_trend(
        self,
        db: Session,
        cause: str,
        days: int = 30
    ) -> Dict[str, Any]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        records = db.query(RiskAssessment).filter(
            RiskAssessment.root_cause.ilike(f'%{cause}%'),
            RiskAssessment.timestamp >= cutoff
        ).order_by(RiskAssessment.timestamp.asc()).all()

        daily_counts = defaultdict(int)
        for r in records:
            date_key = r.timestamp.date()
            daily_counts[date_key] += 1

        dates = sorted(daily_counts.keys())
        counts = [daily_counts[d] for d in dates]

        if len(counts) >= 2:
            x = np.arange(len(counts))
            slope = np.polyfit(x, counts, 1)[0]
            if slope > 0.05:
                direction = "INCREASING"
            elif slope < -0.05:
                direction = "DECREASING"
            else:
                direction = "STABLE"
        else:
            direction = "INSUFFICIENT_DATA"
            slope = 0.0

        daily_data = [{"date": str(d), "count": c} for d, c in zip(dates, counts)]
        return {
            "cause": cause,
            "daily_data": daily_data,
            "trend_direction": direction,
            "trend_strength": round(float(slope), 4),
            "total_occurrences": len(records)
        }


class PredictiveRCA:
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.scaler = RobustScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.classes_: List[str] = []
        self.feature_columns: List[str] = FIXED_FEATURE_COLUMNS_RCA.copy()

    def _build_feature_vector(self, features_dict: Dict[str, float]) -> np.ndarray:
        vec = [features_dict.get(col, 0.0) for col in self.feature_columns]
        vec = [0.0 if (np.isnan(v) or np.isinf(v)) else v for v in vec]
        return np.array(vec, dtype=np.float32)

    def _extract_training_data(
        self,
        db: Session,
        unit_name: Optional[str] = None,
        min_samples_per_class: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        query = db.query(RiskAssessment).filter(
            RiskAssessment.root_cause.isnot(None),
            RiskAssessment.root_cause != "Normal",
            RiskAssessment.root_cause != "None"
        )
        if unit_name:
            unit = db.query(Unit).filter(Unit.name == unit_name).first()
            if unit:
                query = query.filter(RiskAssessment.unit_id == unit.id)

        records = query.order_by(RiskAssessment.timestamp).all()
        if len(records) < 20:
            return np.array([]), np.array([])

        X_list = []
        y_list = []
        for ra in records:
            features = self._get_features_for_risk_assessment(db, ra)
            if features:
                X_list.append(self._build_feature_vector(features))
                y_list.append(ra.root_cause)

        if len(X_list) < 20:
            return np.array([]), np.array([])

        label_counts = Counter(y_list)
        valid_labels = {lbl for lbl, cnt in label_counts.items() if cnt >= min_samples_per_class}
        if len(valid_labels) < 2:
            return np.array([]), np.array([])

        filtered_X = []
        filtered_y = []
        for x, y in zip(X_list, y_list):
            if y in valid_labels:
                filtered_X.append(x)
                filtered_y.append(y)

        return np.array(filtered_X), np.array(filtered_y)

    def _get_features_for_risk_assessment(
        self,
        db: Session,
        risk_assessment: RiskAssessment
    ) -> Optional[Dict[str, float]]:
        """
        Extract complete feature dictionary for a given RiskAssessment
        using its sensor_reading_id and closest environment/maintenance/hse records.
        This method is deterministic and does not rely on timestamp matching.
        """
        sensor = risk_assessment.sensor_reading
        if not sensor:
            logger.warning(f"No sensor reading for risk assessment {risk_assessment.id}")
            return None

        unit = risk_assessment.unit
        if not unit:
            logger.warning(f"No unit for risk assessment {risk_assessment.id}")
            return None

        features = {
            "temp_in": sensor.temperature_in or 0.0,
            "temp_out": sensor.temperature_out or 0.0,
            "press_in": sensor.pressure_in or 0.0,
            "press_out": sensor.pressure_out or 0.0,
            "flow": sensor.flow_rate or 0.0,
            "level": sensor.level or 0.0,
            "vib": sensor.vibration or 0.0,
            "bearing": sensor.bearing_temp or 0.0,
            "rpm": sensor.motor_rpm or 0.0,
            "current": sensor.motor_current or 0.0,
            "torque": sensor.torque or 0.0,
            "seal": sensor.seal_pressure or 0.0,
            "valve1": sensor.valve_position_1 or 0.0,
            "valve2": sensor.valve_position_2 or 0.0,
            "gas": sensor.gas_concentration or 0.0,
            "conduct": sensor.conductivity or 0.0,
            "ph": sensor.ph or 0.0,
            "coil_temp": sensor.temp_ambient_coil or 0.0,
            "steam": sensor.pressure_steam or 0.0,
            "oil_level": sensor.oil_level or 0.0,
            "cooling_temp": sensor.cooling_water_temp or 0.0,
        }

        # Environment data (closest by timestamp, use sensor.timestamp)
        env = db.query(EnvironmentData).filter(
            EnvironmentData.sensor_reading_id == sensor.id
        ).first()
        if env:
            features.update({
                "ambient_temp": env.ambient_temp or 0.0,
                "humidity": env.humidity or 0.0,
                "wind": env.wind_speed or 0.0,
                "atm": env.atmospheric_pressure or 0.0,
                "rain": env.rainfall or 0.0,
                "solar": env.solar_radiation or 0.0,
            })
        else:
            for k in ["ambient_temp", "humidity", "wind", "atm", "rain", "solar"]:
                features[k] = 0.0

        # Maintenance log: latest before or equal to risk assessment timestamp
        maint = db.query(MaintenanceLog).filter(
            MaintenanceLog.unit_id == unit.id,
            MaintenanceLog.timestamp <= risk_assessment.timestamp
        ).order_by(desc(MaintenanceLog.timestamp)).first()
        if maint:
            features.update({
                "last_service_days": maint.last_service_days or 0,
                "comp_age": maint.component_age_days or 0,
                "repl_count": maint.replacement_count or 0,
                "mtbf": maint.mtbf or 0.0,
                "mttr": maint.mttr or 0.0,
                "repair_cost": maint.last_repair_cost or 0.0,
            })
        else:
            for k in ["last_service_days", "comp_age", "repl_count", "mtbf", "mttr", "repair_cost"]:
                features[k] = 0.0

        # HSE report: latest before or equal
        hse = db.query(HSEReport).filter(
            HSEReport.unit_id == unit.id,
            HSEReport.timestamp <= risk_assessment.timestamp
        ).order_by(desc(HSEReport.timestamp)).first()
        if hse:
            features["last_incident_days"] = hse.last_incident_days or 0
            features["safety_score"] = hse.safety_audit_score or 0.0
        else:
            features["last_incident_days"] = 0
            features["safety_score"] = 0.0

        features["risk_score"] = risk_assessment.risk_score
        features["timestamp"] = risk_assessment.timestamp.isoformat()


        return features

    def train(self, db: Session, unit_name: Optional[str] = None) -> bool:
        X, y = self._extract_training_data(db, unit_name, min_samples_per_class=5)
        if X.shape[0] < 20:
            logger.warning(f"Not enough training data for RCA: {X.shape[0]} samples")
            return False

        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_.tolist()

        X_scaled = self.scaler.fit_transform(X)

        self.models = {
            'xgb': xgb.XGBClassifier(
                n_estimators=150, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                use_label_encoder=False, eval_metric='mlogloss'
            ),
            'lgb': lgb.LGBMClassifier(
                n_estimators=150, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                verbose=-1
            )
        }
        for name, model in self.models.items():
            model.fit(X_scaled, y_encoded)

        self.is_trained = True
        logger.info(f"Predictive RCA trained on {X.shape[0]} samples, {len(self.classes_)} classes")
        return True

    def predict(self, features_dict: Dict[str, float]) -> PredictiveRCAResult:
        if not self.is_trained:
            return PredictiveRCAResult(
                predicted_cause="Unknown",
                confidence=0.0,
                top_causes=[("Model not trained", 0.0)]
            )

        vec = self._build_feature_vector(features_dict).reshape(1, -1)
        X_scaled = self.scaler.transform(vec)

        all_probs = []
        for model in self.models.values():
            probs = model.predict_proba(X_scaled)[0]
            all_probs.append(probs)
        avg_probs = np.mean(all_probs, axis=0)
        top_indices = np.argsort(avg_probs)[::-1][:3]
        top_causes = [(self.classes_[i], float(avg_probs[i])) for i in top_indices]

        predicted_idx = np.argmax(avg_probs)
        return PredictiveRCAResult(
            predicted_cause=self.classes_[predicted_idx],
            confidence=float(avg_probs[predicted_idx]),
            top_causes=top_causes
        )

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.models, os.path.join(path, 'rca_models.pkl'))
        joblib.dump(self.scaler, os.path.join(path, 'rca_scaler.pkl'))
        joblib.dump(self.label_encoder, os.path.join(path, 'rca_label_encoder.pkl'))
        with open(os.path.join(path, 'rca_classes.pkl'), 'wb') as f:
            pickle.dump(self.classes_, f)
        with open(os.path.join(path, 'rca_feature_columns.pkl'), 'wb') as f:
            pickle.dump(self.feature_columns, f)
        logger.info(f"Predictive RCA saved to {path}")

    @classmethod
    def load(cls, path: str) -> Optional['PredictiveRCA']:
        model_path = os.path.join(path, 'rca_models.pkl')
        if not os.path.exists(model_path):
            return None
        instance = cls()
        instance.models = joblib.load(model_path)
        instance.scaler = joblib.load(os.path.join(path, 'rca_scaler.pkl'))
        instance.label_encoder = joblib.load(os.path.join(path, 'rca_label_encoder.pkl'))
        with open(os.path.join(path, 'rca_classes.pkl'), 'rb') as f:
            instance.classes_ = pickle.load(f)
        feat_path = os.path.join(path, 'rca_feature_columns.pkl')
        if os.path.exists(feat_path):
            with open(feat_path, 'rb') as f:
                instance.feature_columns = pickle.load(f)
        instance.is_trained = True
        return instance


class RootCauseAnalyzerService:
    def __init__(self, base_model_dir: str = MODEL_DIR):
        self.base_model_dir = base_model_dir
        self.weighted_analyzer = WeightedCauseAnalyzer()
        self.trend_analyzer = CauseTrendAnalyzer()
        self._predictive_models: Dict[str, PredictiveRCA] = {}
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

    def _load_predictive_model(self, unit_name: str) -> bool:
        unit_path = self._get_unit_path(unit_name)
        model = PredictiveRCA.load(unit_path)
        if model:
            self._predictive_models[unit_name] = model
            return True
        return False

    def train_predictive(self, db: Session, unit_name: Optional[str] = None) -> bool:
        key = unit_name if unit_name else "global"
        lock = self._get_lock(key)
        with lock:
            model = PredictiveRCA()
            success = model.train(db, unit_name)
            if success:
                save_path = self._get_unit_path(key)
                model.save(save_path)
                self._predictive_models[key] = model
                logger.info(f"Predictive RCA trained for {key}")
                return True
            return False

    def get_complete_analysis(
        self,
        db: Session,
        unit_name: Optional[str] = None,
        days_back: int = 30
    ) -> CompleteRCAAnalysis:
        top_causes = self.weighted_analyzer.analyze(db, unit_name, days_back=days_back, limit=10)

        cause_trends = {}
        for cause_item in top_causes[:3]:
            trend = self.trend_analyzer.analyze_trend(db, cause_item['cause'], days=days_back)
            cause_trends[cause_item['cause']] = trend

        predictive_result = None
        if unit_name:
            key = unit_name
            if key not in self._predictive_models:
                self._load_predictive_model(key)
            model = self._predictive_models.get(key)
            if model and model.is_trained:
                from backend.db.crud import get_latest_features_for_unit
                features = get_latest_features_for_unit(db, unit_name)
                if features:
                    pred = model.predict(features)
                    predictive_result = {
                        "predicted_cause": pred.predicted_cause,
                        "confidence": pred.confidence,
                        "top_causes": pred.top_causes
                    }

        total_events = db.query(RiskAssessment).count()
        avg_risk = db.query(func.avg(RiskAssessment.risk_score)).scalar() or 0.0
        summary = {
            "total_risk_events": int(total_events),
            "average_risk_score": round(float(avg_risk), 2),
            "analysis_days": days_back
        }

        recommendations = []
        if top_causes:
            top = top_causes[0]
            recommendations.append(f"Primary risk driver: {top['cause']} (weight {top['weight']})")
        if predictive_result and predictive_result['confidence'] > 0.6:
            recommendations.append(f"ML predicts next cause: {predictive_result['predicted_cause']} (confidence {predictive_result['confidence']:.0%})")
        if any(t.get('trend_direction') == 'INCREASING' for t in cause_trends.values()):
            recommendations.append("Increasing trend in one or more causes - investigate")
        if not recommendations:
            recommendations.append("No critical trends detected - normal operation")

        return CompleteRCAAnalysis(
            timestamp=datetime.now(timezone.utc).isoformat(),
            unit_name=unit_name or "All Units",
            top_causes=top_causes,
            cause_trends=cause_trends,
            predictive_rca=predictive_result,
            recommendations=recommendations,
            summary_statistics=summary
        )


_rca_service = RootCauseAnalyzerService()


def get_top_root_causes_professional(
    db: Session,
    unit_name: Optional[str] = None,
    limit: int = 5
) -> List[Dict[str, Any]]:
    return _rca_service.weighted_analyzer.analyze(db, unit_name, limit=limit)


def get_cause_trend_professional(
    db: Session,
    cause: str,
    days: int = 30
) -> List[Dict[str, Any]]:
    trend = _rca_service.trend_analyzer.analyze_trend(db, cause, days)
    return trend.get('daily_data', [])


def predict_root_cause_professional(
    db: Session,
    unit_name: str
) -> Dict[str, Any]:
    key = unit_name
    if key not in _rca_service._predictive_models:
        if not _rca_service._load_predictive_model(key):
            success = _rca_service.train_predictive(db, unit_name)
            if not success:
                return {"error": "Predictive RCA model not available for this unit"}
    model = _rca_service._predictive_models.get(key)
    if not model or not model.is_trained:
        return {"error": "Predictive RCA model not trained"}
    from backend.db.crud import get_latest_features_for_unit
    features = get_latest_features_for_unit(db, unit_name)
    if not features:
        return {"error": "No feature data for unit"}
    result = model.predict(features)
    return {
        "predicted_cause": result.predicted_cause,
        "confidence": result.confidence,
        "top_causes": result.top_causes
    }


def get_complete_rca_analysis(
    db: Session,
    unit_name: Optional[str] = None
) -> Dict[str, Any]:
    report = _rca_service.get_complete_analysis(db, unit_name)
    return {
        "timestamp": report.timestamp,
        "unit_name": report.unit_name,
        "top_causes": report.top_causes,
        "cause_trends": report.cause_trends,
        "predictive_rca": report.predictive_rca,
        "recommendations": report.recommendations,
        "summary_statistics": report.summary_statistics
    }