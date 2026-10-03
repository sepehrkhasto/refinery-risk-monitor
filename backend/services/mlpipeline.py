"""
ML pipeline for training stacking ensemble risk prediction model.
Uses fixed 39 feature columns, time series cross-validation, and saves models
in a versioned registry. Compatible with quantile, anomaly, and RCA services.
"""

import os
import sys
import json
import warnings
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import joblib
from sqlalchemy.orm import Session

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)
from backend.db.database import SessionLocal
from backend.db.crud import get_aligned_features_and_risk
from backend.db.db_models import Unit
from backend.logger import logger

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
REGISTRY_DIR = os.path.join(MODEL_DIR, 'ensemble_registry')

# Fixed feature columns (must match predictor_ml)
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
# Feature Builder
# ============================================================================
class FeatureBuilder:
    """Build feature matrix from database using fixed columns."""

    @staticmethod
    def build_features(
            db: Session,
            unit_name: Optional[str] = None,
            limit: int = 5000
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        if unit_name:
            features_list, risk_scores = get_aligned_features_and_risk(db, unit_name, limit=limit)
        else:
            units = db.query(Unit).all()
            all_features = []
            all_risks = []
            for u in units:
                f, r = get_aligned_features_and_risk(db, u.name, limit=max(100, limit // len(units) + 10))
                all_features.extend(f)
                all_risks.extend(r)
            paired = sorted(zip(all_features, all_risks), key=lambda x: x[0].get('timestamp', datetime.min))
            features_list = [p[0] for p in paired]
            risk_scores = [p[1] for p in paired]

        if len(features_list) < 20 or len(risk_scores) < 20:
            raise ValueError(f"Insufficient data: {len(features_list)} features, {len(risk_scores)} risks")

        min_len = min(len(features_list), len(risk_scores))
        X = []
        y = []
        for i in range(min_len):
            vec = [features_list[i].get(col, 0.0) for col in FIXED_FEATURE_COLUMNS]
            vec = [0.0 if (v is None or np.isnan(v) or np.isinf(v)) else float(v) for v in vec]
            X.append(vec)
            y.append(float(risk_scores[i]))

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        return X, y, FIXED_FEATURE_COLUMNS


# ============================================================================
# Stacking Ensemble Model
# ============================================================================
class StackingEnsemble:
    """
    Stacking ensemble with base models (XGBoost, LightGBM, CatBoost, RandomForest, GBR)
    and a Ridge meta-learner. Trained with time series cross-validation.
    """
    
    def __init__(self, use_cv: bool = True, n_folds: int = 5):
        self.use_cv = use_cv
        self.n_folds = n_folds
        self.base_models: Dict[str, Any] = {}
        self.meta_model = None
        self.scaler = RobustScaler()
        self.is_fitted = False
        
    def _init_base_models(self) -> Dict[str, Any]:
        return {
            'xgb': xgb.XGBRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42
            ),
            'lgb': lgb.LGBMRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
            ),
            'cb': cb.CatBoostRegressor(
                iterations=200, depth=6, learning_rate=0.05,
                subsample=0.8, random_seed=42, verbose=False
            ),
            'rf': RandomForestRegressor(
                n_estimators=150, max_depth=8, random_state=42, n_jobs=-1
            ),
            'gbr': GradientBoostingRegressor(
                n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42
            )
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'StackingEnsemble':
        """Fit ensemble using time series cross-validation."""
        if X.shape[0] < 30:
            raise ValueError(f"Not enough samples for ensemble: {X.shape[0]}")
        
        X_scaled = self.scaler.fit_transform(X)
        
        self.base_models = self._init_base_models()
        
        if self.use_cv and X.shape[0] > self.n_folds * 10:
            tscv = TimeSeriesSplit(n_splits=self.n_folds)
            meta_features = np.zeros((X.shape[0], len(self.base_models)))
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train = y[train_idx]
                
                for i, (name, model) in enumerate(self.base_models.items()):
                    model_clone = type(model)(**model.get_params())
                    model_clone.fit(X_train, y_train)
                    meta_features[val_idx, i] = model_clone.predict(X_val)
            
            self.meta_model = Ridge(alpha=0.5)
            self.meta_model.fit(meta_features, y)
            
            for name, model in self.base_models.items():
                model.fit(X_scaled, y)
        else:
            # Without CV: base models are trained directly and averaged
            for name, model in self.base_models.items():
                model.fit(X_scaled, y)
            self.meta_model = None
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict risk scores."""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted")
        
        X_scaled = self.scaler.transform(X)
        
        base_preds = []
        for model in self.base_models.values():
            base_preds.append(model.predict(X_scaled))
        base_preds = np.column_stack(base_preds)
        
        if self.meta_model is not None:
            predictions = self.meta_model.predict(base_preds)
        else:
            predictions = np.mean(base_preds, axis=1)
        
        return np.clip(predictions, 0, 100)
    
    def save(self, path: str, metadata: Dict[str, Any]) -> None:
        """Save ensemble model and metadata."""
        os.makedirs(path, exist_ok=True)
        joblib.dump(self, os.path.join(path, 'ensemble_model.pkl'))
        joblib.dump(self.scaler, os.path.join(path, 'scaler.pkl'))
        with open(os.path.join(path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"StackingEnsemble saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> Optional['StackingEnsemble']:
        """Load ensemble model from disk."""
        model_path = os.path.join(path, 'ensemble_model.pkl')
        if os.path.exists(model_path):
            try:
                return joblib.load(model_path)
            except Exception as e:
                logger.error(f"Failed to load ensemble: {e}")
        return None


# ============================================================================
# Main Training Pipeline
# ============================================================================
def train_and_save_professional(db: Session, unit_name: Optional[str] = None) -> bool:
    """
    Main training pipeline. If unit_name provided, trains per-unit ensemble.
    Otherwise, trains a global ensemble on all units.
    """
    try:
        logger.info(f"Starting ensemble training for {unit_name or 'all units'}")
        
        X, y, feature_names = FeatureBuilder.build_features(db, unit_name=unit_name, limit=3000)
        if X.shape[0] < 50:
            logger.warning(f"Not enough data: {X.shape[0]} samples")
            return False
        
        ensemble = StackingEnsemble(use_cv=True, n_folds=5)
        ensemble.fit(X, y)
        
        predictions = ensemble.predict(X)
        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        r2 = r2_score(y, predictions)
        logger.info(f"Ensemble performance: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if unit_name:
            safe_name = unit_name.replace(' ', '_')
            save_dir = os.path.join(REGISTRY_DIR, safe_name, f'v{timestamp}')
        else:
            save_dir = os.path.join(REGISTRY_DIR, 'global', f'v{timestamp}')
        
        metadata = {
            "model_type": "stacking_ensemble",
            "unit_name": unit_name or "global",
            "training_samples": int(X.shape[0]),
            "feature_count": len(feature_names),
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        ensemble.save(save_dir, metadata)
        
        latest_dir = os.path.dirname(save_dir)
        ensemble.save(latest_dir, metadata)
        joblib.dump(feature_names, os.path.join(latest_dir, 'feature_names.pkl'))
        
        logger.info(f"Ensemble training completed and saved to {save_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        return False


# ============================================================================
# Load Latest Ensemble
# ============================================================================
def load_latest_ensemble(unit_name: Optional[str] = None) -> Optional[StackingEnsemble]:
    """Load the latest trained ensemble model for a unit or global."""
    if unit_name:
        model_dir = os.path.join(REGISTRY_DIR, unit_name.replace(' ', '_'))
    else:
        model_dir = os.path.join(REGISTRY_DIR, 'global')
    
    model_path = os.path.join(model_dir, 'ensemble_model.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


# ============================================================================
# ============================================================================
if __name__ == "__main__":
    db = SessionLocal()
    try:
        train_and_save_professional(db)
        units = db.query(Unit).all()
        for unit in units:
            train_and_save_professional(db, unit.name)
    finally:
        db.close()