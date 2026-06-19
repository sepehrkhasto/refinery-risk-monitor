"""
Professional refinery simulator – Enterprise Edition.
Includes First-Principles models, causal soft sensor, GAN for fault data generation,
and risk assessment engine. Modular and extensible design.
"""

import math, random, numpy as np, pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Thermodynamic properties
# ============================================================================
@dataclass
class ThermodynamicProperties:
    cp: float = 2.2      # kJ/(kg·K)
    rho: float = 850      # kg/m³
    mu: float = 0.8       # cP
    k: float = 0.13       # W/(m·K)


# ============================================================================
# First-Principles Model
# ============================================================================
class FirstPrinciplesModel:
    """Real physical models: energy, momentum, mass balance."""

    def __init__(self, unit_specs: Dict[str, Any]):
        self.specs = unit_specs
        self.thermo = ThermodynamicProperties()
        self.state = {}

    def energy_balance(self, T_in: float, flow_rate: float, heat_input: float, ambient_temp: float) -> float:
        m_dot = flow_rate * self.thermo.rho / 3600
        Q_process = m_dot * self.thermo.cp * T_in
        UA = self.specs.get('ua_value', 50)
        Q_loss = UA * (T_in - ambient_temp)
        Q_input = heat_input * 1000
        T_out = T_in + (Q_input - Q_loss) / (m_dot * self.thermo.cp)
        return max(T_out, T_in - 50)

    def momentum_balance(self, flow_rate, pressure_in, pipe_diameter=0.3, pipe_length=50):
        area = math.pi * (pipe_diameter / 2) ** 2
        velocity = flow_rate / (3600 * area)
        Re = (self.thermo.rho * velocity * pipe_diameter) / (self.thermo.mu * 1e-3)
        if Re < 2000:
            f = 64 / Re
        else:
            f = 0.0055 * (1 + (20000 * 1e-6 / pipe_diameter + 1e6 / Re) ** (1 / 3))
        delta_P = f * (pipe_length / pipe_diameter) * (self.thermo.rho * velocity ** 2 / 2)
        return pressure_in - delta_P / 100000

    def mass_balance(self, flow_in, flow_out, tank_area=10, rho=850):
        mass_in = flow_in * rho / 3600
        mass_out = flow_out * rho / 3600
        dm_dt = (mass_in - mass_out) * 3
        dh = dm_dt / (tank_area * rho) * 100
        return dh


# ============================================================================
# Causal Soft Sensor (ML)
# ============================================================================
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from statsmodels.tsa.stattools import grangercausalitytests
import shap


class CausalSoftSensor:
    """Advanced soft sensor with causal feature selection and ensemble."""

    def __init__(self, max_lags=5, test_size=0.2):
        self.max_lags = max_lags
        self.models = {
            'xgb': xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42),
            'lgb': lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, verbose=-1),
            'rf': RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42)
        }
        self.scalers = {}
        self.causal_features = {}

    def granger_causality_selection(self, data, target_col, significance_level=0.05):
        causal = []
        for col in data.columns:
            if col == target_col:
                continue
            test_data = data[[col, target_col]].dropna()
            if len(test_data) < self.max_lags + 10:
                continue
            try:
                result = grangercausalitytests(test_data, maxlag=self.max_lags, verbose=False)
                min_p = min(result[lag][0]['ssr_ftest'][1] for lag in range(1, self.max_lags + 1))
                if min_p < significance_level:
                    causal.append(col)
            except:
                continue
        return causal

    def train_soft_sensor(self, X_train, y_train, feature_names, target_name):
        train_df = X_train.copy()
        train_df[target_name] = y_train
        causal_feats = self.granger_causality_selection(train_df, target_name)
        if not causal_feats:
            causal_feats = feature_names
        self.causal_features[target_name] = causal_feats
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train[causal_feats])
        self.scalers[target_name] = scaler
        for name, model in self.models.items():
            model.fit(X_scaled, y_train)

    def predict(self, X: pd.DataFrame, target_name: str):
        causal_feats = self.causal_features.get(target_name, X.columns.tolist())
        X_scaled = self.scalers[target_name].transform(X[causal_feats])
        preds = np.zeros(len(X))
        for name, model in self.models.items():
            preds += model.predict(X_scaled)
        return preds / len(self.models), None


# ============================================================================
# Time-Series GAN for fault data
# ============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim, seq_length):
        super().__init__()
        self.seq_length = seq_length
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, output_dim * seq_length), nn.Tanh()
        )

    def forward(self, z):
        out = self.model(z)
        return out.view(-1, self.seq_length, -1)


class Discriminator(nn.Module):
    def __init__(self, input_dim, seq_length):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim * seq_length, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)


class TimeSeriesGAN:
    def __init__(self, input_dim, seq_length=24, latent_dim=100):
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim, input_dim, seq_length)
        self.discriminator = Discriminator(input_dim, seq_length)
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002)
        self.criterion = nn.BCELoss()

    def train(self, real_data, epochs=200, batch_size=32):
        real_tensor = torch.FloatTensor(real_data)
        dataset = TensorDataset(real_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            for batch in dataloader:
                real_batch = batch[0]
                bs = real_batch.size(0)
                real_labels = torch.ones(bs, 1)
                fake_labels = torch.zeros(bs, 1)
                self.d_optimizer.zero_grad()
                real_out = self.discriminator(real_batch)
                d_real_loss = self.criterion(real_out, real_labels)
                z = torch.randn(bs, self.latent_dim)
                fake_data = self.generator(z)
                fake_out = self.discriminator(fake_data.detach())
                d_fake_loss = self.criterion(fake_out, fake_labels)
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                self.d_optimizer.step()
                self.g_optimizer.zero_grad()
                z = torch.randn(bs, self.latent_dim)
                fake_data = self.generator(z)
                fake_out = self.discriminator(fake_data)
                g_loss = self.criterion(fake_out, real_labels)
                g_loss.backward()
                self.g_optimizer.step()

    def generate(self, n_samples):
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim)
            return self.generator(z).numpy()


# ============================================================================
# Risk Assessment Engine
# ============================================================================
class RiskAssessmentEngine:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)

    def mahalanobis_distance(self, x, mean, cov_inv):
        delta = x - mean
        return np.sqrt(np.dot(np.dot(delta, cov_inv), delta))

    def compute_risk_score(self, current_readings, historical_df, degradation_metrics):
        components = {}
        for sensor, value in current_readings.items():
            if sensor in historical_df.columns and historical_df[sensor].std() > 0:
                z = abs(value - historical_df[sensor].mean()) / historical_df[sensor].std()
                components[f'{sensor}_deviation'] = min(25, z * 5)
        components['degradation'] = min(30, degradation_metrics.get('degradation_rate', 0) * 30)
        components['failure_risk'] = degradation_metrics.get('failure_probability', 0) * 40
        total = sum(components.values())
        total = min(100, max(0, total))
        status = "CRITICAL" if total > 70 else "WARNING" if total > 45 else "ATTENTION" if total > 20 else "NORMAL"
        return {'total_risk_score': total, 'status': status, 'components': components}

    def predict_failure_probability(self, sensor_data, degradation_rate):
        shape, scale = 2.5, 100
        effective_age = degradation_rate * scale
        hazard = (shape / scale) * (effective_age / scale) ** (shape - 1)
        return min(0.99, max(0, 1 - np.exp(-hazard * 10)))


# ============================================================================
# Professional Refinery Simulator (main class)
# ============================================================================
class ProfessionalRefinerySimulator:
    """Integrated simulator with all capabilities."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.first_principles = FirstPrinciplesModel(config)
        self.soft_sensor = CausalSoftSensor()
        self.risk_engine = RiskAssessmentEngine()
        self.gan_model = TimeSeriesGAN(input_dim=10, seq_length=24)
        self.historical_df = pd.DataFrame()

    def generate_realistic_sensor_data(self, unit_name, operating_hours, ambient_temp):
        flow_rate = 50 + 10 * math.sin(operating_hours / 24 * 2 * math.pi)
        temp_in = 150 + 5 * math.sin(operating_hours / 12 * 2 * math.pi)
        temp_out = self.first_principles.energy_balance(temp_in, flow_rate, self.config.get('heat_input', 1000), ambient_temp)
        press_in = 3.0 + 0.5 * math.sin(operating_hours / 8 * math.pi)
        press_out = self.first_principles.momentum_balance(flow_rate, press_in)
        level = 50 + self.first_principles.mass_balance(flow_rate, flow_rate - 5)
        degradation = min(1.0, operating_hours / (self.config.get('mtbf', 8000) * 24))
        return {
            'temperature_in': temp_in + random.gauss(0, 0.5) + degradation * 5,
            'temperature_out': temp_out + random.gauss(0, 0.6) + degradation * 3,
            'pressure_in': press_in + random.gauss(0, 0.02),
            'pressure_out': press_out + random.gauss(0, 0.02),
            'flow_rate': flow_rate + random.gauss(0, 0.3) - degradation * 2,
            'level': level + random.gauss(0, 0.5),
            'vibration': 0.5 + abs(flow_rate - 50) * 0.01 + degradation * 0.05 + random.gauss(0, 0.05),
            'motor_current': 10 + degradation * 2 + random.gauss(0, 0.2),
            'seal_pressure': 2.0 - degradation * 0.1 + random.gauss(0, 0.03),
            'gas_concentration': 50 + degradation * 10 + random.gauss(0, 2)
        }

    def get_reading(self, unit_id, db_session=None):
        now = datetime.now(timezone.utc)
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        operating_hours = ((now - start).total_seconds() / 3600) % (24 * 365)
        ambient_temp = 25 + 8 * math.sin(2 * math.pi * (operating_hours % 24) / 24)
        raw = self.generate_realistic_sensor_data(self.config.get('unit_name', 'Distillation'), operating_hours, ambient_temp)
        degradation_rate = min(1.0, operating_hours / 8000)
        failure_prob = self.risk_engine.predict_failure_probability(list(raw.values()), degradation_rate)
        risk = self.risk_engine.compute_risk_score(
            raw,
            self.historical_df if not self.historical_df.empty else pd.DataFrame(),
            {'degradation_rate': degradation_rate, 'failure_probability': failure_prob}
        )
        return {
            **raw,
            'degradation_rate': degradation_rate,
            'failure_probability': failure_prob,
            'risk_score': risk['total_risk_score'],
            'risk_status': risk['status'],
            'operational_mode': self._determine_mode(risk['total_risk_score']),
            'maintenance_priority': 'CRITICAL' if risk['total_risk_score'] > 70 else 'HIGH' if risk['total_risk_score'] > 45 else 'MEDIUM'
        }

    def _determine_mode(self, score):
        if score > 75:
            return "EMERGENCY_SHUTDOWN"
        if score > 55:
            return "MAINTENANCE_REQUIRED"
        if score > 35:
            return "MONITORING"
        return "NORMAL_OPERATION"