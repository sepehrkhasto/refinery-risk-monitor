"""
Trend analysis service.
Includes Mann-Kendall, seasonality detection, change point detection, and Holt-Winters forecasting.
Forces all outputs to standard Python types (no NumPy).
Caches expensive bootstrap computations per (data_hash, confidence_level).
"""

import os, sys
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
from functools import lru_cache

from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sqlalchemy.orm import Session

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)
from backend.db.crud import get_unit_risk_history, get_all_features_for_unit


def _safe_value(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def _safe_dict(d):
    if isinstance(d, dict):
        return {k: _safe_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [_safe_dict(i) for i in d]
    else:
        return _safe_value(d)


class TrendDirection:
    STABLE = "STABLE"
    INCREASING = "INCREASING"
    DECREASING = "DECREASING"
    CYCLICAL = "CYCLICAL"
    UNKNOWN = "UNKNOWN"


class TrendSeverity:
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NONE = "NONE"


@dataclass
class TrendResult:
    variable_name: str
    direction: str
    slope: float
    p_value: float
    is_significant: bool
    confidence_interval: Tuple[float, float]
    trend_strength: str
    severity: str
    recommendation: str


@dataclass
class SeasonalityResult:
    has_seasonality: bool
    periods: List[int]
    strengths: List[float]
    decomposition_available: bool
    trend_component: Optional[np.ndarray] = None
    seasonal_component: Optional[np.ndarray] = None
    residual_component: Optional[np.ndarray] = None


@dataclass
class ChangePointResult:
    has_changes: bool
    change_points: List[int]
    timestamps: List[str]
    segments: List[Dict]
    recent_change: bool


@dataclass
class ForecastResult:
    forecast_values: List[float]
    confidence_lower: List[float]
    confidence_upper: List[float]
    steps: int
    method: str
    model_quality: Dict[str, float]


class MannKendallAnalyzer:
    @staticmethod
    @lru_cache(maxsize=256)
    def analyze_cached(data_tuple: Tuple[float, ...], confidence_level: float = 0.95) -> Dict:
        data = np.array(data_tuple)
        return MannKendallAnalyzer._analyze_impl(data, confidence_level)

    @staticmethod
    def _analyze_impl(data: np.ndarray, confidence_level: float = 0.95) -> Dict:
        n = len(data)
        if n < 4:
            return {
                'direction': TrendDirection.UNKNOWN, 'slope': 0, 'p_value': 1.0,
                'is_significant': False, 'confidence_interval': (0, 0), 'trend_strength': 'WEAK'
            }
        s = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                s += np.sign(data[j] - data[i])
        var_s = n * (n - 1) * (2 * n + 5) / 18
        unique, counts = np.unique(data, return_counts=True)
        for cnt in counts[counts > 1]:
            var_s -= cnt * (cnt - 1) * (2 * cnt + 5) / 18
        z = (s - 1) / np.sqrt(var_s) if s > 0 else (s + 1) / np.sqrt(var_s) if s < 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        is_sig = p_value < (1 - confidence_level)
        slopes = [(data[j] - data[i]) / (j - i) for i in range(n - 1) for j in range(i + 1, n) if data[j] != data[i]]
        slope = np.median(slopes) if slopes else 0
        boot_slopes = []
        for _ in range(1000):
            idx = np.random.choice(n, n, replace=True)
            samp = data[idx]
            bs = [(samp[j] - samp[i]) / (j - i) for i in range(len(samp) - 1) for j in range(i + 1, len(samp)) if samp[j] != samp[i]]
            if bs:
                boot_slopes.append(np.median(bs))
        if boot_slopes:
            ci_low = np.percentile(boot_slopes, (1 - confidence_level) * 50)
            ci_high = np.percentile(boot_slopes, (1 + confidence_level) * 50)
        else:
            ci_low, ci_high = slope, slope
        direction = TrendDirection.STABLE if not is_sig else (TrendDirection.INCREASING if slope > 0 else TrendDirection.DECREASING)
        abs_z = abs(z)
        if abs_z > 2.58:
            strength = "STRONG"
        elif abs_z > 1.96:
            strength = "MODERATE"
        else:
            strength = "WEAK"
        return {
            'direction': direction,
            'slope': float(slope),
            'p_value': float(p_value),
            'is_significant': bool(is_sig),
            'confidence_interval': (float(ci_low), float(ci_high)),
            'trend_strength': strength
        }

    @staticmethod
    def analyze(data: np.ndarray, confidence_level: float = 0.95) -> Dict:
        return MannKendallAnalyzer.analyze_cached(tuple(data.tolist()), confidence_level)


class SeasonalityDetector:
    def __init__(self, max_period=168, min_period=2):
        self.max_period = max_period
        self.min_period = min_period

    def detect(self, data: np.ndarray, freq_hint=None) -> SeasonalityResult:
        n = len(data)
        # Always returns a valid SeasonalityResult, never None
        if n < 10:
            return SeasonalityResult(False, [], [], False)
        autocorr = self._autocorrelation(data)
        periods, strengths = [], []
        for period in range(self.min_period, min(self.max_period, n // 2)):
            if period < len(autocorr) and autocorr[period] > 0.2:
                is_peak = True
                for nb in [-1, 1]:
                    p_idx = period + nb
                    if 0 <= p_idx < len(autocorr) and autocorr[p_idx] >= autocorr[period]:
                        is_peak = False
                        break
                if is_peak:
                    periods.append(period)
                    strengths.append(float(autocorr[period]))
        if periods:
            pairs = sorted(zip(periods, strengths), key=lambda x: x[1], reverse=True)
            periods = [p for p, _ in pairs[:3]]
            strengths = [s for _, s in pairs[:3]]
        decomposition_available = False
        trend_comp, seasonal_comp, resid_comp = None, None, None
        if periods and len(data) >= 2 * periods[0]:
            try:
                decomp = seasonal_decompose(data, model='additive', period=periods[0])
                trend_comp = decomp.trend.values
                seasonal_comp = decomp.seasonal.values
                resid_comp = decomp.resid.values
                decomposition_available = True
            except Exception:
                pass
        has_seasonality = len(periods) > 0 and max(strengths) > 0.25
        return SeasonalityResult(bool(has_seasonality), periods, strengths, decomposition_available,
                                 trend_comp, seasonal_comp, resid_comp)

    def _autocorrelation(self, data):
        n = len(data)
        max_lag = min(self.max_period, n // 2)
        mean = np.mean(data)
        var = np.var(data)
        if var == 0:
            return np.zeros(max_lag + 1)
        autocorr = np.zeros(max_lag + 1)
        for lag in range(max_lag + 1):
            if lag == 0:
                autocorr[lag] = 1
            else:
                cov = np.sum((data[:-lag] - mean) * (data[lag:] - mean))
                autocorr[lag] = cov / ((n - lag) * var)
        return autocorr


class ChangePointDetector:
    def detect(self, data, timestamps=None):
        n = len(data)
        if n < 10:
            return ChangePointResult(False, [], [], [], False)
        import ruptures as rpt
        signal = data.reshape(-1, 1)
        algo = rpt.Pelt(model='l2', min_size=2).fit(signal)
        change_pts = algo.predict(pen=10)
        change_pts = [cp for cp in change_pts if cp < n]
        change_times = [timestamps[cp] for cp in change_pts if timestamps and cp < len(timestamps)]
        segments = []
        start = 0
        for cp in change_pts:
            seg_data = data[start:cp]
            if len(seg_data) >= 3:
                slope = np.polyfit(range(len(seg_data)), seg_data, 1)[0]
                segments.append({'start': start, 'end': cp, 'mean': float(np.mean(seg_data)), 'slope': float(slope)})
            start = cp
        if start < n:
            seg_data = data[start:]
            if len(seg_data) >= 3:
                slope = np.polyfit(range(len(seg_data)), seg_data, 1)[0]
                segments.append({'start': start, 'end': n, 'mean': float(np.mean(seg_data)), 'slope': float(slope)})
        recent = False
        if change_pts and segments and len(segments) >= 2 and segments[-1]['end'] - segments[-1]['start'] <= 0.1 * n:
            recent = True
        return ChangePointResult(len(change_pts) > 0, change_pts, change_times, segments, recent)


class TrendForecaster:
    def forecast(self, data, steps=12, confidence_level=0.95):
        n = len(data)
        if n < 10:
            return ForecastResult([np.mean(data)] * steps if n else [0] * steps,
                                  [0] * steps, [0] * steps, steps, 'none', {})
        period = min(len(data) // 3, 24)
        try:
            model = ExponentialSmoothing(data, trend='add', seasonal='add', seasonal_periods=period)
            fitted = model.fit()
            pred = fitted.forecast(steps)
            resid_std = float(np.std(fitted.resid)) if fitted.resid is not None else float(np.std(data)) * 0.1
            z = stats.norm.ppf((1 + confidence_level) / 2)
            ci = z * resid_std
            return ForecastResult(
                [float(x) for x in pred],
                [float(x - ci) for x in pred],
                [float(x + ci) for x in pred],
                steps, 'holtwinters',
                {'aic': float(fitted.aic), 'bic': float(fitted.bic)}
            )
        except Exception:
            return ForecastResult([float(data[-1])] * steps,
                                  [float(data[-1] - 5)] * steps,
                                  [float(data[-1] + 5)] * steps,
                                  steps, 'fallback', {})


class TrendAnalysisService:
    def __init__(self):
        self.mk = MannKendallAnalyzer()
        self.season = SeasonalityDetector()
        self.cp = ChangePointDetector()
        self.forecaster = TrendForecaster()

    def analyze_risk_trend(self, risk_scores: np.ndarray) -> TrendResult:
        mk = self.mk.analyze(risk_scores)
        current = float(risk_scores[-1]) if len(risk_scores) else 0.0
        if mk['direction'] == TrendDirection.INCREASING and mk['slope'] > 0.5:
            severity = TrendSeverity.CRITICAL if current > 70 else TrendSeverity.HIGH if current > 50 else TrendSeverity.MEDIUM
        elif mk['direction'] == TrendDirection.INCREASING:
            severity = TrendSeverity.LOW
        else:
            severity = TrendSeverity.NONE
        rec = self._trend_recommendation(mk['direction'], mk['slope'], current, 'risk')
        return TrendResult('risk_score', mk['direction'], float(mk['slope']),
                           float(mk['p_value']), bool(mk['is_significant']),
                           mk['confidence_interval'], mk['trend_strength'], severity, rec)

    def analyze_sensor_trend(self, sensor_data: Dict[str, List[float]]) -> List[TrendResult]:
        results = []
        critical_sensors = ['temperature_out', 'vibration', 'pressure_out', 'flow_rate']
        for name, values in sensor_data.items():
            if len(values) < 5:
                continue
            mk = self.mk.analyze(np.array(values))
            is_crit = any(cs in name.lower() for cs in critical_sensors)
            if mk['direction'] == TrendDirection.INCREASING and mk['is_significant']:
                severity = TrendSeverity.HIGH if (is_crit and mk['slope'] > 0.1) else TrendSeverity.MEDIUM if is_crit else TrendSeverity.LOW
            else:
                severity = TrendSeverity.NONE
            rec = self._trend_recommendation(mk['direction'], mk['slope'], values[-1], name)
            results.append(TrendResult(name, mk['direction'], float(mk['slope']),
                                       float(mk['p_value']), bool(mk['is_significant']),
                                       mk['confidence_interval'], mk['trend_strength'], severity, rec))
        severity_order = {TrendSeverity.CRITICAL: 0, TrendSeverity.HIGH: 1, TrendSeverity.MEDIUM: 2,
                          TrendSeverity.LOW: 3, TrendSeverity.NONE: 4}
        results.sort(key=lambda x: (severity_order[x.severity], -abs(x.slope)))
        return results

    def _trend_recommendation(self, direction, slope, current, var):
        name = var.replace('_', ' ').title()
        if direction == TrendDirection.INCREASING:
            if 'risk' in var.lower():
                if slope > 1:
                    return f"{name} rising rapidly ({slope:.2f}/step). Immediate investigation."
                elif slope > 0.3:
                    return f"{name} increasing. Schedule review."
                else:
                    return f"{name} slowly increasing. Monitor."
            else:
                if slope > 0.5:
                    return f"{name} increasing significantly. Check for issues."
                else:
                    return f"{name} upward trend. Monitor."
        elif direction == TrendDirection.DECREASING:
            return f"{name} decreasing. Positive trend, continue."
        else:
            return f"{name} stable. Normal operation."

    def generate_text_viz(self, data, forecast=None, width=60):
        if len(data) == 0:
            return "No data"
        lines = ["-" * width, "TREND VISUALIZATION", "-" * width]
        data_min, data_max = np.min(data), np.max(data)
        rng = data_max - data_min if data_max > data_min else 1
        chart_height = 10
        for level in range(chart_height - 1, -1, -1):
            thresh = data_min + rng * level / (chart_height - 1)
            line = ''.join('|' if v >= thresh else ' ' for v in data)
            lines.append(line)
        lines.append("-" * width)
        if forecast:
            lines.append(f"Forecast method: {forecast.method}, next: {forecast.forecast_values[:3]}")
        return "\n".join(lines)

    def generate_alerts(self, risk_trend, change_points, current_risk):
        alerts = []
        if risk_trend.direction == TrendDirection.INCREASING and risk_trend.is_significant:
            if risk_trend.severity in [TrendSeverity.CRITICAL, TrendSeverity.HIGH]:
                alerts.append({'level': 'CRITICAL', 'message': f"Risk increasing (slope {risk_trend.slope:.3f})"})
        if change_points.recent_change:
            alerts.append({'level': 'INFO', 'message': "Recent change point detected."})
        return alerts

    def generate_recommendations(self, risk_trend, sensor_trends, seasonality, change_points):
        recs = []
        if risk_trend.direction == TrendDirection.INCREASING:
            if risk_trend.severity in [TrendSeverity.CRITICAL, TrendSeverity.HIGH]:
                recs.append(f"URGENT: Risk trending {risk_trend.direction} with slope {risk_trend.slope:.3f}")
        crit_sensors = [t for t in sensor_trends if t.severity in [TrendSeverity.HIGH, TrendSeverity.MEDIUM]]
        if crit_sensors:
            recs.append(f"Monitor sensors: {', '.join(t.variable_name for t in crit_sensors[:3])}")
        if seasonality.has_seasonality:
            recs.append(f"Seasonality detected (period {seasonality.periods[0]}). Adjust baselines.")
        if change_points.has_changes:
            recs.append("Trend behavior changed; review operational events.")
        return recs[:5]


_trend_service = TrendAnalysisService()


def analyze_unit_trend_professional(
    db: Session,
    unit_name: str,
    limit: int = 60,
    include_sensors: bool = True,
    forecast_steps: int = 12
) -> Dict:
    records = get_unit_risk_history(db, unit_name, limit=limit)
    if len(records) < 5:
        return {"error": "Insufficient data", "success": False}
    records = sorted(records, key=lambda x: x.timestamp)
    risk_scores = np.array([r.risk_score for r in records])
    timestamps = [r.timestamp.isoformat() for r in records]

    mk = MannKendallAnalyzer.analyze(risk_scores)
    season = _trend_service.season.detect(risk_scores)
    cp = _trend_service.cp.detect(risk_scores, timestamps)
    forecast = _trend_service.forecaster.forecast(risk_scores, steps=min(forecast_steps, 24))

    result = {
        "success": True,
        "unit_name": unit_name,
        "direction": mk['direction'],
        "slope": round(float(mk['slope']), 4),
        "p_value": round(float(mk['p_value']), 4),
        "significant": bool(mk['is_significant']),
        "seasonal_period": int(season.periods[0]) if season.periods else 0,
        "is_seasonal": bool(season.has_seasonality),
        "risk_trend_detail": {
            "direction": mk['direction'],
            "slope": float(mk['slope']),
            "is_significant": bool(mk['is_significant']),
            "trend_strength": mk['trend_strength'],
            "severity": _trend_service.analyze_risk_trend(risk_scores).severity,
            "confidence_interval": (float(mk['confidence_interval'][0]), float(mk['confidence_interval'][1])),
            "recommendation": _trend_service.analyze_risk_trend(risk_scores).recommendation
        },
        "seasonality_detail": {
            "has_seasonality": bool(season.has_seasonality),
            "periods": [int(p) for p in season.periods],
            "strengths": [float(s) for s in season.strengths]
        },
        "change_points": {
            "has_changes": bool(cp.has_changes),
            "change_points": [int(x) for x in cp.change_points],
            "timestamps": cp.timestamps,
            "recent_change": bool(cp.recent_change),
            "segments_count": len(cp.segments)
        },
        "forecast": {
            "values": [round(float(v), 2) for v in forecast.forecast_values[:12]],
            "method": forecast.method,
            "confidence_lower": [round(float(v), 2) for v in forecast.confidence_lower[:12]],
            "confidence_upper": [round(float(v), 2) for v in forecast.confidence_upper[:12]]
        },
        "recommendations": _trend_service.generate_recommendations(
            _trend_service.analyze_risk_trend(risk_scores), [], season, cp
        ),
        "text_visualization": _trend_service.generate_text_viz(risk_scores, forecast)
    }
    if include_sensors:
        features_list = get_all_features_for_unit(db, unit_name, limit=limit)
        if features_list and len(features_list) >= 5:
            sensor_data = {}
            for sensor in ['temperature_out', 'vibration', 'pressure_out', 'flow_rate']:
                vals = [f.get(sensor, 0) for f in features_list if f.get(sensor) is not None]
                if len(vals) >= 5:
                    sensor_data[sensor] = vals
            if sensor_data:
                sensor_trends = _trend_service.analyze_sensor_trend(sensor_data)
                result["sensor_trends"] = [
                    {"sensor": t.variable_name, "direction": t.direction, "slope": float(t.slope),
                     "severity": t.severity, "recommendation": t.recommendation}
                    for t in sensor_trends[:5]
                ]
    return _safe_dict(result)


def analyze_unit_trend(db: Session, unit_name: str, limit: int = 30):
    result = analyze_unit_trend_professional(db, unit_name, limit=limit, include_sensors=False, forecast_steps=6)
    if result.get('error'):
        return None
    return {
        "direction": result['direction'],
        "slope": result['slope'],
        "p_value": result['p_value'],
        "significant": result['significant'],
        "seasonal_period": result['seasonal_period'],
        "is_seasonal": result['is_seasonal']
    }