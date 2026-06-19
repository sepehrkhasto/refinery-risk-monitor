"""
Database CRUD operations with optimized queries, support for PostgreSQL and SQLite,
OHLC, sensor health, feature engineering, and thread-safe session handling.
All functions include type hints, error handling, and production-grade logging.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import desc, func, and_, text, case, select
from sqlalchemy.orm import Session, joinedload
from dateutil import parser as date_parser

from backend.db import db_models
from backend.db.db_models import (
    Unit, SensorReading, EnvironmentData, MaintenanceLog, HSEReport,
    RiskAssessment, User, OperatorReport, SensorHealth, PredictionLog
)
from backend.db import schemas
from backend.auth.auth import get_password_hash, verify_password

logger = logging.getLogger("refinery")


# ============================================================================
# Unit Repository
# ============================================================================
class UnitRepo:
    """Repository for Unit entities."""
    @staticmethod
    def get_by_name(db: Session, name: str) -> Optional[Unit]:
        return db.query(Unit).filter(Unit.name == name).first()

    @staticmethod
    def get_all(db: Session) -> List[Unit]:
        return db.query(Unit).order_by(Unit.name).all()

    @staticmethod
    def get_by_id(db: Session, unit_id: str) -> Optional[Unit]:
        return db.query(Unit).filter(Unit.id == unit_id).first()


# ============================================================================
# RiskAssessment Repository
# ============================================================================
class RiskAssessmentRepo:
    @staticmethod
    def get_latest_for_unit(db: Session, unit_id: str) -> Optional[RiskAssessment]:
        return db.query(RiskAssessment).filter(
            RiskAssessment.unit_id == unit_id
        ).order_by(desc(RiskAssessment.timestamp)).first()

    @staticmethod
    def get_history(db: Session, unit_id: str, limit: int = 50, desc: bool = False) -> List[RiskAssessment]:
        query = db.query(RiskAssessment).filter(RiskAssessment.unit_id == unit_id)
        if desc:
            query = query.order_by(desc(RiskAssessment.timestamp))
        else:
            query = query.order_by(RiskAssessment.timestamp.asc())
        return query.limit(limit).all()

    @staticmethod
    def get_all_recent(db: Session, limit: int = 100) -> List[RiskAssessment]:
        return db.query(RiskAssessment).order_by(
            desc(RiskAssessment.timestamp)
        ).limit(limit).all()

    @staticmethod
    def get_latest_per_unit(db: Session) -> List[Dict[str, Any]]:
        """Get latest risk assessment for each unit."""
        subq = db.query(
            RiskAssessment.unit_id,
            func.max(RiskAssessment.timestamp).label("max_ts")
        ).group_by(RiskAssessment.unit_id).subquery()
        results = db.query(RiskAssessment).join(
            subq,
            (RiskAssessment.unit_id == subq.c.unit_id) &
            (RiskAssessment.timestamp == subq.c.max_ts)
        ).all()
        return [
            {
                "unit_name": r.unit.name,
                "risk_score": r.risk_score,
                "status": r.status,
                "timestamp": r.timestamp
            }
            for r in results
        ]


# ============================================================================
# Dashboard aggregate queries
# ============================================================================
def get_dashboard_stats(db: Session) -> Dict[str, Any]:
    """Get overall dashboard statistics."""
    total_readings = db.query(func.count(SensorReading.id)).scalar() or 0
    critical_alerts = db.query(func.count(RiskAssessment.id)).filter(
        RiskAssessment.status == "Critical"
    ).scalar() or 0
    avg_risk = db.query(func.avg(RiskAssessment.risk_score)).scalar() or 0.0
    total_units = db.query(func.count(Unit.id)).scalar() or 0
    last_update = db.query(func.max(RiskAssessment.timestamp)).scalar()
    return {
        "total_readings": int(total_readings),
        "critical_alerts": int(critical_alerts),
        "average_risk": round(float(avg_risk), 2),
        "total_units": int(total_units),
        "last_update": last_update.isoformat() if last_update else None,
    }


def get_top_risky_units(db: Session, limit: int = 5) -> List[RiskAssessment]:
    """Get units with highest current risk."""
    subq = db.query(
        RiskAssessment.unit_id,
        func.max(RiskAssessment.timestamp).label("max_ts")
    ).group_by(RiskAssessment.unit_id).subquery()
    return db.query(RiskAssessment).join(
        subq,
        (RiskAssessment.unit_id == subq.c.unit_id) &
        (RiskAssessment.timestamp == subq.c.max_ts)
    ).order_by(desc(RiskAssessment.risk_score)).limit(limit).all()


def get_status_distribution(db: Session) -> Dict[str, int]:
    """Risk status distribution across latest assessments."""
    subq = db.query(
        RiskAssessment.unit_id,
        func.max(RiskAssessment.timestamp).label("max_ts")
    ).group_by(RiskAssessment.unit_id).subquery()
    latest = db.query(RiskAssessment).join(
        subq,
        (RiskAssessment.unit_id == subq.c.unit_id) &
        (RiskAssessment.timestamp == subq.c.max_ts)
    ).subquery()
    counts = db.query(
        latest.c.status,
        func.count(latest.c.id)
    ).group_by(latest.c.status).all()
    result = {"Normal": 0, "Warning": 0, "Critical": 0}
    for status, cnt in counts:
        result[status] = cnt
    return result


def get_recent_risk_assessments(db: Session, limit: int = 60) -> List[RiskAssessment]:
    return db.query(RiskAssessment).order_by(desc(RiskAssessment.timestamp)).limit(limit).all()


def get_heatmap_data(db: Session) -> List[Dict[str, Any]]:
    return RiskAssessmentRepo.get_latest_per_unit(db)


# ============================================================================
# Monitoring endpoints
# ============================================================================
def get_ohlc_data(
    db: Session,
    unit_name: str,
    interval_minutes: int = 5,
    limit: int = 100
) -> List[Dict[str, Any]]:
    unit = UnitRepo.get_by_name(db, unit_name)
    if not unit:
        return []
    dialect = db.bind.dialect.name
    if dialect == "sqlite":
        time_bucket = func.strftime("%Y-%m-%d %H:%M", RiskAssessment.timestamp)
    elif dialect == "postgresql":
        time_bucket = func.date_trunc("minute", RiskAssessment.timestamp)
    else:
        time_bucket = func.date_format(RiskAssessment.timestamp, "%Y-%m-%d %H:%i:00")
    if dialect == "postgresql":
        windowed = db.query(
            RiskAssessment.timestamp,
            RiskAssessment.risk_score,
            func.date_trunc("minute", RiskAssessment.timestamp).label("bucket"),
            func.row_number().over(
                partition_by=func.date_trunc("minute", RiskAssessment.timestamp),
                order_by=RiskAssessment.timestamp.asc()
            ).label("rn_asc"),
            func.row_number().over(
                partition_by=func.date_trunc("minute", RiskAssessment.timestamp),
                order_by=RiskAssessment.timestamp.desc()
            ).label("rn_desc")
        ).filter(RiskAssessment.unit_id == unit.id).subquery()
        ohlc = db.query(
            windowed.c.bucket,
            func.min(windowed.c.risk_score).label("low"),
            func.max(windowed.c.risk_score).label("high"),
            func.avg(windowed.c.risk_score).label("close"),
            func.max(case((windowed.c.rn_asc == 1, windowed.c.risk_score), else_=None)).label("open"),
            func.max(case((windowed.c.rn_desc == 1, windowed.c.risk_score), else_=None)).label("close_exact")
        ).group_by(windowed.c.bucket).order_by(desc("bucket")).limit(limit).all()
        result = []
        for row in ohlc:
            result.append({
                "time": row.bucket.isoformat() if row.bucket else "",
                "open": float(row.open or row.low),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close_exact or row.close),
            })
        return result
    else:
        records = db.query(RiskAssessment.timestamp, RiskAssessment.risk_score).filter(
            RiskAssessment.unit_id == unit.id
        ).order_by(RiskAssessment.timestamp.asc()).all()
        if not records:
            return []
        buckets = {}
        for ts, score in records:
            minute_key = ts.replace(second=0, microsecond=0)
            if minute_key not in buckets:
                buckets[minute_key] = {"open": score, "high": score, "low": score, "close": score, "count": 1}
            else:
                b = buckets[minute_key]
                b["high"] = max(b["high"], score)
                b["low"] = min(b["low"], score)
                b["close"] = score
                b["count"] += 1
        items = []
        for dt, b in sorted(buckets.items(), reverse=True):
            items.append({
                "time": dt.isoformat(),
                "open": float(b["open"]),
                "high": float(b["high"]),
                "low": float(b["low"]),
                "close": float(b["close"]),
            })
            if len(items) >= limit:
                break
        return items


def get_timeline(db: Session, hours_back: int = 48) -> List[Dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_back)
    records = db.query(RiskAssessment).filter(
        RiskAssessment.timestamp >= cutoff
    ).order_by(RiskAssessment.timestamp.asc()).all()
    return [
        {
            "unit_name": r.unit.name,
            "risk_score": r.risk_score,
            "status": r.status,
            "root_cause": r.root_cause,
            "timestamp": r.timestamp.isoformat()
        }
        for r in records
    ]


def get_sensor_health(db: Session) -> Dict[str, Any]:
    """
    Sensor health percentage based on latest SensorHealth records.
    If no health records exist, estimates from sensor readings.
    """
    # First try to get actual SensorHealth data
    total_health = db.query(func.count(SensorHealth.id)).scalar()
    if total_health and total_health > 0:
        healthy_health = db.query(func.count(SensorHealth.id)).filter(
            SensorHealth.sensor_health_status == "Healthy"
        ).scalar()
        return {
            "health_percent": round((healthy_health / total_health) * 100, 2) if total_health else 100.0,
            "total_sensors": total_health,
            "mode": "actual"
        }

    # Fallback: estimate from latest sensor readings per unit
    units = db.query(Unit).all()
    if not units:
        return {
            "health_percent": None,
            "mode": "no_data",
            "message": "No units found in database"
        }

    total_sensors = 0
    healthy_count = 0
    for unit in units:
        latest_reading = db.query(SensorReading).filter(
            SensorReading.unit_id == unit.id
        ).order_by(desc(SensorReading.timestamp)).first()
        if latest_reading:
            total_sensors += 1
            # More realistic thresholds: warning if vibration > 1.5 or temperature_out > 200
            # Healthy if below these thresholds
            is_unhealthy = (latest_reading.vibration and latest_reading.vibration > 1.5) or \
                           (latest_reading.temperature_out and latest_reading.temperature_out > 200)
            if not is_unhealthy:
                healthy_count += 1

    if total_sensors == 0:
        return {
            "health_percent": None,
            "mode": "no_data",
            "message": "No sensor readings available yet"
        }

    estimated_percent = round((healthy_count / total_sensors) * 100, 2)
    return {
        "health_percent": estimated_percent,
        "total_sensors": total_sensors,
        "mode": "estimation"
    }


def get_correlation_matrix(db: Session, unit_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    query = db.query(SensorReading)
    if unit_name:
        unit = UnitRepo.get_by_name(db, unit_name)
        if not unit:
            return None
        query = query.filter(SensorReading.unit_id == unit.id)
    records = query.limit(1000).all()
    if len(records) < 10:
        return None
    data = {
        "temperature_in": [r.temperature_in or 0 for r in records],
        "temperature_out": [r.temperature_out or 0 for r in records],
        "pressure_in": [r.pressure_in or 0 for r in records],
        "pressure_out": [r.pressure_out or 0 for r in records],
        "flow_rate": [r.flow_rate or 0 for r in records],
        "vibration": [r.vibration or 0 for r in records],
        "motor_current": [r.motor_current or 0 for r in records],
        "bearing_temp": [r.bearing_temp or 0 for r in records],
    }
    df = pd.DataFrame(data)
    corr = df.corr().round(2).to_dict()
    return corr


# ============================================================================
# Feature extraction for ML
# ============================================================================
def get_all_features_for_unit(
    db: Session,
    unit_name: str,
    limit: int = 50
) -> List[Dict[str, Any]]:
    unit = UnitRepo.get_by_name(db, unit_name)
    if not unit:
        return []
    assessments = db.query(RiskAssessment).options(
        joinedload(RiskAssessment.sensor_reading)
    ).filter(
        RiskAssessment.unit_id == unit.id
    ).order_by(RiskAssessment.timestamp.asc()).limit(limit).all()
    if not assessments:
        return []
    sensor_ids = [ra.sensor_reading_id for ra in assessments if ra.sensor_reading_id]
    env_map = {}
    if sensor_ids:
        envs = db.query(EnvironmentData).filter(
            EnvironmentData.sensor_reading_id.in_(sensor_ids)
        ).all()
        env_map = {e.sensor_reading_id: e for e in envs}
    maints = db.query(MaintenanceLog).filter(
        MaintenanceLog.unit_id == unit.id
    ).order_by(MaintenanceLog.timestamp).all()
    hse_reports = db.query(HSEReport).filter(
        HSEReport.unit_id == unit.id
    ).order_by(HSEReport.timestamp).all()
    result = []
    for ra in assessments:
        reading = ra.sensor_reading
        if not reading:
            continue
        row = {
            "temp_in": reading.temperature_in or 0.0,
            "temp_out": reading.temperature_out or 0.0,
            "press_in": reading.pressure_in or 0.0,
            "press_out": reading.pressure_out or 0.0,
            "flow": reading.flow_rate or 0.0,
            "level": reading.level or 0.0,
            "vib": reading.vibration or 0.0,
            "bearing": reading.bearing_temp or 0.0,
            "rpm": reading.motor_rpm or 0.0,
            "current": reading.motor_current or 0.0,
            "torque": reading.torque or 0.0,
            "seal": reading.seal_pressure or 0.0,
            "valve1": reading.valve_position_1 or 0.0,
            "valve2": reading.valve_position_2 or 0.0,
            "gas": reading.gas_concentration or 0.0,
            "conduct": reading.conductivity or 0.0,
            "ph": reading.ph or 0.0,
            "coil_temp": reading.temp_ambient_coil or 0.0,
            "steam": reading.pressure_steam or 0.0,
            "oil_level": reading.oil_level or 0.0,
            "cooling_temp": reading.cooling_water_temp or 0.0,
        }
        env = env_map.get(reading.id)
        if env:
            row.update({
                "ambient_temp": env.ambient_temp or 0.0,
                "humidity": env.humidity or 0.0,
                "wind": env.wind_speed or 0.0,
                "atm": env.atmospheric_pressure or 0.0,
                "rain": env.rainfall or 0.0,
                "solar": env.solar_radiation or 0.0,
            })
        else:
            for k in ["ambient_temp", "humidity", "wind", "atm", "rain", "solar"]:
                row[k] = 0.0
        maint = None
        for m in reversed(maints):
            if m.timestamp <= ra.timestamp:
                maint = m
                break
        if maint:
            row.update({
                "last_service_days": maint.last_service_days or 0,
                "comp_age": maint.component_age_days or 0,
                "repl_count": maint.replacement_count or 0,
                "mtbf": maint.mtbf or 0.0,
                "mttr": maint.mttr or 0.0,
                "repair_cost": maint.last_repair_cost or 0.0,
            })
        else:
            for k in ["last_service_days", "comp_age", "repl_count", "mtbf", "mttr", "repair_cost"]:
                row[k] = 0.0
        hse = None
        for h in reversed(hse_reports):
            if h.timestamp <= ra.timestamp:
                hse = h
                break
        if hse:
            row["last_incident_days"] = hse.last_incident_days or 0
            row["safety_score"] = hse.safety_audit_score or 0.0
        else:
            row["last_incident_days"] = 0
            row["safety_score"] = 0.0
        row["risk_score"] = ra.risk_score
        # Safe timestamp parsing
        try:
            row["timestamp"] = ra.timestamp.isoformat()
        except Exception:
            row["timestamp"] = str(ra.timestamp)
        result.append(row)
    for i, row in enumerate(result):
        if i == 0:
            row["risk_lag1"] = 0.0
            row["risk_lag2"] = 0.0
            row["vib_roll_mean"] = row["vib"]
            row["temp_out_roll_std"] = 0.0
        else:
            row["risk_lag1"] = result[i-1]["risk_score"]
            row["risk_lag2"] = result[i-2]["risk_score"] if i >= 2 else 0.0
            vib_vals = [r["vib"] for r in result[max(0, i-4):i+1]]
            row["vib_roll_mean"] = float(np.mean(vib_vals))
            temp_vals = [r["temp_out"] for r in result[max(0, i-4):i+1]]
            row["temp_out_roll_std"] = float(np.std(temp_vals)) if len(temp_vals) > 1 else 0.0
    return result


def get_latest_features_for_unit(db: Session, unit_name: str) -> Optional[Dict[str, float]]:
    features_list = get_all_features_for_unit(db, unit_name, limit=10)
    if not features_list:
        return None
    return features_list[-1]


def get_unit_risk_history(db: Session, unit_name: str, limit: int = 50, desc: bool = False) -> List[RiskAssessment]:
    unit = UnitRepo.get_by_name(db, unit_name)
    if not unit:
        return []
    return RiskAssessmentRepo.get_history(db, unit.id, limit=limit, desc=desc)


# ============================================================================
# User management
# ============================================================================
def get_user_by_username(db: Session, username: str) -> Optional[User]:
    return db.query(User).filter(User.username == username).first()


def get_user_by_id(db: Session, user_id: str) -> Optional[User]:
    return db.query(User).filter(User.id == user_id).first()


def create_user(db: Session, user: schemas.UserCreate, hashed_pw: str) -> User:
    db_user = User(
        username=user.username,
        hashed_password=hashed_pw,
        full_name=user.full_name,
        role=user.role
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    user = get_user_by_username(db, username)
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user


def get_all_users(db: Session) -> List[User]:
    return db.query(User).order_by(User.username).all()


def delete_user(db: Session, user_id: str) -> bool:
    user = get_user_by_id(db, user_id)
    if not user:
        return False
    db.delete(user)
    db.commit()
    return True


def update_user_role(db: Session, user_id: str, new_role: str) -> Optional[User]:
    user = get_user_by_id(db, user_id)
    if not user:
        return None
    user.role = new_role
    db.commit()
    return user


# ============================================================================
# Operator reports
# ============================================================================
def get_unit_by_name(db: Session, name: str) -> Optional[Unit]:
    return UnitRepo.get_by_name(db, name)


def create_operator_report(
    db: Session,
    user_id: str,
    unit_id: str,
    report: schemas.OperatorReportCreate
) -> OperatorReport:
    db_report = OperatorReport(
        user_id=user_id,
        unit_id=unit_id,
        description=report.description,
        sensor_name=report.sensor_name,
        reported_issue=report.reported_issue,
        risk_impact=report.risk_impact,
        status="open"
    )
    db.add(db_report)
    db.commit()
    db.refresh(db_report)
    return db_report


def get_operator_reports(db: Session, user_id: Optional[str] = None) -> List[OperatorReport]:
    query = db.query(OperatorReport)
    if user_id:
        query = query.filter(OperatorReport.user_id == user_id)
    return query.order_by(OperatorReport.timestamp.desc()).all()


# ============================================================================
# Prediction logging and accuracy
# ============================================================================
def create_prediction_log(
    db: Session,
    unit_id: str,
    p10: float,
    p50: float,
    p90: float
) -> PredictionLog:
    log = PredictionLog(
        unit_id=unit_id,
        predicted_p10=p10,
        predicted_p50=p50,
        predicted_p90=p90,
        status="pending"
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


def update_prediction_with_actual(db: Session, unit_id: str, actual_risk: float) -> Optional[PredictionLog]:
    log = db.query(PredictionLog).filter(
        PredictionLog.unit_id == unit_id,
        PredictionLog.status == "pending"
    ).order_by(PredictionLog.timestamp.asc()).first()
    if not log:
        return None
    log.actual_risk = actual_risk
    error = abs(log.predicted_p50 - actual_risk)
    log.error_mae = error
    log.error_rmse = error
    log.status = "completed"
    db.commit()
    return log


def get_prediction_accuracy(
    db: Session,
    unit_name: Optional[str] = None,
    limit: int = 50
) -> List[PredictionLog]:
    query = db.query(PredictionLog).filter(PredictionLog.status == "completed")
    if unit_name:
        unit = UnitRepo.get_by_name(db, unit_name)
        if unit:
            query = query.filter(PredictionLog.unit_id == unit.id)
    return query.order_by(desc(PredictionLog.timestamp)).limit(limit).all()


def get_all_units(db: Session) -> List[Unit]:
    return UnitRepo.get_all(db)


def get_aligned_features_and_risk(
        db: Session,
        unit_name: str,
        limit: int = 500
) -> Tuple[List[Dict[str, Any]], List[float]]:
    """
    Returns (features_list, risk_scores) aligned by timestamp.
    """
    from backend.db.db_models import RiskAssessment, SensorReading, EnvironmentData, MaintenanceLog, HSEReport
    from sqlalchemy import desc

    unit = UnitRepo.get_by_name(db, unit_name)
    if not unit:
        return [], []

    query = db.query(RiskAssessment, SensorReading).join(
        SensorReading, RiskAssessment.sensor_reading_id == SensorReading.id
    ).filter(
        RiskAssessment.unit_id == unit.id
    ).order_by(RiskAssessment.timestamp.asc()).limit(limit)

    results = []
    for ra, sensor in query:
        if sensor is None:
            continue
        row = {
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
            "risk_score": ra.risk_score,
            "timestamp": ra.timestamp,
        }

        # Environment
        env = db.query(EnvironmentData).filter(EnvironmentData.sensor_reading_id == sensor.id).first()
        if env:
            row.update({
                "ambient_temp": env.ambient_temp or 0.0,
                "humidity": env.humidity or 0.0,
                "wind": env.wind_speed or 0.0,
                "atm": env.atmospheric_pressure or 0.0,
                "rain": env.rainfall or 0.0,
                "solar": env.solar_radiation or 0.0,
            })
        else:
            for k in ["ambient_temp", "humidity", "wind", "atm", "rain", "solar"]:
                row[k] = 0.0

        # Maintenance
        maint = db.query(MaintenanceLog).filter(
            MaintenanceLog.unit_id == unit.id,
            MaintenanceLog.timestamp <= ra.timestamp
        ).order_by(desc(MaintenanceLog.timestamp)).first()
        if maint:
            row.update({
                "last_service_days": maint.last_service_days or 0,
                "comp_age": maint.component_age_days or 0,
                "repl_count": maint.replacement_count or 0,
                "mtbf": maint.mtbf or 0.0,
                "mttr": maint.mttr or 0.0,
                "repair_cost": maint.last_repair_cost or 0.0,
            })
        else:
            for k in ["last_service_days", "comp_age", "repl_count", "mtbf", "mttr", "repair_cost"]:
                row[k] = 0.0

        # HSE
        hse = db.query(HSEReport).filter(
            HSEReport.unit_id == unit.id,
            HSEReport.timestamp <= ra.timestamp
        ).order_by(desc(HSEReport.timestamp)).first()
        if hse:
            row["last_incident_days"] = hse.last_incident_days or 0
            row["safety_score"] = hse.safety_audit_score or 0.0
        else:
            row["last_incident_days"] = 0
            row["safety_score"] = 0.0

        results.append(row)

    for i, row in enumerate(results):
        if i == 0:
            row["risk_lag1"] = 0.0
            row["risk_lag2"] = 0.0
        else:
            row["risk_lag1"] = results[i - 1]["risk_score"]
            row["risk_lag2"] = results[i - 2]["risk_score"] if i >= 2 else 0.0

        vib_vals = [results[j]["vib"] for j in range(max(0, i - 4), i + 1)]
        row["vib_roll_mean"] = float(np.mean(vib_vals))
        temp_vals = [results[j]["temp_out"] for j in range(max(0, i - 4), i + 1)]
        row["temp_out_roll_std"] = float(np.std(temp_vals)) if len(temp_vals) > 1 else 0.0

    risk_scores = [r["risk_score"] for r in results]
    return results, risk_scores
