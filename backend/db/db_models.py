"""
Normalized SQLAlchemy data models following 4NF principles.
Adds PredictionLog for tracking prediction accuracy, performance metrics, and model behavior.
"""

import uuid
from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey, Index, Text, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from backend.db.database import Base

def gen_uuid():
    return str(uuid.uuid4())

class Unit(Base):
    __tablename__ = "units"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    name = Column(String(100), unique=True, nullable=False, index=True)
    capacity = Column(Float)
    max_temperature = Column(Float)
    max_pressure = Column(Float)
    location = Column(String(200))
    created_at = Column(DateTime, server_default=func.now())

    sensor_readings = relationship("SensorReading", back_populates="unit", cascade="all, delete-orphan")
    maintenance_logs = relationship("MaintenanceLog", back_populates="unit", cascade="all, delete-orphan")
    hse_reports = relationship("HSEReport", back_populates="unit", cascade="all, delete-orphan")
    risk_assessments = relationship("RiskAssessment", back_populates="unit", cascade="all, delete-orphan")
    prediction_logs = relationship("PredictionLog", back_populates="unit")

class SensorReading(Base):
    __tablename__ = "sensor_readings"
    __table_args__ = (
        Index("ix_sensor_unit_time", "unit_id", "timestamp"),
    )
    id = Column(String(36), primary_key=True, default=gen_uuid)
    unit_id = Column(String(36), ForeignKey("units.id"), nullable=False, index=True)
    timestamp = Column(DateTime, server_default=func.now(), index=True)
    temperature_in = Column(Float)
    temperature_out = Column(Float)
    pressure_in = Column(Float)
    pressure_out = Column(Float)
    flow_rate = Column(Float)
    level = Column(Float)
    vibration = Column(Float)
    bearing_temp = Column(Float)
    motor_rpm = Column(Float)
    motor_current = Column(Float)
    torque = Column(Float)
    seal_pressure = Column(Float)
    valve_position_1 = Column(Float)
    valve_position_2 = Column(Float)
    gas_concentration = Column(Float)
    conductivity = Column(Float)
    ph = Column(Float)
    temp_ambient_coil = Column(Float)
    pressure_steam = Column(Float)
    oil_level = Column(Float)
    cooling_water_temp = Column(Float)

    unit = relationship("Unit", back_populates="sensor_readings")
    environment = relationship("EnvironmentData", uselist=False, back_populates="sensor_reading", cascade="all, delete-orphan")
    operator_note = relationship("OperatorNote", uselist=False, back_populates="sensor_reading", cascade="all, delete-orphan")
    health = relationship("SensorHealth", uselist=False, back_populates="sensor_reading", cascade="all, delete-orphan")
    risk_assessment = relationship("RiskAssessment", uselist=False, back_populates="sensor_reading", cascade="all, delete-orphan")

class EnvironmentData(Base):
    __tablename__ = "environment_data"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    sensor_reading_id = Column(String(36), ForeignKey("sensor_readings.id"), unique=True, nullable=False)
    ambient_temp = Column(Float)
    humidity = Column(Float)
    wind_speed = Column(Float)
    atmospheric_pressure = Column(Float)
    rainfall = Column(Float)
    solar_radiation = Column(Float)
    sensor_reading = relationship("SensorReading", back_populates="environment")

class MaintenanceLog(Base):
    __tablename__ = "maintenance_logs"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    unit_id = Column(String(36), ForeignKey("units.id"), nullable=False, index=True)
    timestamp = Column(DateTime, server_default=func.now())
    last_service_days = Column(Integer)
    component_age_days = Column(Integer)
    maintenance_type = Column(String(100))
    replacement_count = Column(Integer)
    mtbf = Column(Float)
    mttr = Column(Float)
    last_repair_cost = Column(Float)
    unit = relationship("Unit", back_populates="maintenance_logs")

class HSEReport(Base):
    __tablename__ = "hse_reports"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    unit_id = Column(String(36), ForeignKey("units.id"), nullable=False, index=True)
    timestamp = Column(DateTime, server_default=func.now())
    last_incident_days = Column(Integer)
    incident_type = Column(String(100))
    safety_audit_score = Column(Float)
    risk_assessment_level = Column(String(50))
    operator_training_level = Column(String(50))
    unit = relationship("Unit", back_populates="hse_reports")

class OperatorNote(Base):
    __tablename__ = "operator_notes"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    sensor_reading_id = Column(String(36), ForeignKey("sensor_readings.id"), unique=True, nullable=False)
    operator_notes = Column(Text)
    shift_handover_issues = Column(Text)
    manual_alert = Column(String(100))
    observation = Column(Text)
    sensor_reading = relationship("SensorReading", back_populates="operator_note")

class SensorHealth(Base):
    __tablename__ = "sensor_health"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    sensor_reading_id = Column(String(36), ForeignKey("sensor_readings.id"), unique=True, nullable=False)
    sensor_health_status = Column(String(50))
    calibration_days_left = Column(Integer)
    drift_indicator = Column(Float)
    sensor_reading = relationship("SensorReading", back_populates="health")

class RiskAssessment(Base):
    __tablename__ = "risk_assessments"
    __table_args__ = (
        Index("ix_risk_unit_time", "unit_id", "timestamp"),
    )
    id = Column(String(36), primary_key=True, default=gen_uuid)
    sensor_reading_id = Column(String(36), ForeignKey("sensor_readings.id"), unique=True, nullable=False)
    unit_id = Column(String(36), ForeignKey("units.id"), nullable=False, index=True)
    timestamp = Column(DateTime, server_default=func.now(), index=True)
    risk_score = Column(Float, nullable=False)
    status = Column(String(50), nullable=False)
    root_cause = Column(String(200))
    prediction_confidence = Column(Float)

    unit = relationship("Unit", back_populates="risk_assessments")
    sensor_reading = relationship("SensorReading", back_populates="risk_assessment")

class User(Base):
    __tablename__ = "users"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    username = Column(String(50), unique=True, nullable=False, index=True)
    hashed_password = Column(String(200), nullable=False)
    full_name = Column(String(100))
    role = Column(String(20), nullable=False, default="operator")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now())

class OperatorReport(Base):
    __tablename__ = "operator_reports"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    unit_id = Column(String(36), ForeignKey("units.id"), nullable=False)
    timestamp = Column(DateTime, server_default=func.now(), index=True)
    description = Column(Text, nullable=False)
    sensor_name = Column(String(100))
    reported_issue = Column(String(200))
    risk_impact = Column(String(20))
    status = Column(String(20), default="open")
    user = relationship("User")
    unit = relationship("Unit")

class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    unit_id = Column(String(36), ForeignKey("units.id"), nullable=False, index=True)
    timestamp = Column(DateTime, server_default=func.now(), index=True)
    predicted_p10 = Column(Float)
    predicted_p50 = Column(Float)
    predicted_p90 = Column(Float)
    actual_risk = Column(Float, nullable=True)
    error_mae = Column(Float, nullable=True)
    error_rmse = Column(Float, nullable=True)
    status = Column(String(20), default="pending")

    unit = relationship("Unit", back_populates="prediction_logs")