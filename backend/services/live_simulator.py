"""
Real-time refinery simulator with fault injection, sensor health tracking,
and integration with all ML services. Thread-safe and production-ready.
"""

import threading
import time
import random
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from sqlalchemy.orm import Session

from backend.db.database import SessionLocal
from backend.db.db_models import (
    Unit, SensorReading, EnvironmentData, MaintenanceLog,
    HSEReport, RiskAssessment, PredictionLog, SensorHealth
)
from backend.services.data_generator import ProfessionalRefinerySimulator
from backend.config import settings
from backend.logger import logger

# Unit catalog (name, capacity, max_temp, max_pressure)
UNITS: List[tuple] = [
    ("Crude Distillation Unit", 10000, 420, 5.5),
    ("Vacuum Distillation Unit", 8000, 400, 4.0),
    ("Fluid Catalytic Cracking", 6000, 520, 3.5),
    ("Hydrocracker Unit", 5000, 450, 7.0),
    ("Reformer Unit", 4000, 400, 6.0),
    ("Hydrotreater Unit", 3500, 380, 5.5),
    ("Alkylation Unit", 3000, 200, 4.0),
    ("Coker Unit", 2500, 480, 2.5),
]

CAUSES: Dict[str, List[str]] = {
    "Normal": ["Normal operation"],
    "Warning": [
        "High Temperature", "High Vibration", "Pressure deviation",
        "Flow reduction", "Seal leakage", "Bearing temperature rise"
    ],
    "Critical": [
        "Pump Cavitation", "Bearing Failure", "Valve Failure",
        "Leak Detected", "Motor Overheating", "Catalyst Deactivation"
    ]
}


@dataclass
class SimulatorConfig:
    """Configuration for the live simulator."""
    interval_seconds: int = 3
    fault_probability: float = 0.05
    warning_probability: float = 0.10
    ambient_temp_base: float = 25.0
    ambient_temp_amplitude: float = 8.0


class LiveSimulator:
    """
    Thread-safe real-time data generator. Each unit's data is committed
    individually to avoid transaction rollback on single unit failure.
    """

    def __init__(self, config: Optional[SimulatorConfig] = None):
        if config is None:
            config = SimulatorConfig()
        # Override interval with global setting if needed
        step_seconds = getattr(settings, "PREDICTION_STEP_SECONDS", 3)
        if config.interval_seconds != step_seconds:
            logger.info(
                f"Adjusting LiveSimulator interval from {config.interval_seconds}s "
                f"to match PREDICTION_STEP_SECONDS={step_seconds}s"
            )
            config.interval_seconds = step_seconds
        self.config = config
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._base_simulator = ProfessionalRefinerySimulator({})

    def start(self) -> None:
        """Start the background data generation thread."""
        with self._lock:
            if self._running:
                logger.warning("LiveSimulator already running")
                return
            self._ensure_units_exist()
            self._running = True
            self._thread = threading.Thread(target=self._run, daemon=True, name="LiveSimulator")
            self._thread.start()
            logger.info(f"LiveSimulator started with interval {self.config.interval_seconds}s")

    def stop(self) -> None:
        """Stop the background thread gracefully."""
        with self._lock:
            self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            logger.info("LiveSimulator stopped")

    def _ensure_units_exist(self) -> None:
        """Create predefined units if they don't exist."""
        db = SessionLocal()
        try:
            existing_names = {u.name for u in db.query(Unit).all()}
            for name, cap, tmax, pmax in UNITS:
                if name not in existing_names:
                    db.add(Unit(name=name, capacity=cap, max_temperature=tmax, max_pressure=pmax))
            db.commit()
        except Exception as e:
            logger.error(f"Failed to ensure units exist: {e}")
            db.rollback()
        finally:
            db.close()

    def _run(self) -> None:
        """Main loop: generate data for all units, commit per unit."""
        while self._running:
            start_time = time.time()
            session = SessionLocal()
            try:
                units = session.query(Unit).all()
                for unit in units:
                    try:
                        self._generate_reading(session, unit)
                        session.commit()   # 🔥 FIXED: commit per unit
                    except Exception as unit_error:
                        session.rollback()
                        logger.error(f"Failed to generate reading for {unit.name}: {unit_error}", exc_info=True)
            except Exception as e:
                logger.error(f"LiveSimulator iteration failed: {e}", exc_info=True)
            finally:
                session.close()
            elapsed = time.time() - start_time
            sleep_time = max(0, self.config.interval_seconds - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _generate_reading(self, db: Session, unit: Unit) -> None:
        """Generate a single sensor reading and all related records."""
        now = datetime.now(timezone.utc)
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        operating_hours = (now - start).total_seconds() / 3600.0
        ambient_temp = self.config.ambient_temp_base + self.config.ambient_temp_amplitude * \
                       (operating_hours % 24 / 24 * 2 * 3.14159)
        ambient_temp = max(5.0, min(45.0, ambient_temp))

        unit_config = {
            'unit_name': unit.name,
            'heat_input': 1200 + random.uniform(-100, 100),
            'mtbf': 8000,
            'ua_value': 55,
            'pipe_diameter': 0.35,
            'pipe_length': 45
        }
        self._base_simulator.config = unit_config
        raw = self._base_simulator.generate_realistic_sensor_data(
            unit.name, operating_hours, ambient_temp
        )

        rand = random.random()
        if rand < self.config.fault_probability:
            status = "Critical"
            raw['temperature_out'] += random.uniform(80, 150)
            raw['vibration'] += random.uniform(4, 10)
            raw['pressure_out'] *= random.uniform(0.7, 0.9)
        elif rand < self.config.fault_probability + self.config.warning_probability:
            status = "Warning"
            raw['temperature_out'] += random.uniform(20, 50)
            raw['vibration'] += random.uniform(1, 3)
        else:
            status = "Normal"

        degradation = min(1.0, operating_hours / (unit_config['mtbf'] * 24))
        base_risk = 10 + degradation * 30 + random.uniform(-5, 5)
        if status == "Critical":
            risk = min(100, base_risk + random.uniform(25, 40))
        elif status == "Warning":
            risk = min(100, base_risk + random.uniform(10, 25))
        else:
            risk = min(100, base_risk)

        cause_list = CAUSES.get(status, ["Unknown"])
        root_cause = random.choice(cause_list)

        sensor = SensorReading(
            unit_id=unit.id,
            temperature_in=raw.get('temperature_in', 150.0),
            temperature_out=raw.get('temperature_out', 180.0),
            pressure_in=raw.get('pressure_in', 3.0),
            pressure_out=raw.get('pressure_out', 2.8),
            flow_rate=raw.get('flow_rate', 50.0),
            level=raw.get('level', 60.0),
            vibration=raw.get('vibration', 0.5),
            motor_current=raw.get('motor_current', 10.0),
            seal_pressure=raw.get('seal_pressure', 2.0),
            gas_concentration=raw.get('gas_concentration', 50.0),
            bearing_temp=raw.get('bearing_temp', 70.0),
            motor_rpm=raw.get('motor_rpm', 1450.0),
            torque=raw.get('torque', 500.0),
            valve_position_1=raw.get('valve_position_1', 50.0),
            valve_position_2=raw.get('valve_position_2', 50.0),
            conductivity=raw.get('conductivity', 0.5),
            ph=raw.get('ph', 7.0),
            temp_ambient_coil=raw.get('temp_ambient_coil', 40.0),
            pressure_steam=raw.get('pressure_steam', 10.0),
            oil_level=raw.get('oil_level', 80.0),
            cooling_water_temp=raw.get('cooling_water_temp', 30.0),
        )
        db.add(sensor)
        db.flush()

        env = EnvironmentData(
            sensor_reading_id=sensor.id,
            ambient_temp=ambient_temp,
            humidity=60.0 + random.uniform(-10, 10),
            wind_speed=5.0 + random.uniform(-3, 3),
            atmospheric_pressure=1013.0 + random.uniform(-5, 5),
            rainfall=max(0, random.uniform(0, 2)),
            solar_radiation=random.uniform(0, 800)
        )
        db.add(env)

        last_service_days = int(random.uniform(10, 300) * (1 + degradation))
        component_age_days = int(random.uniform(365, 3000) * (1 + degradation))
        maint = MaintenanceLog(
            unit_id=unit.id,
            last_service_days=last_service_days,
            component_age_days=component_age_days,
            maintenance_type="Corrective" if status != "Normal" else "Preventive",
            replacement_count=random.randint(0, 5),
            mtbf=8000.0,
            mttr=24.0,
            last_repair_cost=random.uniform(5000, 50000) if status != "Normal" else 0
        )
        db.add(maint)

        last_incident_days = random.randint(30, 365) if status == "Normal" else random.randint(0, 30)
        hse = HSEReport(
            unit_id=unit.id,
            last_incident_days=last_incident_days,
            incident_type="None" if last_incident_days > 30 else "Near Miss",
            safety_audit_score=random.uniform(60, 95),
            risk_assessment_level=status,
            operator_training_level=random.choice(["Basic", "Intermediate", "Advanced"])
        )
        db.add(hse)

        health_status = "Healthy"
        drift = 0.0
        calibration_days = random.randint(30, 180)
        if sensor.vibration > 3.0 or sensor.temperature_out > 250:
            health_status = "Degraded"
            drift = random.uniform(0.1, 0.5)
            calibration_days = max(0, calibration_days - 30)
        elif sensor.vibration > 2.0 or sensor.temperature_out > 200:
            health_status = "Caution"
            drift = random.uniform(0.05, 0.2)
        else:
            health_status = "Healthy"
            drift = random.uniform(0, 0.05)
        sensor_health = SensorHealth(
            sensor_reading_id=sensor.id,
            sensor_health_status=health_status,
            calibration_days_left=calibration_days,
            drift_indicator=drift
        )
        db.add(sensor_health)

        risk_assessment = RiskAssessment(
            sensor_reading_id=sensor.id,
            unit_id=unit.id,
            risk_score=risk,
            status=status,
            root_cause=root_cause,
            prediction_confidence=random.uniform(0.7, 0.95) if status != "Normal" else 0.9
        )
        db.add(risk_assessment)
        db.flush()

        # Update pending prediction logs
        pending = db.query(PredictionLog).filter(
            PredictionLog.unit_id == unit.id,
            PredictionLog.status == "pending"
        ).order_by(PredictionLog.timestamp.asc()).first()
        if pending:
            pending.actual_risk = risk
            error = abs(pending.predicted_p50 - risk)
            pending.error_mae = error
            pending.error_rmse = error
            pending.status = "completed"

    def inject_fault(self, db: Session, unit_name: str, severity: str = "Critical") -> None:
        """
        Inject a deliberate fault into a specific unit for stress testing.
        This method is synchronous and uses a separate session internally
        to avoid deadlocks with the background thread. It does NOT use the
        passed db session for writes (only for reading unit).
        """
        # First, get the unit using the provided session (read-only)
        unit = db.query(Unit).filter(Unit.name == unit_name).first()
        if not unit:
            logger.warning(f"Cannot inject fault: unit {unit_name} not found")
            return

        # Use a separate session for writing to avoid locks with live_simulator
        write_db = SessionLocal()
        try:
            now = datetime.now(timezone.utc)
            start = datetime(2024, 1, 1, tzinfo=timezone.utc)
            operating_hours = (now - start).total_seconds() / 3600.0
            ambient_temp = 25.0

            unit_config = {'unit_name': unit.name, 'heat_input': 1200, 'mtbf': 8000}
            self._base_simulator.config = unit_config
            raw = self._base_simulator.generate_realistic_sensor_data(unit.name, operating_hours, ambient_temp)

            if severity == "Critical":
                raw['temperature_out'] += random.uniform(120, 200)
                raw['vibration'] += random.uniform(5, 10)
                raw['pressure_out'] *= random.uniform(0.5, 0.8)
                status = "Critical"
                risk = random.uniform(85, 100)
                root_cause = random.choice(CAUSES['Critical'])
            else:
                raw['temperature_out'] += random.uniform(20, 50)
                raw['vibration'] += random.uniform(1, 3)
                status = "Warning"
                risk = random.uniform(55, 75)
                root_cause = random.choice(CAUSES['Warning'])

            sensor = SensorReading(
                unit_id=unit.id,
                temperature_in=raw.get('temperature_in', 150.0),
                temperature_out=raw.get('temperature_out', 180.0),
                pressure_in=raw.get('pressure_in', 3.0),
                pressure_out=raw.get('pressure_out', 2.8),
                flow_rate=raw.get('flow_rate', 50.0),
                level=raw.get('level', 60.0),
                vibration=raw.get('vibration', 0.5),
                motor_current=raw.get('motor_current', 10.0),
                seal_pressure=raw.get('seal_pressure', 2.0),
                gas_concentration=raw.get('gas_concentration', 50.0),
            )
            write_db.add(sensor)
            write_db.flush()

            env = EnvironmentData(sensor_reading_id=sensor.id, ambient_temp=ambient_temp, humidity=60, wind_speed=5, atmospheric_pressure=1013)
            write_db.add(env)

            maint = MaintenanceLog(unit_id=unit.id, last_service_days=300, component_age_days=2500,
                                   maintenance_type="Emergency", replacement_count=3, mtbf=8000, mttr=24)
            write_db.add(maint)

            hse = HSEReport(unit_id=unit.id, last_incident_days=0, incident_type="Critical", safety_audit_score=50)
            write_db.add(hse)

            health_status = "Critical" if severity == "Critical" else "Degraded"
            sensor_health = SensorHealth(sensor_reading_id=sensor.id, sensor_health_status=health_status,
                                         calibration_days_left=0, drift_indicator=0.8)
            write_db.add(sensor_health)

            risk_assessment = RiskAssessment(
                sensor_reading_id=sensor.id,
                unit_id=unit.id,
                risk_score=risk,
                status=status,
                root_cause=root_cause
            )
            write_db.add(risk_assessment)
            write_db.commit()
            logger.info(f"Injected {severity} fault into {unit_name}, risk={risk:.1f}")
        except Exception as e:
            write_db.rollback()
            logger.error(f"Failed to inject fault into {unit_name}: {e}", exc_info=True)
            raise
        finally:
            write_db.close()