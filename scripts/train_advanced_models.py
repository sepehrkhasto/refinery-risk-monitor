#!/usr/bin/env python
"""
Advanced model training script – Enterprise Edition.
- Generates sample data if database is empty (8 units, 500 records per unit)
- Trains per-unit quantile models (P10/P50/P90)
- Trains per-unit anomaly detection models
- Trains per-unit predictive RCA models
- Uses progress bars (tqdm) for long operations
"""

import sys
import os
import time
from typing import List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sqlalchemy.orm import Session
from tqdm import tqdm

from backend.db.database import SessionLocal
from backend.db.db_models import Unit, SensorReading, EnvironmentData, MaintenanceLog, HSEReport, RiskAssessment
from backend.services.data_generator import ProfessionalRefinerySimulator
from backend.services.predictor_ml import QuantilePredictionService
from backend.services.anomaly_detector import AnomalyDetectionService
from backend.services.rootcauseanalyze import RootCauseAnalyzerService
from backend.logger import logger

# ============================================================================
# Unit catalog – MUST match the units in live_simulator.py
# ============================================================================
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


def generate_sample_data_if_empty(db: Session) -> bool:
    """
    Generate synthetic data for 8 units if no risk assessments exist.
    Returns True if data was generated, False if data already present.
    """
    if db.query(RiskAssessment).count() > 0:
        logger.info("Database already contains risk data. Skipping sample generation.")
        return False

    logger.info("No data found – generating sample records for 8 units (500 records each)...")

    total_records = 0
    for name, capacity, max_temp, max_pressure in UNITS:
        # ✅ Check if unit already exists (to avoid IntegrityError)
        unit = db.query(Unit).filter(Unit.name == name).first()
        if unit is None:
            unit = Unit(
                name=name,
                capacity=capacity,
                max_temperature=max_temp,
                max_pressure=max_pressure,
                location="Refinery"
            )
            db.add(unit)
            db.commit()
            logger.info(f"✅ Created new unit: {name}")
        else:
            logger.info(f"Unit {name} already exists, reusing it.")

        config = {'unit_name': name, 'heat_input': 1200, 'mtbf': 8000}
        sim = ProfessionalRefinerySimulator(config)

        with tqdm(total=500, desc=f"Generating {name}", unit="records") as pbar:
            for i in range(500):
                reading = sim.get_reading(unit.id)

                sensor = SensorReading(
                    unit_id=unit.id,
                    temperature_in=reading.get('temperature_in', 150.0),
                    temperature_out=reading.get('temperature_out', 180.0),
                    pressure_in=reading.get('pressure_in', 3.0),
                    pressure_out=reading.get('pressure_out', 2.8),
                    flow_rate=reading.get('flow_rate', 50.0),
                    level=reading.get('level', 60.0),
                    vibration=reading.get('vibration', 0.5),
                    motor_current=reading.get('motor_current', 10.0),
                    seal_pressure=reading.get('seal_pressure', 2.0),
                    gas_concentration=reading.get('gas_concentration', 50.0),
                    bearing_temp=reading.get('bearing_temp', 70.0),
                    motor_rpm=reading.get('motor_rpm', 1450.0),
                    torque=reading.get('torque', 500.0),
                    valve_position_1=reading.get('valve_position_1', 50.0),
                    valve_position_2=reading.get('valve_position_2', 50.0),
                    conductivity=reading.get('conductivity', 0.5),
                    ph=reading.get('ph', 7.0),
                    temp_ambient_coil=reading.get('temp_ambient_coil', 40.0),
                    pressure_steam=reading.get('pressure_steam', 10.0),
                    oil_level=reading.get('oil_level', 80.0),
                    cooling_water_temp=reading.get('cooling_water_temp', 30.0),
                )
                db.add(sensor)
                db.flush()

                env = EnvironmentData(
                    sensor_reading_id=sensor.id,
                    ambient_temp=25.0 + (i % 10) * 0.5,
                    humidity=60.0,
                    wind_speed=5.0,
                    atmospheric_pressure=1013.0
                )
                db.add(env)

                last_service = 30
                comp_age = 365
                status = "Normal"
                root_cause = "None"
                if i % 10 == 0:
                    status = "Warning"
                    root_cause = "High Temperature"
                elif i % 25 == 0:
                    status = "Critical"
                    root_cause = "Pump Cavitation"
                    last_service = 300
                    comp_age = 2500

                maint = MaintenanceLog(
                    unit_id=unit.id,
                    last_service_days=last_service,
                    component_age_days=comp_age,
                    replacement_count=2,
                    mtbf=8000,
                    mttr=24
                )
                db.add(maint)

                hse = HSEReport(
                    unit_id=unit.id,
                    last_incident_days=180,
                    safety_audit_score=75.0
                )
                db.add(hse)

                base_risk = reading.get('risk_score', 50.0)
                if status == "Warning":
                    base_risk = max(55.0, base_risk + 15.0)
                elif status == "Critical":
                    base_risk = max(80.0, base_risk + 30.0)

                risk = RiskAssessment(
                    sensor_reading_id=sensor.id,
                    unit_id=unit.id,
                    risk_score=min(100.0, base_risk),
                    status=status,
                    root_cause=root_cause
                )
                db.add(risk)

                total_records += 1
                pbar.update(1)
                if (i + 1) % 100 == 0:
                    db.commit()

        db.commit()
        logger.info(f"  {name} – 500 records generated")

    logger.info(f"Sample data generation complete: {total_records} records across {len(UNITS)} units.")
    return True


def train_per_unit_models(db: Session, units: List[Unit]) -> None:
    """Train quantile, anomaly, and RCA models for each unit."""
    quantile_service = QuantilePredictionService()
    anomaly_service = AnomalyDetectionService()
    rca_service = RootCauseAnalyzerService()

    logger.info("Training quantile models (per unit)...")
    for unit in tqdm(units, desc="Quantile models", unit="unit"):
        try:
            quantile_service.train(db, unit.name, force_retrain=True)
        except Exception as e:
            logger.error(f"Quantile training failed for {unit.name}: {e}")

    logger.info("Training anomaly detection models (per unit)...")
    for unit in tqdm(units, desc="Anomaly models", unit="unit"):
        try:
            anomaly_service.train(db, unit.name, force_retrain=True)
        except Exception as e:
            logger.error(f"Anomaly training failed for {unit.name}: {e}")

    logger.info("Training predictive RCA models (per unit)...")
    for unit in tqdm(units, desc="RCA models", unit="unit"):
        try:
            rca_service.train_predictive(db, unit.name)
        except Exception as e:
            logger.error(f"RCA training failed for {unit.name}: {e}")


def main() -> None:
    db = SessionLocal()
    try:
        logger.info("Starting advanced model training...")
        start_time = time.time()

        generate_sample_data_if_empty(db)
        units = db.query(Unit).all()
        if not units:
            logger.error("No units found in database. Cannot train models.")
            return
        train_per_unit_models(db, units)

        elapsed = time.time() - start_time
        logger.info(f"All training completed successfully in {elapsed:.1f} seconds.")
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
    finally:
        db.close()


if __name__ == "__main__":
    main()