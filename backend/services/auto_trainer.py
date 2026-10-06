"""
Automatic retraining of all ML models (quantile, anomaly, RCA, ensemble)
on a schedule. Thread-safe and production-ready with graceful shutdown.
"""

import threading
from datetime import datetime, timedelta, timezone
from typing import Optional


from backend.db.database import SessionLocal
from backend.db.db_models import Unit
from backend.services.mlpipeline import train_and_save_professional
from backend.services.predictor_ml import QuantilePredictionService
from backend.services.anomaly_detector import AnomalyDetectionService
from backend.services.rootcauseanalyze import RootCauseAnalyzerService
from backend.logger import logger


class AutoTrainer:
    """
    Automatic retraining scheduler for all ML models.
    Runs daily at specified hour/minute. Supports graceful shutdown.
    """

    def __init__(self, train_hour: int = 2, train_minute: int = 0):
        self.train_hour = train_hour
        self.train_minute = train_minute
        self.running = False
        self._stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self.running:
                logger.warning("AutoTrainer already running")
                return
            self.running = True
            self._stop_event.clear()
            self.thread = threading.Thread(target=self._run_scheduler, daemon=True, name="AutoTrainer")
            self.thread.start()
            logger.info(f"AutoTrainer scheduled for {self.train_hour:02d}:{self.train_minute:02d} daily")

    def stop(self) -> None:
        with self._lock:
            self.running = False
            self._stop_event.set()
        if self.thread:
            self.thread.join(timeout=10.0)
            logger.info("AutoTrainer stopped")

    def _run_scheduler(self) -> None:
        while self.running and not self._stop_event.is_set():
            now = datetime.now(timezone.utc)
            target = now.replace(hour=self.train_hour, minute=self.train_minute, second=0, microsecond=0)
            if target <= now:
                target += timedelta(days=1)

            seconds_until_target = (target - now).total_seconds()
            if seconds_until_target > 0:
                self._sleep(seconds_until_target)

            if not self.running or self._stop_event.is_set():
                break

            self._retrain_all_models()

    def _sleep(self, seconds: float) -> None:
        """Sleep in chunks to allow early termination."""
        while seconds > 0 and not self._stop_event.is_set():
            sleep_chunk = min(30, seconds)
            self._stop_event.wait(sleep_chunk)
            seconds -= sleep_chunk

    def _retrain_all_models(self) -> None:
        """Retrain all ML models with periodic check for stop."""
        logger.info("Starting automatic retraining of all models")
        db = SessionLocal()
        try:
            if self._stop_event.is_set():
                return
            logger.info("Retraining ensemble models...")
            train_and_save_professional(db)
            units = db.query(Unit).all()
            for unit in units:
                if self._stop_event.is_set():
                    return
                train_and_save_professional(db, unit.name)

            if self._stop_event.is_set():
                return
            logger.info("Retraining quantile models...")
            quantile_service = QuantilePredictionService()
            for unit in units:
                if self._stop_event.is_set():
                    return
                try:
                    quantile_service.train(db, unit.name, force_retrain=True)
                except Exception as e:
                    logger.error(f"Quantile retrain failed for {unit.name}: {e}")

            if self._stop_event.is_set():
                return
            logger.info("Retraining anomaly detection models...")
            anomaly_service = AnomalyDetectionService()
            for unit in units:
                if self._stop_event.is_set():
                    return
                try:
                    anomaly_service.train(db, unit.name, force_retrain=True)
                except Exception as e:
                    logger.error(f"Anomaly retrain failed for {unit.name}: {e}")

            if self._stop_event.is_set():
                return
            logger.info("Retraining predictive RCA models...")
            rca_service = RootCauseAnalyzerService()
            for unit in units:
                if self._stop_event.is_set():
                    return
                try:
                    rca_service.train_predictive(db, unit.name)
                except Exception as e:
                    logger.error(f"RCA retrain failed for {unit.name}: {e}")

            logger.info("Automatic retraining completed successfully")
        except Exception as e:
            logger.error(f"AutoTrainer failed: {e}", exc_info=True)
        finally:
            db.close()


_auto_trainer = AutoTrainer()


def get_auto_trainer() -> AutoTrainer:
    return _auto_trainer