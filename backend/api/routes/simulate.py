"""
Stress test and emergency scenario simulation endpoints.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from backend.api.dependencies import get_db
from backend.auth.permissions import allow_admin 
from backend.db.db_models import Unit
from backend.services.live_simulator import LiveSimulator
from backend.logger import logger

router = APIRouter()


@router.post("/simulate/cascade")
def simulate_cascade(
    db: Session = Depends(get_db),
    user = Depends(allow_admin)  
):
    units = db.query(Unit).all()
    sim = LiveSimulator()
    affected = 0
    for u in units:
        try:
            sim.inject_fault(db, u.name, "Critical")
            affected += 1
        except Exception as e:
            logger.error(f"Error injecting fault for {u.name}: {e}", exc_info=True)
    return {"status": "stress test executed", "units_affected": affected}