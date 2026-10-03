"""
WebSocket manager for real-time risk data push to connected clients.
Thread-safe, supports multiple units and broadcast with automatic heartbeat.
Heartbeat tasks are properly cancelled on disconnect.
"""

import asyncio
from typing import Dict, Set

from fastapi import WebSocket

from backend.db.database import SessionLocal
from backend.db.crud import RiskAssessmentRepo
from backend.logger import logger


class ConnectionManager:
    """
    Manages WebSocket connections and broadcasts risk updates.
    Each connection has its own heartbeat task that is cancelled on disconnect.
    All operations are thread-safe using asyncio locks.
    """

    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()
        self._heartbeat_interval = 30  # seconds
        self._heartbeat_tasks: Dict[WebSocket, asyncio.Task] = {}

    async def connect(self, websocket: WebSocket, unit_name: str = "all") -> None:
        """Accept new connection and register it."""
        await websocket.accept()
        async with self._lock:
            if unit_name not in self.active_connections:
                self.active_connections[unit_name] = set()
            self.active_connections[unit_name].add(websocket)
        logger.info(f"WebSocket connected for unit {unit_name} (total: {len(self.active_connections[unit_name])})")
        task = asyncio.create_task(self._heartbeat(websocket))
        self._heartbeat_tasks[websocket] = task

    async def disconnect(self, websocket: WebSocket, unit_name: str = "all") -> None:
        """Remove disconnected client and cancel its heartbeat task."""
        task = self._heartbeat_tasks.pop(websocket, None)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        async with self._lock:
            if unit_name in self.active_connections:
                self.active_connections[unit_name].discard(websocket)
                if not self.active_connections[unit_name]:
                    del self.active_connections[unit_name]
        logger.info(f"WebSocket disconnected for unit {unit_name}")

    async def _heartbeat(self, websocket: WebSocket) -> None:
        """Send periodic ping to client to keep connection alive."""
        try:
            while True:
                await asyncio.sleep(self._heartbeat_interval)
                await websocket.send_json({"type": "ping"})
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Heartbeat error for WebSocket: {e}")

    async def get_stats(self) -> Dict[str, int]:
        """
        Return statistics about active WebSocket connections.
        Thread-safe using the same asyncio lock as connect/disconnect.
        """
        async with self._lock:
            total = sum(len(conns) for conns in self.active_connections.values())
            per_unit = {unit: len(conns) for unit, conns in self.active_connections.items()}
        return {"total_connections": total, "per_unit": per_unit}

    async def broadcast_risk_update(self, unit_name: str, risk_data: dict) -> None:
        """Broadcast risk data to all clients subscribed to this unit or 'all'."""
        async with self._lock:
            if unit_name in self.active_connections:
                for connection in self.active_connections[unit_name]:
                    try:
                        await connection.send_json(risk_data)
                    except Exception as e:
                        logger.error(f"Failed to send to {unit_name}: {e}")
            if "all" in self.active_connections:
                for connection in self.active_connections["all"]:
                    try:
                        await connection.send_json(risk_data)
                    except Exception as e:
                        logger.error(f"Failed to send to global: {e}")

    async def broadcast_to_all(self, data: dict) -> None:
        """Broadcast to all connected clients regardless of unit."""
        async with self._lock:
            for unit_connections in self.active_connections.values():
                for connection in unit_connections:
                    try:
                        await connection.send_json(data)
                    except Exception:
                        pass


manager = ConnectionManager()


async def periodic_risk_broadcast():
    """
    Background task that reads latest risks from database and pushes via WebSocket.
    Runs every 2 seconds.
    """
    while True:
        try:
            db = SessionLocal()
            try:
                latest_per_unit = RiskAssessmentRepo.get_latest_per_unit(db)
                for item in latest_per_unit:
                    await manager.broadcast_risk_update(
                        unit_name=item["unit_name"],
                        risk_data={
                            "type": "risk_update",
                            "unit_name": item["unit_name"],
                            "risk_score": item["risk_score"],
                            "status": item["status"],
                            "timestamp": item["timestamp"].isoformat() if item["timestamp"] else None
                        }
                    )
                from backend.db.crud import get_dashboard_stats
                stats = get_dashboard_stats(db)
                await manager.broadcast_to_all({
                    "type": "stats",
                    "data": stats
                })
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Periodic broadcast error: {e}")
        await asyncio.sleep(2)