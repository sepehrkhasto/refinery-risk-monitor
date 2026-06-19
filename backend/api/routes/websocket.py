"""
WebSocket endpoint for real-time risk data with JWT authentication.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, status
from backend.websocket.manager import manager
from backend.api.dependencies import get_current_user_ws
from backend.logger import logger

router = APIRouter(tags=["WebSocket"])


@router.websocket("/ws/{unit_name}")
async def websocket_endpoint(
    websocket: WebSocket,
    unit_name: str = "all"
):
    """
    WebSocket connection for real-time risk updates.
    Use unit_name = "all" to receive updates for all units.
    Authentication required via token query parameter.
    """
    # Authenticate user first
    try:
        user = await get_current_user_ws(websocket)
    except Exception as e:
        # get_current_user_ws already closes the connection on error
        logger.warning(f"WebSocket authentication failed: {e}")
        return

    await manager.connect(websocket, unit_name)
    try:
        # Send initial confirmation with user info
        await websocket.send_json({
            "type": "connected",
            "unit_name": unit_name,
            "user": user.username,
            "message": "Connected to risk monitor"
        })
        # Keep connection alive
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        await manager.disconnect(websocket, unit_name)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await manager.disconnect(websocket, unit_name)