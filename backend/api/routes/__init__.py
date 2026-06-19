"""
API routes package initializer.
Exports all route modules including WebSocket and Health.
"""

from . import dashboard
from . import predictions
from . import analysis
from . import root_cause
from . import monitoring
from . import auth
from . import reports
from . import simulate
from . import websocket
from . import health

__all__ = [
    "dashboard",
    "predictions",
    "analysis",
    "root_cause",
    "monitoring",
    "auth",
    "reports",
    "simulate",
    "websocket",
    "health"
]