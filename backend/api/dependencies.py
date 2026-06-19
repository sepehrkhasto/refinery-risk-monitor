"""
FastAPI dependencies: database session, current user, and WebSocket auth.
Supports JWT from both Authorization header and httpOnly cookie.
"""

from typing import Optional
from fastapi import Depends, Request, WebSocket
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from backend.db.database import SessionLocal
from backend.auth.auth import decode_access_token
from backend.db.crud import get_user_by_username
from backend.db.db_models import User
from backend.exceptions import UnauthorizedException
from backend.logger import set_correlation_id, correlation_id_var

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)


def get_db() -> Session:
    """Create a database session and close it after request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_token_from_request(request: Request, token: Optional[str] = Depends(oauth2_scheme)) -> Optional[str]:
    """
    Extract token from either Authorization header or cookie.
    Priority: Authorization header > cookie > None.
    """
    if token:
        return token
    # Try to get from cookie
    cookie_token = request.cookies.get("access_token")
    if cookie_token:
        return cookie_token
    return None


def get_current_user(
    request: Request,
    token: Optional[str] = Depends(get_token_from_request),
    db: Session = Depends(get_db)
) -> User:
    """Get current user from JWT token (supports header or cookie)."""
    # Set correlation_id from request header if exists (for logging)
    cid = request.headers.get("X-Request-ID")
    if cid:
        set_correlation_id(cid)

    if token is None:
        raise UnauthorizedException("No token provided")
    payload = decode_access_token(token)
    if payload is None:
        raise UnauthorizedException("Invalid token")
    username: str = payload.get("sub")
    if username is None:
        raise UnauthorizedException("Invalid token payload")
    user = get_user_by_username(db, username)
    if user is None:
        raise UnauthorizedException("User not found")
    if not user.is_active:
        raise UnauthorizedException("User is inactive")
    return user


async def get_current_user_ws(websocket: WebSocket, token: str = None) -> User:
    """
    Get current user from WebSocket connection.
    Token can be passed as query parameter "token" or in cookie.
    """
    # Extract token from query string
    if token is None:
        token = websocket.query_params.get("token")
    if token is None:
        # try cookie
        cookie_header = websocket.headers.get("cookie", "")
        for part in cookie_header.split(";"):
            part = part.strip()
            if part.startswith("access_token="):
                token = part.split("=", 1)[1]
                break
    if token is None:
        await websocket.close(code=1008, reason="No token provided")
        raise UnauthorizedException("No token provided")

    payload = decode_access_token(token)
    if payload is None:
        await websocket.close(code=1008, reason="Invalid token")
        raise UnauthorizedException("Invalid token")

    username = payload.get("sub")
    if username is None:
        await websocket.close(code=1008, reason="Invalid token payload")
        raise UnauthorizedException("Invalid token payload")

    db = SessionLocal()
    try:
        user = get_user_by_username(db, username)
        if user is None or not user.is_active:
            await websocket.close(code=1008, reason="User not found or inactive")
            raise UnauthorizedException("User not found")
        # Set correlation_id for subsequent logs (optional, use a random one)
        import uuid
        set_correlation_id(str(uuid.uuid4()))
        return user
    finally:
        db.close()