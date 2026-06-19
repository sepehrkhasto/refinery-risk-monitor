"""
JWT-based authentication module with SHA-256 + Salt password hashing.
Supports both Authorization header and httpOnly cookie for token delivery.
"""

import hashlib
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
from backend.config import settings


def get_password_hash(password: str) -> str:
    """Generate secure hash with SHA-256 and random salt."""
    salt = os.urandom(32).hex()
    h = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}${h}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against stored salt+hash."""
    try:
        salt, original_hash = hashed_password.split('$', 1)
        new_hash = hashlib.sha256((salt + plain_password).encode()).hexdigest()
        return new_hash == original_hash
    except (ValueError, AttributeError):
        return False


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT token with username and role."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_access_token(token: str) -> Optional[dict]:
    """Decode JWT token and extract payload."""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except JWTError:
        return None


def create_access_token_cookie_response(data: dict, expires_delta: Optional[timedelta] = None):
    """
    Create a response object with httpOnly cookie containing the access token.
    This is a helper for login endpoints that want to set the token in a cookie.
    """
    from fastapi.responses import JSONResponse
    token = create_access_token(data, expires_delta)
    response = JSONResponse(content={"access_token": token, "token_type": "bearer"})
    expires = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        secure=settings.ENVIRONMENT == "production",  # Only secure in production
        samesite="lax",
        expires=int(expires.timestamp())
    )
    return response