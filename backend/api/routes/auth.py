"""
Authentication endpoints: register, login, logout, and admin user management.
Uses httpOnly cookie for JWT storage.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Response
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import List

from backend.api.dependencies import get_db, get_current_user
from backend.db import crud, schemas
from backend.auth.auth import (
    verify_password, create_access_token, get_password_hash,
    create_access_token_cookie_response
)
from backend.exceptions import UnauthorizedException, ForbiddenException

router = APIRouter()


@router.post("/auth/register", response_model=schemas.UserResponse)
def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    """Register a new user."""
    db_user = crud.get_user_by_username(db, user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already exists")
    hashed_pw = get_password_hash(user.password)
    new_user = crud.create_user(db, user, hashed_pw)
    return new_user


@router.post("/auth/login")
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """
    Authenticate user and set JWT token as httpOnly cookie.
    Returns JSON with access_token for compatibility, but token is also in cookie.
    """
    user = crud.authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise UnauthorizedException("Incorrect username or password")
    return create_access_token_cookie_response(
        data={"sub": user.username, "role": user.role}
    )


@router.post("/auth/logout")
def logout(response: Response):
    """Clear the authentication cookie."""
    response.delete_cookie("access_token")
    return {"detail": "Logged out successfully"}


@router.post("/auth/login-json", response_model=schemas.Token)
def login_json(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """
    Alternative login endpoint that returns token in JSON (no cookie).
    Useful for API clients that cannot handle cookies.
    """
    user = crud.authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise UnauthorizedException("Incorrect username or password")
    access_token = create_access_token(data={"sub": user.username, "role": user.role})
    return {"access_token": access_token, "token_type": "bearer"}


# ---------- Admin user management ----------
@router.get("/auth/users", response_model=List[schemas.UserResponse])
def list_users(current_user=Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "admin":
        raise ForbiddenException("Only admin can list users")
    return crud.get_all_users(db)


@router.delete("/auth/users/{user_id}")
def delete_user(
    user_id: str,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role != "admin":
        raise ForbiddenException("Only admin can delete users")
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    success = crud.delete_user(db, user_id)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    return {"detail": "User deleted"}


@router.put("/auth/users/{user_id}/role")
def change_user_role(
    user_id: str,
    role: str,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role != "admin":
        raise ForbiddenException("Only admin can change roles")
    if role not in ("operator", "engineer", "admin"):
        raise HTTPException(status_code=400, detail="Invalid role")
    updated = crud.update_user_role(db, user_id, role)
    if not updated:
        raise HTTPException(status_code=404, detail="User not found")
    return {"detail": f"User role changed to {role}"}