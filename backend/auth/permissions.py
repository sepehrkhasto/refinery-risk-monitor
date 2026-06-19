"""
Role-Based Access Control (RBAC) for API endpoints.
Defines permission checkers based on user roles.
"""

from fastapi import Depends, HTTPException, status
from backend.api.dependencies import get_current_user
from backend.db.db_models import User


class RoleChecker:
    """Dependency that checks if current user has an allowed role."""
    def __init__(self, allowed_roles: list[str]):
        self.allowed_roles = allowed_roles

    def __call__(self, user: User = Depends(get_current_user)) -> User:
        if user.role not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to access this resource"
            )
        return user


allow_admin = RoleChecker(["admin"])
allow_engineer = RoleChecker(["engineer", "admin"])
allow_operator = RoleChecker(["operator", "engineer", "admin"])