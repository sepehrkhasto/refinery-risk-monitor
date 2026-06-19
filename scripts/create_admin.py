#!/usr/bin/env python
"""
Admin user creation script (similar to Django's createsuperuser).
"""

import sys
import os
from getpass import getpass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.db.database import SessionLocal
from backend.db.crud import get_user_by_username, create_user
from backend.db.schemas import UserCreate
from backend.auth.auth import get_password_hash


def create_admin():
    db = SessionLocal()
    try:
        print("=== Create Admin User ===")

        while True:
            username = input("Username: ").strip()
            if not username:
                print("Error: Username cannot be empty.")
                continue
            if get_user_by_username(db, username):
                print(f"Error: User '{username}' already exists.")
                continue
            break

        while True:
            password = getpass("Password (min 6 characters): ")
            if len(password) < 6:
                print("Password must be at least 6 characters.")
                continue
            password2 = getpass("Repeat password: ")
            if password != password2:
                print("Passwords do not match.")
                continue
            break

        role_input = input("Role [admin/engineer/operator] (default: admin): ").strip().lower()
        if role_input not in ('admin', 'engineer', 'operator'):
            role_input = 'admin'

        full_name = input("Full name (optional): ").strip() or None

        hashed_pw = get_password_hash(password)
        user_data = UserCreate(
            username=username,
            password=password,
            full_name=full_name,
            role=role_input
        )
        new_user = create_user(db, user_data, hashed_pw)
        print(f"\n✅ User '{new_user.username}' with role '{new_user.role}' created successfully.")

    finally:
        db.close()


if __name__ == "__main__":
    create_admin()