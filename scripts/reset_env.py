#!/usr/bin/env python
"""
Full test environment reset script.
Deletes SQLite database files and all trained ML models.
"""

import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def remove_db_files() -> None:
    """Remove SQLite database and associated WAL/SHM files."""
    patterns = ['refinery_risk.db', 'refinery_risk.db-shm', 'refinery_risk.db-wal']
    for pattern in patterns:
        filepath = os.path.join(PROJECT_ROOT, pattern)
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"✓ Deleted: {filepath}")


def remove_model_files() -> None:
    """Delete all contents of backend/models directory."""
    models_dir = os.path.join(PROJECT_ROOT, 'backend', 'models')
    if not os.path.exists(models_dir):
        print(f"⚠️ Models directory not found: {models_dir}")
        return
    for root, dirs, files in os.walk(models_dir, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            os.remove(file_path)
            print(f"✓ Deleted: {file_path}")
        for name in dirs:
            dir_path = os.path.join(root, name)
            os.rmdir(dir_path)
            print(f"✓ Removed directory: {dir_path}")


def main() -> None:
    print("Resetting test environment...\n")
    remove_db_files()
    remove_model_files()
    print("\n✅ Environment reset successfully.")
    print("Next steps:")
    print("  alembic upgrade head")
    print("  python scripts/train_advanced_models.py")


if __name__ == "__main__":
    main()