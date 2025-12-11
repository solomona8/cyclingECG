#!/usr/bin/env python
"""
Standalone migration script for manual execution.

Usage:
    python run_migration.py

This can be run on Render or any environment to update the database schema.
"""
from app.migrate_db import migrate_database

if __name__ == "__main__":
    print("=" * 60)
    print("ECG Database Migration Tool")
    print("=" * 60)
    migrate_database()
    print("=" * 60)
