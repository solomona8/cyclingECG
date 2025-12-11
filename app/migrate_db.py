"""
Database migration script to add missing columns to existing tables.
Run this on startup or manually to ensure schema is up-to-date.
"""
import sqlite3
import os
from typing import List, Tuple


def get_db_path() -> str:
    """Get the database file path from DATABASE_URL or use default"""
    database_url = os.environ.get("DATABASE_URL", "sqlite:///./ecg_data.db")
    if database_url.startswith("sqlite:///"):
        return database_url.replace("sqlite:///", "")
    return "./ecg_data.db"


def column_exists(cursor: sqlite3.Cursor, table: str, column: str) -> bool:
    """Check if a column exists in a table"""
    cursor.execute(f"PRAGMA table_info({table})")
    columns = [row[1] for row in cursor.fetchall()]
    return column in columns


def migrate_database():
    """
    Add missing columns to the ecg_analyses table if they don't exist.
    This handles migrations from the old schema to the new one with interval metrics.
    """
    db_path = get_db_path()

    # If database doesn't exist yet, no need to migrate
    if not os.path.exists(db_path):
        print(f"Database {db_path} doesn't exist yet, no migration needed")
        return

    print(f"Checking database schema at {db_path}...")

    # Define columns to add (column_name, column_type, nullable)
    new_columns: List[Tuple[str, str, bool]] = [
        ("p_wave_duration", "REAL", True),
        ("pr_interval", "REAL", True),
        ("pr_segment", "REAL", True),
        ("qrs_duration", "REAL", False),
        ("st_segment", "REAL", True),
        ("t_wave_duration", "REAL", True),
        ("qt_interval", "REAL", False),
        ("qtc", "REAL", False),
        ("artifact_burden", "REAL", True),
        ("ectopy_burden", "REAL", True),
    ]

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ecg_analyses'")
        if not cursor.fetchone():
            print("Table ecg_analyses doesn't exist yet, no migration needed")
            return

        # Add missing columns
        columns_added = []
        for column_name, column_type, nullable in new_columns:
            if not column_exists(cursor, "ecg_analyses", column_name):
                null_clause = "" if nullable else "NOT NULL DEFAULT 0"
                try:
                    cursor.execute(f"ALTER TABLE ecg_analyses ADD COLUMN {column_name} {column_type} {null_clause}")
                    columns_added.append(column_name)
                    print(f"✓ Added column: {column_name}")
                except sqlite3.Error as e:
                    print(f"✗ Error adding column {column_name}: {e}")
            else:
                print(f"• Column {column_name} already exists")

        conn.commit()

        if columns_added:
            print(f"\nMigration complete! Added {len(columns_added)} columns.")
        else:
            print("\nNo migration needed - schema is up to date.")

    except Exception as e:
        print(f"Migration error: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    migrate_database()
