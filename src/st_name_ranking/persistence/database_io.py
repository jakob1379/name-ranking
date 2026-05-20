"""SQLite database import/export operations."""

import datetime as dt
import logging
import shutil
import sqlite3
import tempfile

from st_name_ranking.persistence import database

logger = logging.getLogger(__name__)


def export_database() -> bytes:
    """Export the current SQLite database file as bytes."""
    if not database.DB_PATH.exists():
        _msg = f"Database file not found at {database.DB_PATH}"
        raise FileNotFoundError(_msg)

    try:
        with database.DB_PATH.open("rb") as f:
            return f.read()
    except OSError as e:
        _msg = f"Failed to read database file: {e}"
        raise OSError(_msg) from e


def import_database(file_bytes: bytes, *, backup: bool = True) -> None:
    """Replace the current SQLite database with uploaded bytes."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            conn = sqlite3.connect(tmp.name)
            try:
                conn.execute("SELECT 1 FROM sqlite_master LIMIT 1")
            finally:
                conn.close()
    except sqlite3.Error as e:
        _msg = f"Uploaded file is not a valid SQLite database: {e}"
        raise ValueError(_msg) from e

    if backup and database.DB_PATH.exists():
        backup_path = database.DB_PATH.with_suffix(
            f".db.backup.{dt.datetime.now(dt.UTC).strftime('%Y%m%d_%H%M%S')}",
        )
        shutil.copy2(database.DB_PATH, backup_path)
        logger.info("Created backup of current database at %s", backup_path)

    try:
        with database.DB_PATH.open("wb") as f:
            f.write(file_bytes)
    except OSError as e:
        _msg = f"Failed to write database file: {e}"
        raise OSError(_msg) from e

    database.reset_database_init_state()
    logger.info("Database imported successfully")
