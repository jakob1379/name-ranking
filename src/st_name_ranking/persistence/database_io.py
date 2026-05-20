"""SQLite database import/export operations."""

import datetime as dt
import logging
import shutil
import sqlite3
import tempfile

from st_name_ranking.persistence.connection import get_db_path, reset_database_init_state

logger = logging.getLogger(__name__)


def export_database() -> bytes:
    """Export the current SQLite database file as bytes."""
    db_path = get_db_path()
    if not db_path.exists():
        _msg = f"Database file not found at {db_path}"
        raise FileNotFoundError(_msg)

    try:
        with db_path.open("rb") as f:
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

    db_path = get_db_path()
    if backup and db_path.exists():
        backup_path = db_path.with_suffix(
            f".db.backup.{dt.datetime.now(dt.UTC).strftime('%Y%m%d_%H%M%S')}",
        )
        shutil.copy2(db_path, backup_path)
        logger.info("Created backup of current database at %s", backup_path)

    try:
        with db_path.open("wb") as f:
            f.write(file_bytes)
    except OSError as e:
        _msg = f"Failed to write database file: {e}"
        raise OSError(_msg) from e

    reset_database_init_state()
    logger.info("Database imported successfully")
