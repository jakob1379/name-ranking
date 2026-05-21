"""Low-level SQLite connection state for persistence modules."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

DB_PATH = Path("data/names.db")
_INIT_STATE = {"db_initialized": False, "db_path": None}

INITIAL_SCORE = 1500.0
MAX_SQL_PARAMS = 500
MILLISECONDS_PER_SECOND = 1000


def reset_database_init_state() -> None:
    """Reset cached database-initialization state."""
    _INIT_STATE["db_initialized"] = False
    _INIT_STATE["db_path"] = None


def get_db_path() -> Path:
    """Return the active SQLite database path."""
    return DB_PATH


def set_db_path(path: str | Path) -> None:
    """Set the active SQLite database path and reset initialization state."""
    global DB_PATH  # noqa: PLW0603
    DB_PATH = Path(path)
    reset_database_init_state()


@contextmanager
def get_connection(timeout: float = 30.0) -> Iterator[sqlite3.Connection]:
    """Context manager for database connections with atomic transactions."""
    timeout_ms = int(timeout * MILLISECONDS_PER_SECOND)
    conn = sqlite3.connect(get_db_path(), timeout=timeout)
    conn.row_factory = sqlite3.Row

    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"PRAGMA busy_timeout={timeout_ms}")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
