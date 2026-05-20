"""User setting persistence."""

from st_name_ranking.persistence.connection import get_connection


def save_user_setting(key: str, value: str) -> None:
    """Save a user setting."""
    with get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO user_settings (key, value) VALUES (?, ?)",
            (key, value),
        )


def load_user_setting(key: str, default: str = "") -> str:
    """Load a user setting."""
    with get_connection() as conn:
        cursor = conn.execute(
            "SELECT value FROM user_settings WHERE key = ?",
            (key,),
        )
        row = cursor.fetchone()
        return row[0] if row else default
