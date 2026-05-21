"""Rating and comparison persistence."""

import logging

from st_name_ranking.persistence.connection import INITIAL_SCORE, get_connection

logger = logging.getLogger(__name__)


def get_ratings() -> dict[str, float]:
    """Get all ratings as name -> rating dictionary."""
    with get_connection() as conn:
        cursor = conn.execute("""
            SELECT n.name, r.rating
            FROM names n
            JOIN ratings r ON n.id = r.name_id
        """)
        return {row[0]: row[1] for row in cursor.fetchall()}


def update_rating(name: str, rating: float) -> list[str]:
    """Update one rating and return the skipped name when it is missing."""
    with get_connection() as conn:
        name_id = conn.execute(
            "SELECT id FROM names WHERE name = ?",
            (name,),
        ).fetchone()
        if not name_id:
            logger.warning("Name not found in database: %s", name)
            return [name]

        name_id = name_id[0]

        conn.execute(
            """
            INSERT OR REPLACE INTO ratings
            (name_id, rating, matches, last_updated)
            VALUES (
                ?,
                ?,
                COALESCE((
                    SELECT matches FROM ratings
                    WHERE name_id = ?
                ), 0),
                CURRENT_TIMESTAMP
            )
        """,
            (name_id, rating, name_id),
        )
    return []


def update_ratings_batch(ratings_dict: dict[str, float]) -> list[str]:
    """Update multiple ratings in one transaction, incrementing match counts."""
    return _update_ratings_batch(ratings_dict, increment_matches=True)


def update_ratings_batch_values(ratings_dict: dict[str, float]) -> list[str]:
    """Update multiple ratings without incrementing match counts."""
    return _update_ratings_batch(ratings_dict, increment_matches=False)


def _update_ratings_batch(
    ratings_dict: dict[str, float],
    *,
    increment_matches: bool,
) -> list[str]:
    if not ratings_dict:
        return []

    missing_names: list[str] = []
    with get_connection() as conn:
        for name, rating in ratings_dict.items():
            name_id = conn.execute(
                "SELECT id FROM names WHERE name = ?",
                (name,),
            ).fetchone()
            if not name_id:
                logger.warning("Name not found in database: %s", name)
                missing_names.append(name)
                continue

            name_id = name_id[0]
            existing_rating = conn.execute(
                "SELECT matches FROM ratings WHERE name_id = ?",
                (name_id,),
            ).fetchone()
            matches = existing_rating[0] + int(increment_matches) if existing_rating else int(increment_matches)

            conn.execute(
                """
                INSERT OR REPLACE INTO ratings
                (name_id, rating, matches, last_updated)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """,
                (name_id, rating, matches),
            )

    return missing_names


def record_comparison(name_a: str, name_b: str, preference: int) -> None:
    """Record a pairwise comparison in the database."""
    if preference not in (-1, 0, 1, 2):
        _msg = "preference must be -1, 0, 1, or 2"
        raise ValueError(_msg)

    with get_connection() as conn:
        cursor = conn.execute(
            "SELECT id FROM names WHERE name = ?",
            (name_a,),
        )
        row = cursor.fetchone()
        if not row:
            _msg = f"Name not found: {name_a}"
            raise ValueError(_msg)
        name_a_id = row[0]

        cursor = conn.execute(
            "SELECT id FROM names WHERE name = ?",
            (name_b,),
        )
        row = cursor.fetchone()
        if not row:
            _msg = f"Name not found: {name_b}"
            raise ValueError(_msg)
        name_b_id = row[0]

        conn.execute(
            """
            INSERT OR IGNORE INTO comparisons
            (name_a_id, name_b_id, preference, created_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (name_a_id, name_b_id, preference),
        )


def get_total_comparisons() -> int:
    """Get total number of recorded pairwise comparisons."""
    with get_connection() as conn:
        return conn.execute("SELECT COUNT(*) FROM comparisons").fetchone()[0]


def get_comparison_count(name: str) -> int:
    """Get number of comparisons involving a name."""
    with get_connection() as conn:
        cursor = conn.execute("SELECT id FROM names WHERE name = ?", (name,))
        row = cursor.fetchone()
        if not row:
            return 0
        name_id = row[0]

        cursor = conn.execute(
            """
            SELECT COUNT(*) FROM comparisons
            WHERE name_a_id = ? OR name_b_id = ?
            """,
            (name_id, name_id),
        )
        return cursor.fetchone()[0]


def initialize_ratings(names: list[str]) -> dict[str, float]:
    """Initialize ratings for a list of names with the default score."""
    return dict.fromkeys(names, INITIAL_SCORE)
