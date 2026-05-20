"""Preference analytics derived from recorded comparisons."""

from collections.abc import Iterable

from st_name_ranking.persistence.connection import get_connection
from st_name_ranking.types import PreferenceStats

_OUTCOME_KEY_MAP = {"win": "wins", "loss": "losses", "draw": "draws"}


def _build_preference_stats(rows: Iterable[tuple[str, str, int]]) -> dict[str, PreferenceStats]:
    result: dict[str, dict[str, int]] = {}
    for group, outcome, count in rows:
        if group not in result:
            result[group] = {"wins": 0, "losses": 0, "draws": 0, "total": 0}
        result[group][_OUTCOME_KEY_MAP[outcome]] = count
        result[group]["total"] += count

    return {
        group: PreferenceStats(
            wins=data["wins"],
            losses=data["losses"],
            draws=data["draws"],
            total=data["total"],
        )
        for group, data in result.items()
    }


def get_preference_stats_by_gender() -> dict[str, PreferenceStats]:
    """Get preference statistics grouped by gender."""
    with get_connection() as conn:
        cursor = conn.execute("""
            WITH name_outcomes AS (
                SELECT
                    name_a_id as name_id,
                    CASE
                        WHEN preference = -1 THEN 'win'
                        WHEN preference = 1 THEN 'loss'
                        ELSE 'draw'
                    END as outcome
                FROM comparisons
                WHERE preference IN (-1, 0, 1)
                UNION ALL
                SELECT
                    name_b_id as name_id,
                    CASE
                        WHEN preference = 1 THEN 'win'
                        WHEN preference = -1 THEN 'loss'
                        ELSE 'draw'
                    END as outcome
                FROM comparisons
                WHERE preference IN (-1, 0, 1)
            )
            SELECT
                COALESCE(n.gender, 'Unknown') as gender,
                no.outcome,
                COUNT(*) as count
            FROM name_outcomes no
            JOIN names n ON no.name_id = n.id
            GROUP BY n.gender, no.outcome
            ORDER BY gender, outcome
        """)
        return _build_preference_stats(cursor.fetchall())


def get_preference_stats_by_origin() -> dict[str, PreferenceStats]:
    """Get preference statistics grouped by origin region."""
    with get_connection() as conn:
        cursor = conn.execute("""
            WITH name_outcomes AS (
                SELECT
                    name_a_id as name_id,
                    CASE
                        WHEN preference = -1 THEN 'win'
                        WHEN preference = 1 THEN 'loss'
                        ELSE 'draw'
                    END as outcome
                FROM comparisons
                WHERE preference IN (-1, 0, 1)
                UNION ALL
                SELECT
                    name_b_id as name_id,
                    CASE
                        WHEN preference = 1 THEN 'win'
                        WHEN preference = -1 THEN 'loss'
                        ELSE 'draw'
                    END as outcome
                FROM comparisons
                WHERE preference IN (-1, 0, 1)
            )
            SELECT
                CASE
                    WHEN n.origin_region IS NULL THEN 'International'
                    ELSE n.origin_region
                END as region,
                no.outcome,
                COUNT(*) as count
            FROM name_outcomes no
            JOIN names n ON no.name_id = n.id
            GROUP BY region, no.outcome
            ORDER BY region, outcome
        """)
        return _build_preference_stats(cursor.fetchall())


def get_preference_stats_by_phonetic() -> dict[str, PreferenceStats]:
    """Get preference statistics grouped by phonetic primary code."""
    with get_connection() as conn:
        cursor = conn.execute("""
            WITH name_outcomes AS (
                SELECT
                    name_a_id as name_id,
                    CASE
                        WHEN preference = -1 THEN 'win'
                        WHEN preference = 1 THEN 'loss'
                        ELSE 'draw'
                    END as outcome
                FROM comparisons
                WHERE preference IN (-1, 0, 1)
                UNION ALL
                SELECT
                    name_b_id as name_id,
                    CASE
                        WHEN preference = 1 THEN 'win'
                        WHEN preference = -1 THEN 'loss'
                        ELSE 'draw'
                    END as outcome
                FROM comparisons
                WHERE preference IN (-1, 0, 1)
            )
            SELECT
                CASE
                    WHEN n.phonetic_primary IS NULL OR n.phonetic_primary = '' THEN 'Unknown'
                    ELSE n.phonetic_primary
                END as phonetic_code,
                no.outcome,
                COUNT(*) as count
            FROM name_outcomes no
            JOIN names n ON no.name_id = n.id
            GROUP BY phonetic_code, no.outcome
            ORDER BY phonetic_code, outcome
        """)
        return _build_preference_stats(cursor.fetchall())
