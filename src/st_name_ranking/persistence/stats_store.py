"""Database statistics queries."""

from st_name_ranking.persistence.connection import get_connection
from st_name_ranking.types import DatabaseStats


def get_stats() -> DatabaseStats:
    """Get database statistics."""
    with get_connection() as conn:
        total_names = conn.execute("SELECT COUNT(*) FROM names").fetchone()[0]
        classified_names = conn.execute(
            "SELECT COUNT(*) FROM names WHERE origin_region IS NOT NULL",
        ).fetchone()[0]
        rated_names = conn.execute("SELECT COUNT(*) FROM ratings").fetchone()[0]

        origin_dist = {}
        cursor = conn.execute("""
            SELECT
                CASE
                    WHEN origin_region IS NULL THEN 'International'
                    ELSE origin_region
                END as region,
                COUNT(*) as count
            FROM names
            GROUP BY region
            ORDER BY count DESC
        """)
        for row in cursor.fetchall():
            origin_dist[row[0]] = row[1]

        return DatabaseStats(
            total_names=total_names,
            classified_names=classified_names,
            unclassified_names=total_names - classified_names,
            rated_names=rated_names,
            origin_distribution=origin_dist,
        )
