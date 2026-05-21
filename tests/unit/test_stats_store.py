"""Tests for database statistics queries."""

from st_name_ranking.persistence.connection import get_connection
from st_name_ranking.persistence.stats_store import get_stats


def test_stats_store_counts_names_ratings_and_origins(initialized_db):
    with get_connection() as conn:
        conn.execute("INSERT INTO names (name, gender, origin_region) VALUES ('Anna', 'Female', 'Nordic')")
        conn.execute("INSERT INTO names (name, gender, origin_region) VALUES ('Bo', 'Male', NULL)")
        conn.execute("INSERT INTO names (name, gender, origin_region) VALUES ('Carla', 'Female', 'European')")
        anna_id = conn.execute("SELECT id FROM names WHERE name = 'Anna'").fetchone()[0]
        bo_id = conn.execute("SELECT id FROM names WHERE name = 'Bo'").fetchone()[0]
        conn.execute("INSERT INTO ratings (name_id, rating) VALUES (?, 1510)", (anna_id,))
        conn.execute("INSERT INTO ratings (name_id, rating) VALUES (?, 1490)", (bo_id,))

    stats = get_stats()

    assert stats.total_names == 3
    assert stats.classified_names == 2
    assert stats.unclassified_names == 1
    assert stats.rated_names == 2
    assert stats.origin_distribution == {
        "European": 1,
        "International": 1,
        "Nordic": 1,
    }
