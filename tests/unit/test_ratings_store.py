"""Tests for rating and comparison persistence."""

import pytest

from st_name_ranking.persistence.connection import INITIAL_SCORE, get_connection
from st_name_ranking.persistence.ratings_store import (
    get_comparison_count,
    get_ratings,
    get_total_comparisons,
    initialize_ratings,
    record_comparison,
    update_rating,
    update_ratings_batch,
    update_ratings_batch_values,
)


def _insert_name(name: str) -> int:
    with get_connection() as conn:
        cursor = conn.execute("INSERT INTO names (name) VALUES (?)", (name,))
        return int(cursor.lastrowid)


def _rating_row(name: str) -> tuple[float, int]:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT r.rating, r.matches
            FROM ratings r
            JOIN names n ON n.id = r.name_id
            WHERE n.name = ?
            """,
            (name,),
        ).fetchone()
        assert row is not None
        return float(row[0]), int(row[1])


def test_rating_updates_read_values_and_report_missing_names(initialized_db):
    _insert_name("Anna")
    _insert_name("Bo")

    assert get_ratings() == {}
    assert update_rating("Anna", 1510.5) == []
    assert update_rating("Missing", 1400) == ["Missing"]

    assert get_ratings() == {"Anna": 1510.5}
    assert _rating_row("Anna") == (1510.5, 0)

    assert update_ratings_batch({"Anna": 1520, "Bo": 1490, "Missing": 1300}) == [
        "Missing",
    ]
    assert get_ratings() == {"Anna": 1520, "Bo": 1490}
    assert _rating_row("Anna") == (1520.0, 1)
    assert _rating_row("Bo") == (1490.0, 1)

    assert update_ratings_batch_values({"Anna": 1530, "Missing": 1300}) == [
        "Missing",
    ]
    assert _rating_row("Anna") == (1530.0, 1)
    assert update_ratings_batch({}) == []
    assert update_ratings_batch_values({}) == []


def test_comparison_recording_counts_names_and_validates_input(initialized_db):
    _insert_name("Anna")
    _insert_name("Bo")
    _insert_name("Charlie")

    record_comparison("Anna", "Bo", -1)
    record_comparison("Anna", "Bo", -1)
    record_comparison("Anna", "Charlie", 0)
    record_comparison("Bo", "Charlie", 2)

    assert get_total_comparisons() == 3
    assert get_comparison_count("Anna") == 2
    assert get_comparison_count("Bo") == 2
    assert get_comparison_count("Missing") == 0

    with pytest.raises(ValueError, match="preference must be"):
        record_comparison("Anna", "Bo", 99)
    with pytest.raises(ValueError, match="Name not found: Missing"):
        record_comparison("Missing", "Bo", 1)


def test_initialize_ratings_uses_initial_score():
    assert initialize_ratings(["Anna", "Bo"]) == {
        "Anna": INITIAL_SCORE,
        "Bo": INITIAL_SCORE,
    }
