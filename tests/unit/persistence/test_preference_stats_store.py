"""Tests for preference-stat aggregation queries."""

from st_name_ranking.persistence.connection import get_connection
from st_name_ranking.persistence.preference_stats_store import (
    get_preference_stats_by_gender,
    get_preference_stats_by_origin,
    get_preference_stats_by_phonetic,
)
from st_name_ranking.types import PreferenceStats


def _insert_name(
    name: str,
    *,
    gender: str | None,
    origin_region: str | None,
    phonetic_primary: str | None,
) -> int:
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO names (name, gender, origin_region, phonetic_primary)
            VALUES (?, ?, ?, ?)
            """,
            (name, gender, origin_region, phonetic_primary),
        )
        return int(cursor.lastrowid)


def _record_comparison(name_a_id: int, name_b_id: int, preference: int) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO comparisons (name_a_id, name_b_id, preference)
            VALUES (?, ?, ?)
            """,
            (name_a_id, name_b_id, preference),
        )


def test_preference_stats_are_empty_without_comparisons(initialized_db):
    assert get_preference_stats_by_gender() == {}
    assert get_preference_stats_by_origin() == {}
    assert get_preference_stats_by_phonetic() == {}


def test_preference_stats_group_comparison_outcomes(initialized_db):
    anna_id = _insert_name(
        "Anna",
        gender="Female",
        origin_region="Nordic",
        phonetic_primary="AN",
    )
    bo_id = _insert_name(
        "Bo",
        gender="Male",
        origin_region=None,
        phonetic_primary="P",
    )
    charlie_id = _insert_name(
        "Charlie",
        gender=None,
        origin_region="European",
        phonetic_primary="",
    )

    _record_comparison(anna_id, bo_id, -1)
    _record_comparison(anna_id, charlie_id, 1)
    _record_comparison(bo_id, charlie_id, 0)

    assert get_preference_stats_by_gender() == {
        "Female": PreferenceStats(wins=1, losses=1, draws=0, total=2),
        "Male": PreferenceStats(wins=0, losses=1, draws=1, total=2),
        "Unknown": PreferenceStats(wins=1, losses=0, draws=1, total=2),
    }
    assert get_preference_stats_by_origin() == {
        "European": PreferenceStats(wins=1, losses=0, draws=1, total=2),
        "International": PreferenceStats(wins=0, losses=1, draws=1, total=2),
        "Nordic": PreferenceStats(wins=1, losses=1, draws=0, total=2),
    }
    assert get_preference_stats_by_phonetic() == {
        "AN": PreferenceStats(wins=1, losses=1, draws=0, total=2),
        "P": PreferenceStats(wins=0, losses=1, draws=1, total=2),
        "Unknown": PreferenceStats(wins=1, losses=0, draws=1, total=2),
    }
