"""Tests for focused persistence store helpers."""

from st_name_ranking.persistence import name_store, settings_store, stats_store
from st_name_ranking.persistence.database import get_connection


def _insert_name(
    name: str,
    *,
    gender: str | None = None,
    origin_region: str | None = None,
    origin_confidence: float | None = None,
    phonetic_primary: str | None = None,
    phonetic_secondary: str | None = None,
) -> int:
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO names (
                name,
                gender,
                origin_region,
                origin_confidence,
                phonetic_primary,
                phonetic_secondary
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                gender,
                origin_region,
                origin_confidence,
                phonetic_primary,
                phonetic_secondary,
            ),
        )
        return int(cursor.lastrowid)


def test_settings_store_round_trips_values(initialized_db):
    assert settings_store.load_user_setting("theme", default="system") == "system"

    settings_store.save_user_setting("theme", "dark")
    assert settings_store.load_user_setting("theme", default="system") == "dark"

    settings_store.save_user_setting("theme", "light")
    assert settings_store.load_user_setting("theme", default="system") == "light"


def test_stats_store_counts_names_ratings_and_origins(initialized_db):
    with get_connection() as conn:
        conn.execute("INSERT INTO names (name, gender, origin_region) VALUES ('Anna', 'Female', 'Nordic')")
        conn.execute("INSERT INTO names (name, gender, origin_region) VALUES ('Bo', 'Male', NULL)")
        conn.execute("INSERT INTO names (name, gender, origin_region) VALUES ('Carla', 'Female', 'European')")
        anna_id = conn.execute("SELECT id FROM names WHERE name = 'Anna'").fetchone()[0]
        bo_id = conn.execute("SELECT id FROM names WHERE name = 'Bo'").fetchone()[0]
        conn.execute("INSERT INTO ratings (name_id, rating) VALUES (?, 1510)", (anna_id,))
        conn.execute("INSERT INTO ratings (name_id, rating) VALUES (?, 1490)", (bo_id,))

    stats = stats_store.get_stats()

    assert stats.total_names == 3
    assert stats.classified_names == 2
    assert stats.unclassified_names == 1
    assert stats.rated_names == 2
    assert stats.origin_distribution == {
        "European": 1,
        "International": 1,
        "Nordic": 1,
    }


def test_name_store_updates_phonetics_and_origin_classification(initialized_db):
    anna_id = _insert_name("Anna")
    bo_id = _insert_name("Bo")

    updated = name_store.update_phonetic_codes(
        limit=1,
        compute_codes=lambda value: (f"{value}-primary", f"{value}-secondary"),
    )

    assert updated == 1
    assert name_store.get_phonetic_codes_batch(["Anna", "Bo", "Missing"]) == {
        "Anna": name_store.PhoneticCodes(primary="Anna-primary", secondary="Anna-secondary"),
        "Bo": name_store.PhoneticCodes(primary="", secondary=""),
    }

    name_store.update_name_origin(anna_id, "Nordic", 0.93)
    name_store.update_name_origin(bo_id, "International", 0.42)

    assert name_store.get_unclassified_names() == []
    assert name_store.get_names_with_origins(confidence_threshold=0.5) == {
        "Anna": ("Nordic", 0.93, "Anna-primary", "Anna-secondary"),
    }


def test_name_store_filters_gender_regions_and_batch_details(initialized_db):
    _insert_name(
        "Anna",
        gender="Female",
        origin_region="Nordic",
        phonetic_primary="AN",
        phonetic_secondary="",
    )
    _insert_name(
        "Bo",
        gender="Male",
        origin_region=None,
        phonetic_primary="P",
        phonetic_secondary="",
    )
    _insert_name(
        "Charlie",
        gender="Unisex",
        origin_region="European",
        phonetic_primary="XRL",
        phonetic_secondary="",
    )

    assert name_store.get_names_by_filters(gender="Female") == ["Anna"]
    assert name_store.get_names_by_filters(origins=["International"]) == ["Bo"]
    assert name_store.get_names_by_filters(origins=["European", "International"]) == [
        "Bo",
        "Charlie",
    ]

    assert name_store.get_names_by_gender() == {
        "All": ["Anna", "Bo", "Charlie"],
        "Female": ["Anna", "Charlie"],
        "Male": ["Bo", "Charlie"],
        "Unisex": ["Charlie"],
    }
    assert name_store.get_all_origin_regions() == [
        "European",
        "International",
        "Nordic",
    ]
    assert name_store.get_name_details_batch(["Charlie", "Missing", "Anna"]) == [
        name_store.NameDetails(gender="Unisex", origin_region="European"),
        name_store.NameDetails(gender=None, origin_region=None),
        name_store.NameDetails(gender="Female", origin_region="Nordic"),
    ]
