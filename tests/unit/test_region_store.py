"""Tests for region mapping seed data."""

import sqlite3

from st_name_ranking.persistence.region_store import insert_default_region_mapping


def _region_mapping_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE region_mapping (
            nationality TEXT PRIMARY KEY,
            region TEXT NOT NULL
        )
        """,
    )
    return conn


def test_insert_default_region_mapping_seeds_expected_regions():
    with _region_mapping_connection() as conn:
        insert_default_region_mapping(conn)

        rows = dict(conn.execute("SELECT nationality, region FROM region_mapping"))

    assert rows["Denmark"] == "Nordic"
    assert rows["Germany"] == "European"
    assert rows["Japan"] == "Asian"
    assert rows["Nigeria"] == "African"
    assert rows["Canada"] == "American"
    assert rows["New Zealand"] == "Oceanian"


def test_insert_default_region_mapping_is_idempotent():
    with _region_mapping_connection() as conn:
        insert_default_region_mapping(conn)
        first_count = conn.execute("SELECT COUNT(*) FROM region_mapping").fetchone()[0]

        insert_default_region_mapping(conn)
        second_count = conn.execute("SELECT COUNT(*) FROM region_mapping").fetchone()[0]

    assert second_count == first_count


def test_insert_default_region_mapping_keeps_first_region_for_duplicate_country():
    with _region_mapping_connection() as conn:
        insert_default_region_mapping(conn)

        pakistan_region = conn.execute(
            "SELECT region FROM region_mapping WHERE nationality = ?",
            ("Pakistan",),
        ).fetchone()[0]

    assert pakistan_region == "Middle Eastern"
