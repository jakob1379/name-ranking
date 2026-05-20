"""Name lookup, classification, and phonetic-code persistence."""

import logging
import sqlite3
from collections.abc import Callable

from metaphone import doublemetaphone

from st_name_ranking.persistence.connection import MAX_SQL_PARAMS, get_connection
from st_name_ranking.types import NameDetails, PhoneticCodes, UnclassifiedName

logger = logging.getLogger(__name__)


def compute_phonetic_codes(name: str) -> tuple[str, str]:
    """Compute Double Metaphone phonetic codes for a name."""
    primary, secondary = doublemetaphone(name)
    return (primary or "", secondary or "")


def update_phonetic_codes(
    limit: int | None = None,
    conn: sqlite3.Connection | None = None,
    compute_codes: Callable[[str], tuple[str, str]] = compute_phonetic_codes,
) -> int:
    """Update phonetic codes for names where codes are missing."""

    def _do_update(connection: sqlite3.Connection) -> int:
        query = """
            SELECT id, name FROM names
            WHERE phonetic_primary IS NULL
            OR phonetic_secondary IS NULL
        """
        if limit:
            query += f" LIMIT {limit}"

        rows = connection.execute(query).fetchall()

        updated = 0
        for name_id, name in rows:
            primary, secondary = compute_codes(name)
            connection.execute(
                """
                UPDATE names
                SET phonetic_primary = ?, phonetic_secondary = ?
                WHERE id = ?
                """,
                (primary, secondary, name_id),
            )
            updated += 1

        if updated > 0:
            logger.info("Updated phonetic codes for %d names", updated)
        else:
            logger.debug("No names need phonetic code updates")
        return updated

    if conn is None:
        with get_connection() as new_conn:
            return _do_update(new_conn)
    return _do_update(conn)


def get_unclassified_names(limit: int | None = None) -> list[UnclassifiedName]:
    """Get names that haven't been classified with origin region."""
    with get_connection() as conn:
        query = """
            SELECT id, name FROM names
            WHERE origin_region IS NULL
            ORDER BY id
        """
        if limit:
            query += f" LIMIT {limit}"

        cursor = conn.execute(query)
        return [UnclassifiedName(id=row[0], name=row[1]) for row in cursor.fetchall()]


def update_name_origin(name_id: int, region: str, confidence: float) -> None:
    """Update a name's origin region and confidence."""
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE names
            SET origin_region = ?,
                origin_confidence = ?,
                origin_classified_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (region, confidence, name_id),
        )


def get_names_with_origins(
    confidence_threshold: float = 0.5,
) -> dict[str, tuple[str, float, str, str]]:
    """Get known name -> (region, confidence, phonetic_primary, phonetic_secondary)."""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT name, origin_region, origin_confidence, phonetic_primary, phonetic_secondary
            FROM names
            WHERE origin_region IS NOT NULL
            AND origin_confidence >= ?
            """,
            (confidence_threshold,),
        )
        result = {}
        for row in cursor:
            result[row["name"]] = (
                row["origin_region"],
                row["origin_confidence"],
                row["phonetic_primary"] or "",
                row["phonetic_secondary"] or "",
            )
        return result


def get_names_by_filters(
    gender: str | None = None,
    origins: list[str] | None = None,
) -> list[str]:
    """Get names filtered by gender and origin regions."""
    query = "SELECT name FROM names WHERE 1=1"
    params = []

    if gender and gender != "All":
        query += " AND gender = ?"
        params.append(gender)

    if origins:
        if "International" in origins:
            placeholders = ", ".join(["?"] * (len(origins) - 1))
            template = " AND (origin_region IN ({ph}) OR origin_region IS NULL)"
            query = query + template.format(ph=placeholders)
            params.extend([o for o in origins if o != "International"])
        else:
            placeholders = ", ".join(["?"] * len(origins))
            template = " AND origin_region IN ({ph})"
            query = query + template.format(ph=placeholders)
            params.extend(origins)

    query += " ORDER BY name"

    with get_connection() as conn:
        cursor = conn.execute(query, params)
        return [row[0] for row in cursor.fetchall()]


def get_names_by_gender() -> dict[str, list[str]]:
    """Get names categorized by gender."""
    with get_connection() as conn:
        cursor = conn.execute("""
            SELECT name, gender FROM names
            WHERE gender IN ('Male', 'Female', 'Unisex')
            ORDER BY name
        """)
        rows = cursor.fetchall()

        gender_lists = {
            "Female": set(),
            "Male": set(),
            "Unisex": set(),
            "All": set(),
        }

        for name, gender in rows:
            gender_lists["All"].add(name)
            if gender in gender_lists:
                gender_lists[gender].add(name)
            if gender == "Unisex":
                gender_lists["Male"].add(name)
                gender_lists["Female"].add(name)

        return {gender: sorted(name_set) for gender, name_set in gender_lists.items()}


def get_all_origin_regions() -> list[str]:
    """Get distinct origin regions from names table, including NULL as International."""
    with get_connection() as conn:
        cursor = conn.execute("""
            SELECT DISTINCT
                CASE
                    WHEN origin_region IS NULL THEN 'International'
                    ELSE origin_region
                END as region
            FROM names
            ORDER BY region
        """)
        return [row[0] for row in cursor.fetchall()]


def get_name_details_batch(
    names: list[str],
) -> list[NameDetails]:
    """Get gender and origin_region for multiple names in batch."""
    if not names:
        return []

    result: list[NameDetails] = []

    for i in range(0, len(names), MAX_SQL_PARAMS):
        chunk = names[i : i + MAX_SQL_PARAMS]

        with get_connection() as conn:
            placeholders = ", ".join(["?"] * len(chunk))
            template = "SELECT name, gender, origin_region FROM names WHERE name IN ({ph})"
            query = template.format(ph=placeholders)
            cursor = conn.execute(query, chunk)
            rows = cursor.fetchall()

            details_map = {row[0]: NameDetails(gender=row[1], origin_region=row[2]) for row in rows}

            result.extend(details_map.get(name, NameDetails(gender=None, origin_region=None)) for name in chunk)

    return result


def get_phonetic_codes_batch(names: list[str]) -> dict[str, PhoneticCodes]:
    """Get phonetic codes for multiple names in batch."""
    if not names:
        return {}

    result: dict[str, PhoneticCodes] = {}

    for i in range(0, len(names), MAX_SQL_PARAMS):
        chunk = names[i : i + MAX_SQL_PARAMS]

        with get_connection() as conn:
            placeholders = ", ".join(["?"] * len(chunk))
            template = "SELECT name, phonetic_primary, phonetic_secondary FROM names WHERE name IN ({ph})"
            query = template.format(ph=placeholders)
            cursor = conn.execute(query, chunk)

            for name, primary, secondary in cursor.fetchall():
                result[name] = PhoneticCodes(primary=primary or "", secondary=secondary or "")

    return result
