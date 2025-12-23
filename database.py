"""
SQLite database for name ranking application.

Handles:
- Names from godkendtefornavne submodule
- Elo ratings
- Origin region classification
- User filter preferences
- Submodule version tracking
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DB_PATH = Path("names.db")


@contextmanager
def get_connection():
    """Context manager for database connections."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_database():
    """Initialize database schema if it doesn't exist."""
    logger.debug("Initializing database schema")
    with get_connection() as conn:
        # Names table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS names (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                gender TEXT CHECK(gender IN ('Male', 'Female', 'Unisex')),
                origin_region TEXT,
                origin_confidence REAL,
                origin_classified_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Ratings table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ratings (
                name_id INTEGER PRIMARY KEY REFERENCES names(id)
                ON DELETE CASCADE,
                rating REAL NOT NULL DEFAULT 1500.0,
                matches INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # User settings table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        # Region mapping table (nationality -> region)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS region_mapping (
                nationality TEXT PRIMARY KEY,
                region TEXT NOT NULL
            )
        """)

        # Source versions table (submodule tracking)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS source_versions (
                id INTEGER PRIMARY KEY,
                commit_hash TEXT NOT NULL,
                synced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_names_gender ON names(gender)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_names_origin "
            "ON names(origin_region)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ratings_rating ON ratings(rating)"
        )

        # Insert default region mapping if empty
        if (
            conn.execute("SELECT COUNT(*) FROM region_mapping").fetchone()[0]
            == 0
        ):
            _insert_default_region_mapping(conn)

        logger.info("Database initialized successfully")


def _insert_default_region_mapping(conn):
    """Insert default nationality -> region mapping."""
    # Nordic countries
    nordic = [
        "Denmark",
        "Norway",
        "Sweden",
        "Iceland",
        "Finland",
        "Faroe Islands",
    ]
    # European (other)
    european = [
        "Germany",
        "France",
        "Italy",
        "Spain",
        "Portugal",
        "Netherlands",
        "Belgium",
        "Switzerland",
        "Austria",
        "Poland",
        "Czech",
        "Slovakia",
        "Hungary",
        "Romania",
        "Bulgaria",
        "Greece",
        "Croatia",
        "Serbia",
        "Slovenia",
        "Lithuania",
        "Latvia",
        "Estonia",
        "United Kingdom",
        "Ireland",
        "Scotland",
        "Wales",
        "England",
        "Russia",
        "Ukraine",
        "Belarus",
        "Moldova",
        "Albania",
        "Bosnia",
        "Montenegro",
        "Macedonia",
        "Kosovo",
    ]
    # Middle Eastern
    middle_eastern = [
        "Egypt",
        "Turkey",
        "Iran",
        "Iraq",
        "Syria",
        "Jordan",
        "Lebanon",
        "Israel",
        "Palestine",
        "Saudi Arabia",
        "Yemen",
        "Oman",
        "United Arab Emirates",
        "Qatar",
        "Kuwait",
        "Bahrain",
        "Afghanistan",
        "Pakistan",
    ]
    # Asian
    asian = [
        "China",
        "Japan",
        "South Korea",
        "North Korea",
        "India",
        "Bangladesh",
        "Pakistan",
        "Sri Lanka",
        "Nepal",
        "Bhutan",
        "Myanmar",
        "Thailand",
        "Vietnam",
        "Cambodia",
        "Laos",
        "Philippines",
        "Indonesia",
        "Malaysia",
        "Singapore",
        "Taiwan",
        "Hong Kong",
        "Mongolia",
    ]
    # African
    african = [
        "Nigeria",
        "Ethiopia",
        "Egypt",
        "South Africa",
        "Kenya",
        "Tanzania",
        "Uganda",
        "Ghana",
        "Morocco",
        "Algeria",
        "Sudan",
        "Mozambique",
        "Madagascar",
        "Cameroon",
        "Ivory Coast",
        "Niger",
        "Burkina Faso",
        "Mali",
        "Senegal",
        "Zambia",
        "Zimbabwe",
        "Tunisia",
        "Libya",
    ]
    # American
    american = [
        "United States",
        "Canada",
        "Mexico",
        "Brazil",
        "Argentina",
        "Chile",
        "Colombia",
        "Peru",
        "Venezuela",
        "Ecuador",
        "Bolivia",
        "Paraguay",
        "Uruguay",
        "Cuba",
        "Dominican Republic",
        "Puerto Rico",
        "Jamaica",
        "Haiti",
        "Trinidad and Tobago",
        "Bahamas",
        "Barbados",
        "Guyana",
        "Suriname",
        "Belize",
        "Costa Rica",
        "Panama",
        "Nicaragua",
        "Honduras",
        "El Salvador",
        "Guatemala",
    ]
    # Oceanian
    oceanian = [
        "Australia",
        "New Zealand",
        "Fiji",
        "Papua New Guinea",
        "Samoa",
        "Tonga",
        "Vanuatu",
        "Solomon Islands",
        "Micronesia",
        "Palau",
        "Marshall Islands",
        "Kiribati",
        "Tuvalu",
        "Nauru",
    ]

    mappings = []
    for country in nordic:
        mappings.append(("Nordic", country))
    for country in european:
        mappings.append(("European", country))
    for country in middle_eastern:
        mappings.append(("Middle Eastern", country))
    for country in asian:
        mappings.append(("Asian", country))
    for country in african:
        mappings.append(("African", country))
    for country in american:
        mappings.append(("American", country))
    for country in oceanian:
        mappings.append(("Oceanian", country))

    conn.executemany(
        "INSERT OR IGNORE INTO region_mapping "
        "(region, nationality) VALUES (?, ?)",
        mappings,
    )


def sync_names_with_submodule(submodule_path: Path = Path("godkendtefornavne")):
    """
    Sync names from submodule JSON file to database.
    Only inserts new names that don't exist in the database.
    Tracks submodule commit hash to avoid redundant processing.
    """
    logger.debug("Syncing names from submodule")
    json_path = submodule_path / "allenavne.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Submodule JSON not found: {json_path}")

    # Get current submodule commit hash
    import subprocess

    try:
        result = subprocess.run(
            ["git", "-C", str(submodule_path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        current_commit = result.stdout.strip()
        logger.debug(f"Submodule commit hash: {current_commit}")
    except Exception as e:
        raise RuntimeError(f"Failed to get submodule commit hash: {e}")

    # Check if we've already synced this commit
    with get_connection() as conn:
        last_sync = conn.execute(
            "SELECT commit_hash FROM source_versions ORDER BY id DESC LIMIT 1"
        ).fetchone()

        if last_sync and last_sync[0] == current_commit:
            logger.debug("Already synced with current commit")
            return 0  # Already synced

    # Load JSON data
    import pandas as pd

    df = pd.read_json(json_path, encoding="utf-8")
    logger.info(f"Loaded {len(df)} rows from JSON")

    # Validate columns
    if not all(col in df.columns for col in ["name", "gender"]):
        raise ValueError("JSON missing required columns 'name' and/or 'gender'")

    # Filter valid names
    from data_loader import is_valid_name

    valid_names = []
    for _, row in df.iterrows():
        name = str(row["name"]).strip()
        gender = str(row["gender"]).strip()
        if is_valid_name(name):
            valid_names.append((name, gender))

    logger.debug(f"Filtered {len(valid_names)} valid names")
    # Insert new names
    inserted_count = 0
    with get_connection() as conn:
        if valid_names:
            before = conn.total_changes
            conn.executemany(
                "INSERT OR IGNORE INTO names (name, gender) VALUES (?, ?)",
                valid_names,
            )
            inserted_count = conn.total_changes - before
            logger.debug(f"Bulk insert attempted, {inserted_count} new rows")

        # Record this sync
        conn.execute(
            "INSERT INTO source_versions (commit_hash) VALUES (?)",
            (current_commit,),
        )

    logger.info(f"Inserted {inserted_count} new names")
    return inserted_count


def get_unclassified_names(limit: Optional[int] = None) -> List[Dict[str, Any]]:
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
        return [dict(row) for row in cursor.fetchall()]


def update_name_origin(name_id: int, region: str, confidence: float):
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


def get_names_by_filters(
    gender: Optional[str] = None, origins: Optional[List[str]] = None
) -> List[str]:
    """
    Get names filtered by gender and origin regions.
    Returns list of names.
    """
    query = "SELECT name FROM names WHERE 1=1"
    params = []

    if gender and gender != "All":
        query += " AND gender = ?"
        params.append(gender)

    if origins:
        # If origins list is provided but empty, return no names
        # If origins contains "International", include NULL origin_region
        if "International" in origins:
            # Include both NULL and specified regions
            placeholders = ", ".join(["?"] * (len(origins) - 1))
            query += (
                f" AND (origin_region IN ({placeholders}) "
                f"OR origin_region IS NULL)"
            )
            params.extend([o for o in origins if o != "International"])
        else:
            placeholders = ", ".join(["?"] * len(origins))
            query += f" AND origin_region IN ({placeholders})"
            params.extend(origins)

    query += " ORDER BY name"

    with get_connection() as conn:
        cursor = conn.execute(query, params)
        return [row[0] for row in cursor.fetchall()]


def get_names_by_gender() -> Dict[str, List[str]]:
    """
    Get names categorized by gender.
    Returns dict with keys: "Male", "Female", "Unisex", "All".
    Unisex names are included in both "Male" and "Female" categories.
    """
    with get_connection() as conn:
        # Get all names with gender
        cursor = conn.execute("""
            SELECT name, gender FROM names 
            WHERE gender IN ('Male', 'Female', 'Unisex')
            ORDER BY name
        """)
        rows = cursor.fetchall()

        # Initialize gender categories
        gender_lists = {
            "Female": set(),
            "Male": set(),
            "Unisex": set(),
            "All": set(),
        }

        # Categorize names
        for name, gender in rows:
            # Always add to 'All' category
            gender_lists["All"].add(name)

            # Add to specific gender category
            if gender in gender_lists:
                gender_lists[gender].add(name)

            # Unisex names also go to both Male and Female categories
            if gender == "Unisex":
                gender_lists["Male"].add(name)
                gender_lists["Female"].add(name)

        # Convert sets to sorted lists
        result = {}
        for gender, name_set in gender_lists.items():
            result[gender] = sorted(list(name_set))

        return result


def get_all_origin_regions() -> List[str]:
    """Get distinct origin regions from names table,
    including NULL as 'International'."""
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


def get_ratings() -> Dict[str, float]:
    """Get all ratings as name -> rating dictionary."""
    with get_connection() as conn:
        cursor = conn.execute("""
            SELECT n.name, r.rating
            FROM names n
            JOIN ratings r ON n.id = r.name_id
        """)
        return {row[0]: row[1] for row in cursor.fetchall()}


def update_rating(name: str, rating: float):
    """Update or insert rating for a name."""
    with get_connection() as conn:
        # Get name_id
        name_id = conn.execute(
            "SELECT id FROM names WHERE name = ?", (name,)
        ).fetchone()
        if not name_id:
            raise ValueError(f"Name not found: {name}")

        name_id = name_id[0]

        # Update or insert rating
        conn.execute(
            """
            INSERT OR REPLACE INTO ratings 
            (name_id, rating, matches, last_updated)
            VALUES (
                ?,
                ?,
                COALESCE((
                    SELECT matches + 1 FROM ratings 
                    WHERE name_id = ?
                ), 1),
                CURRENT_TIMESTAMP
            )
        """,
            (name_id, rating, name_id),
        )


def update_rating_with_match(name: str, rating: float):
    """Update rating and increment match count."""
    update_rating(name, rating)


def save_user_setting(key: str, value: str):
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
            "SELECT value FROM user_settings WHERE key = ?", (key,)
        )
        row = cursor.fetchone()
        return row[0] if row else default


def migrate_ratings_from_json(json_path: Path = Path("ratings.json")) -> int:
    """
    Migrate ratings from JSON file to database.
    Returns number of ratings migrated.
    """
    if not json_path.exists():
        return 0

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ratings = data.get("ratings", data) if isinstance(data, dict) else data

    migrated = 0
    with get_connection() as conn:
        for name, rating in ratings.items():
            # Find name_id
            cursor = conn.execute(
                "SELECT id FROM names WHERE name = ?", (name,)
            )
            row = cursor.fetchone()
            if row:
                name_id = row[0]
                # Insert rating
                conn.execute(
                    """
                    INSERT OR REPLACE INTO ratings 
                    (name_id, rating) VALUES (?, ?)
                """,
                    (name_id, rating),
                )
                migrated += 1

    return migrated


def get_stats() -> Dict[str, Any]:
    """Get database statistics."""
    with get_connection() as conn:
        total_names = conn.execute("SELECT COUNT(*) FROM names").fetchone()[0]
        classified_names = conn.execute(
            "SELECT COUNT(*) FROM names WHERE origin_region IS NOT NULL"
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

        return {
            "total_names": total_names,
            "classified_names": classified_names,
            "rated_names": rated_names,
            "origin_distribution": origin_dist,
        }


if __name__ == "__main__":
    # Initialize database if run directly
    init_database()
    print("Database initialized successfully")
    stats = get_stats()
    print(f"Total names: {stats['total_names']}")
    print(f"Classified names: {stats['classified_names']}")
    print(f"Rated names: {stats['rated_names']}")
