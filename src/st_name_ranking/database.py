"""SQLite database for name ranking application.

Handles:
- Names from godkendtefornavne submodule
- Elo ratings
- Origin region classification
- User filter preferences
- Submodule version tracking
"""

import datetime as dt
import logging
import shutil
import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path

from metaphone import doublemetaphone

from st_name_ranking.types import (
    DatabaseStats,
    NameDetails,
    PhoneticCodes,
    PreferenceStats,
    SourceVersion,
    UnclassifiedName,
)

logger = logging.getLogger(__name__)

# Track initialization status
_initialized = False

DB_PATH = Path("data/names.db")

# Default rating for new names
INITIAL_SCORE = 1500.0

# SQLite parameter limit (safely below 999)
MAX_SQL_PARAMS = 500


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


def _compute_phonetic_codes(name: str) -> tuple[str, str]:
    """Compute Double Metaphone phonetic codes for a name.
    Returns (primary_code, secondary_code).
    """
    primary, secondary = doublemetaphone(name)
    return (primary or "", secondary or "")


def update_phonetic_codes(limit: int | None = None) -> int:
    """Update phonetic codes for names where phonetic_primary is NULL.
    Returns number of names updated.
    """
    with get_connection() as conn:
        # Get names missing phonetic codes
        query = """
            SELECT id, name FROM names
            WHERE phonetic_primary IS NULL
            OR phonetic_secondary IS NULL
        """
        if limit:
            query += f" LIMIT {limit}"

        cursor = conn.execute(query)
        rows = cursor.fetchall()

        updated = 0
        for row in rows:
            name_id, name = row
            primary, secondary = _compute_phonetic_codes(name)
            conn.execute(
                """
                UPDATE names
                SET phonetic_primary = ?, phonetic_secondary = ?
                WHERE id = ?
                """,
                (primary, secondary, name_id),
            )
            updated += 1

        if updated > 0:
            logger.info(f"Updated phonetic codes for {updated} names")
        else:
            logger.debug("No names need phonetic code updates")
        return updated


def _migrate_comparisons_table_if_needed(conn) -> None:
    """Migrate comparisons table to support preference=2 (both disliked) if needed."""
    # Check if CHECK constraint already includes 2
    # SQLite doesn't expose CHECK constraints easily, so we try a test insert
    # Get a valid name_id pair for testing (create dummy names if none exist)
    cursor = conn.execute("SELECT id FROM names LIMIT 2")
    rows = cursor.fetchall()
    if len(rows) < 2:
        # Need at least two names for test - insert temporary names
        conn.execute("INSERT OR IGNORE INTO names (name) VALUES ('__temp_a')")
        conn.execute("INSERT OR IGNORE INTO names (name) VALUES ('__temp_b')")
        cursor = conn.execute("SELECT id FROM names WHERE name IN ('__temp_a', '__temp_b')")
        rows = cursor.fetchall()

    id_a, id_b = rows[0][0], rows[1][0]

    try:
        # Try to insert a comparison with preference=2
        conn.execute(
            "INSERT INTO comparisons (name_a_id, name_b_id, preference) VALUES (?, ?, 2)",
            (id_a, id_b),
        )
        # If successful, constraint already allows 2
        conn.execute("DELETE FROM comparisons WHERE name_a_id = ? AND name_b_id = ?", (id_a, id_b))
        logger.debug("Comparisons table already supports preference=2")
    except sqlite3.IntegrityError as e:
        if "CHECK" in str(e):
            # Constraint violation - need to migrate
            logger.info("Migrating comparisons table to support preference=2")
            # Clean up any leftover comparisons_new from previous failed migration
            conn.execute("DROP TABLE IF EXISTS comparisons_new")
            # Create new table with updated constraint
            conn.execute("""
                CREATE TABLE comparisons_new (
                    id INTEGER PRIMARY KEY,
                    name_a_id INTEGER NOT NULL REFERENCES names(id),
                    name_b_id INTEGER NOT NULL REFERENCES names(id),
                    preference INTEGER NOT NULL CHECK(preference IN (-1, 0, 1, 2)),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(name_a_id, name_b_id, preference)
                )
            """)
            # Copy existing data, deduplicating on (name_a_id, name_b_id, preference)
            # Keep the row with the latest created_at for each duplicate (most recent preference)
            conn.execute("""
                INSERT OR IGNORE INTO comparisons_new (name_a_id, name_b_id, preference, created_at)
                SELECT name_a_id, name_b_id, preference, MAX(created_at) as created_at
                FROM comparisons
                GROUP BY name_a_id, name_b_id, preference
            """)
            # Drop old table
            conn.execute("DROP TABLE comparisons")
            # Rename new table
            conn.execute("ALTER TABLE comparisons_new RENAME TO comparisons")
            # Recreate indexes (they will be recreated later in init_database)
            logger.info("Comparisons table migrated successfully")
        else:
            # Some other integrity error (e.g., UNIQUE) - ignore
            pass
    finally:
        # Clean up temporary names if we inserted them
        conn.execute("DELETE FROM names WHERE name IN ('__temp_a', '__temp_b')")


def init_database():
    """Initialize database schema if it doesn't exist."""
    global _initialized
    if _initialized:
        logger.debug("Database already initialized, skipping initialization")
        return

    # Mark as initialized immediately to prevent race conditions
    # If initialization fails, the app won't work anyway, so we don't need to retry
    _initialized = True

    # Check if database file exists
    db_exists = DB_PATH.exists()

    with get_connection() as conn:
        # Check if names table exists to determine if database is already set up
        table_exists = False
        if db_exists:
            try:
                # Try to query the names table
                conn.execute("SELECT 1 FROM names LIMIT 1").fetchone()
                table_exists = True
            except sqlite3.OperationalError:
                # Table doesn't exist
                table_exists = False

        # Log appropriate message based on database state
        if not db_exists:
            logger.info("Creating new database at %s", DB_PATH)
        elif not table_exists:
            logger.info("Initializing database schema for existing file")
        else:
            logger.info("Using existing database, ensuring schema is up to date")

        # Database is already marked as initialized to prevent race conditions

        # Names table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS names (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                gender TEXT CHECK(gender IN ('Male', 'Female', 'Unisex')),
                origin_region TEXT,
                origin_confidence REAL,
                origin_classified_at TIMESTAMP,
                phonetic_primary TEXT,
                phonetic_secondary TEXT,
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

        # Model state table for Bradley-Terry model
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_state (
                id INTEGER PRIMARY KEY,
                feature_weights BLOB NOT NULL,
                uncertainty_matrix BLOB NOT NULL,
                training_samples INTEGER DEFAULT 0,
                feature_names_json TEXT NOT NULL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Comparisons table for tracking user preferences
        conn.execute("""
            CREATE TABLE IF NOT EXISTS comparisons (
                id INTEGER PRIMARY KEY,
                name_a_id INTEGER NOT NULL REFERENCES names(id),
                name_b_id INTEGER NOT NULL REFERENCES names(id),
                preference INTEGER NOT NULL CHECK(preference IN (-1, 0, 1)),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(name_a_id, name_b_id, preference)  -- Prevent duplicate comparisons
            )
        """)

        # Migrate comparisons table to support preference=2 if needed
        _migrate_comparisons_table_if_needed(conn)

        # Ensure phonetic columns exist (migration for existing databases)
        cursor = conn.execute("PRAGMA table_info(names)")
        columns = [row[1] for row in cursor.fetchall()]
        if "phonetic_primary" not in columns:
            conn.execute("ALTER TABLE names ADD COLUMN phonetic_primary TEXT")
            logger.debug("Added phonetic_primary column")
        if "phonetic_secondary" not in columns:
            conn.execute("ALTER TABLE names ADD COLUMN phonetic_secondary TEXT")
            logger.debug("Added phonetic_secondary column")

        # Update phonetic codes for existing names if missing
        update_phonetic_codes()

        # Create indexes
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_names_gender ON names(gender)",
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_names_origin ON names(origin_region)",
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ratings_rating ON ratings(rating)",
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_names_phonetic_primary ON names(phonetic_primary)",
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_names_phonetic_secondary ON names(phonetic_secondary)",
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_comparisons_name_a ON comparisons(name_a_id)",
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_comparisons_name_b ON comparisons(name_b_id)",
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_comparisons_created ON comparisons(created_at)",
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_model_state_updated ON model_state(last_updated)",
        )

        # Insert default region mapping if empty
        if conn.execute("SELECT COUNT(*) FROM region_mapping").fetchone()[0] == 0:
            _insert_default_region_mapping(conn)

        logger.info("Database schema verified successfully")


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
        "INSERT OR IGNORE INTO region_mapping (region, nationality) VALUES (?, ?)",
        mappings,
    )


def sync_names_with_submodule(submodule_path: Path = Path("godkendtefornavne")):
    """Sync names from submodule JSON file to database.
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
            check=False,
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
            "SELECT commit_hash FROM source_versions ORDER BY id DESC LIMIT 1",
        ).fetchone()

        if last_sync and last_sync[0] == current_commit:
            logger.debug("Already synced with current commit")
            return 0  # Already synced

    # Load JSON data
    import pandas as pd

    df = pd.read_json(json_path, encoding="utf-8")
    logger.info(f"Loaded {len(df)} rows from JSON")

    # Handle empty JSON
    if df.empty:
        logger.debug("Empty JSON, nothing to sync")
        return 0

    # Validate columns
    if not all(col in df.columns for col in ["name", "gender"]):
        raise ValueError("JSON missing required columns 'name' and/or 'gender'")

    # Filter valid names
    from st_name_ranking.data_loader import is_valid_name

    valid_names = []
    for _, row in df.iterrows():
        name = str(row["name"]).strip()
        gender_raw = str(row["gender"]).strip()
        # Map gender codes to full names
        gender_map = {
            "F": "Female",
            "M": "Male",
            "U": "Unisex",
            "female": "Female",
            "male": "Male",
            "unisex": "Unisex",
            "Female": "Female",
            "Male": "Male",
            "Unisex": "Unisex",
        }
        gender = gender_map.get(gender_raw)
        if not gender:
            # Try case-insensitive match
            gender = gender_map.get(gender_raw.lower())
        if not gender:
            logger.warning(
                f"Invalid gender '{gender_raw}' for name '{name}', skipping",
            )
            continue
        if is_valid_name(name):
            # Compute phonetic codes for similarity matching
            primary, secondary = doublemetaphone(name)
            primary = primary or ""
            secondary = secondary or ""
            valid_names.append((name, gender, primary, secondary))

    logger.debug(f"Filtered {len(valid_names)} valid names")
    # Insert new names
    inserted_count = 0
    with get_connection() as conn:
        if valid_names:
            before = conn.total_changes
            conn.executemany(
                "INSERT OR IGNORE INTO names (name, gender, phonetic_primary, phonetic_secondary) VALUES (?, ?, ?, ?)",
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


def get_latest_submodule_version() -> SourceVersion | None:
    """Get the latest submodule version (commit hash) from source_versions."""
    with get_connection() as conn:
        cursor = conn.execute(
            "SELECT commit_hash FROM source_versions ORDER BY id DESC LIMIT 1",
        )
        row = cursor.fetchone()
        if row:
            return SourceVersion(commit_hash=row[0])
        return None


def update_submodule_version(commit_hash: str, names_count: int):
    """Update submodule version (commit hash) and names count."""
    with get_connection() as conn:
        # Insert new record (always add new row)
        conn.execute(
            "INSERT INTO source_versions (commit_hash) VALUES (?)",
            (commit_hash,),
        )


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


def get_names_with_origins(
    confidence_threshold: float = 0.5,
) -> dict[str, tuple[str, float, str, str]]:
    """Get dictionary of known name -> (region, confidence, phonetic_primary, phonetic_secondary)
    for names with origin classification above confidence threshold.

    Useful for phonetic similarity classification.
    """
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
    """Get names filtered by gender and origin regions.
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
            query += f" AND (origin_region IN ({placeholders}) OR origin_region IS NULL)"
            params.extend([o for o in origins if o != "International"])
        else:
            placeholders = ", ".join(["?"] * len(origins))
            query += f" AND origin_region IN ({placeholders})"
            params.extend(origins)

    query += " ORDER BY name"

    with get_connection() as conn:
        cursor = conn.execute(query, params)
        return [row[0] for row in cursor.fetchall()]


def get_names_by_gender() -> dict[str, list[str]]:
    """Get names categorized by gender.
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


def get_all_origin_regions() -> list[str]:
    """Get distinct origin regions from names table,
    including NULL as 'International'.
    """
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


def get_ratings() -> dict[str, float]:
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
            "SELECT id FROM names WHERE name = ?",
            (name,),
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


def update_rating_value(name: str, rating: float):
    """Update rating value without incrementing match count."""
    with get_connection() as conn:
        # Get name_id
        name_id = conn.execute(
            "SELECT id FROM names WHERE name = ?",
            (name,),
        ).fetchone()
        if not name_id:
            raise ValueError(f"Name not found: {name}")

        name_id = name_id[0]

        # Update or insert rating, preserving existing matches or default 0
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


def update_ratings_batch(ratings_dict: dict[str, float]) -> None:
    """Update multiple ratings efficiently in a single transaction.

    Args:
        ratings_dict: Dictionary mapping name -> new rating

    """
    if not ratings_dict:
        return

    with get_connection() as conn:
        for name, rating in ratings_dict.items():
            # Get name_id
            name_id = conn.execute(
                "SELECT id FROM names WHERE name = ?",
                (name,),
            ).fetchone()
            if not name_id:
                logger.warning(f"Name not found in database: {name}")
                continue

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


def update_ratings_batch_values(ratings_dict: dict[str, float]) -> None:
    """Update multiple ratings efficiently without incrementing match counts.

    Args:
        ratings_dict: Dictionary mapping name -> new rating

    """
    if not ratings_dict:
        return

    with get_connection() as conn:
        for name, rating in ratings_dict.items():
            # Get name_id
            name_id = conn.execute(
                "SELECT id FROM names WHERE name = ?",
                (name,),
            ).fetchone()
            if not name_id:
                logger.warning(f"Name not found in database: {name}")
                continue

            name_id = name_id[0]

            # Update or insert rating, preserving existing matches
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


def record_comparison(name_a: str, name_b: str, preference: int) -> None:
    """Record a pairwise comparison in the database.

    Args:
        name_a: First name in comparison
        name_b: Second name in comparison
        preference: -1 (name_a preferred), 0 (draw), 1 (name_b preferred), 2 (both disliked)

    Raises:
        ValueError: If preference not in (-1, 0, 1, 2)
    """
    if preference not in (-1, 0, 1, 2):
        raise ValueError("preference must be -1, 0, 1, or 2")

    with get_connection() as conn:
        # Get name IDs
        cursor = conn.execute(
            "SELECT id FROM names WHERE name = ?",
            (name_a,),
        )
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Name not found: {name_a}")
        name_a_id = row[0]

        cursor = conn.execute(
            "SELECT id FROM names WHERE name = ?",
            (name_b,),
        )
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Name not found: {name_b}")
        name_b_id = row[0]

        # Insert comparison (ignore duplicates due to UNIQUE constraint)
        conn.execute(
            """
            INSERT OR IGNORE INTO comparisons
            (name_a_id, name_b_id, preference, created_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (name_a_id, name_b_id, preference),
        )


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
            "SELECT value FROM user_settings WHERE key = ?",
            (key,),
        )
        row = cursor.fetchone()
        return row[0] if row else default


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

        unclassified_names = total_names - classified_names

        return DatabaseStats(
            total_names=total_names,
            classified_names=classified_names,
            unclassified_names=unclassified_names,
            rated_names=rated_names,
            origin_distribution=origin_dist,
        )


def get_comparison_count(name: str) -> int:
    """Get number of comparisons involving a name."""
    with get_connection() as conn:
        # Get name ID
        cursor = conn.execute("SELECT id FROM names WHERE name = ?", (name,))
        row = cursor.fetchone()
        if not row:
            return 0
        name_id = row[0]

        # Count comparisons where name appears as name_a or name_b
        cursor = conn.execute(
            """
            SELECT COUNT(*) FROM comparisons
            WHERE name_a_id = ? OR name_b_id = ?
            """,
            (name_id, name_id),
        )
        return cursor.fetchone()[0]


def get_preference_stats_by_gender() -> dict[str, PreferenceStats]:
    """Get preference statistics grouped by gender.

    Returns:
        Dictionary mapping gender -> PreferenceStats
    """
    with get_connection() as conn:
        # CTE to compute outcomes for each name in each comparison
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
        rows = cursor.fetchall()

        # Initialize result dict
        result: dict[str, dict[str, int]] = {}
        key_map = {"win": "wins", "loss": "losses", "draw": "draws"}
        for gender, outcome, count in rows:
            if gender not in result:
                result[gender] = {"wins": 0, "losses": 0, "draws": 0, "total": 0}
            result[gender][key_map[outcome]] = count
            result[gender]["total"] += count

        # Convert to PreferenceStats
        return {
            gender: PreferenceStats(
                wins=data["wins"],
                losses=data["losses"],
                draws=data["draws"],
                total=data["total"],
            )
            for gender, data in result.items()
        }


def get_preference_stats_by_origin() -> dict[str, PreferenceStats]:
    """Get preference statistics grouped by origin region.

    Returns:
        Dictionary mapping origin region -> PreferenceStats
    """
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
        rows = cursor.fetchall()

        result: dict[str, dict[str, int]] = {}
        key_map = {"win": "wins", "loss": "losses", "draw": "draws"}
        for region, outcome, count in rows:
            if region not in result:
                result[region] = {"wins": 0, "losses": 0, "draws": 0, "total": 0}
            result[region][key_map[outcome]] = count
            result[region]["total"] += count

        # Convert to PreferenceStats
        return {
            region: PreferenceStats(
                wins=data["wins"],
                losses=data["losses"],
                draws=data["draws"],
                total=data["total"],
            )
            for region, data in result.items()
        }


def get_preference_stats_by_phonetic() -> dict[str, PreferenceStats]:
    """Get preference statistics grouped by phonetic primary code.

    Returns:
        Dictionary mapping phonetic code -> PreferenceStats
    """
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
        rows = cursor.fetchall()

        result: dict[str, dict[str, int]] = {}
        key_map = {"win": "wins", "loss": "losses", "draw": "draws"}
        for phonetic_code, outcome, count in rows:
            if phonetic_code not in result:
                result[phonetic_code] = {"wins": 0, "losses": 0, "draws": 0, "total": 0}
            result[phonetic_code][key_map[outcome]] = count
            result[phonetic_code]["total"] += count

        # Convert to PreferenceStats
        return {
            phonetic_code: PreferenceStats(
                wins=data["wins"],
                losses=data["losses"],
                draws=data["draws"],
                total=data["total"],
            )
            for phonetic_code, data in result.items()
        }


def get_name_details_batch(
    names: list[str],
) -> list[NameDetails]:
    """Get gender and origin_region for multiple names in batch.
    Returns list of NameDetails.
    """
    if not names:
        return []

    # Process in chunks to avoid SQL parameter limit
    chunk_size = MAX_SQL_PARAMS
    result: list[NameDetails] = []

    for i in range(0, len(names), chunk_size):
        chunk = names[i : i + chunk_size]

        with get_connection() as conn:
            placeholders = ", ".join(["?"] * len(chunk))
            query = f"""
                SELECT name, gender, origin_region FROM names
                WHERE name IN ({placeholders})
            """
            cursor = conn.execute(query, chunk)
            rows = cursor.fetchall()

            # Create mapping for fast lookup
            details_map = {row[0]: NameDetails(gender=row[1], origin_region=row[2]) for row in rows}

            # Append results for this chunk in order
            for name in chunk:
                if name in details_map:
                    result.append(details_map[name])
                else:
                    result.append(NameDetails(gender=None, origin_region=None))

    return result


def get_phonetic_codes_batch(names: list[str]) -> dict[str, PhoneticCodes]:
    """Get phonetic codes for multiple names in batch.
    Returns dict mapping name -> PhoneticCodes.
    Missing names are omitted from the result.
    """
    if not names:
        return {}

    # Process in chunks to avoid SQL parameter limit
    chunk_size = MAX_SQL_PARAMS
    result: dict[str, PhoneticCodes] = {}

    for i in range(0, len(names), chunk_size):
        chunk = names[i : i + chunk_size]

        with get_connection() as conn:
            placeholders = ", ".join(["?"] * len(chunk))
            query = f"""
                SELECT name, phonetic_primary, phonetic_secondary FROM names
                WHERE name IN ({placeholders})
            """
            cursor = conn.execute(query, chunk)
            rows = cursor.fetchall()

            for row in rows:
                name = row[0]
                primary = row[1] or ""
                secondary = row[2] or ""
                result[name] = PhoneticCodes(primary=primary, secondary=secondary)

    return result


def initialize_ratings(names: list[str]) -> dict[str, float]:
    """Initialize ratings for a list of names with default score.

    Args:
        names: List of names

    Returns:
        Dictionary mapping name -> INITIAL_SCORE

    """
    return dict.fromkeys(names, INITIAL_SCORE)


def export_database() -> bytes:
    """Export the entire SQLite database as bytes.

    Returns:
        Bytes of the database file.

    Raises:
        FileNotFoundError: If database file doesn't exist.
        IOError: If unable to read database file.
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database file not found at {DB_PATH}")

    # Ensure any pending writes are flushed by closing all connections
    # SQLite handles concurrent reads fine, but we need to ensure no open write transactions
    # We'll just read the file bytes
    try:
        with open(DB_PATH, "rb") as f:
            return f.read()
    except Exception as e:
        raise OSError(f"Failed to read database file: {e}")


def import_database(file_bytes: bytes, backup: bool = True) -> None:
    """Replace current database with uploaded SQLite database.

    Args:
        file_bytes: Bytes of the SQLite database file.
        backup: Whether to create a backup of the current database.

    Raises:
        ValueError: If uploaded file is not a valid SQLite database.
        IOError: If unable to write database file.
    """
    # Validate uploaded file is a valid SQLite database
    try:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            # Try to connect and query PRAGMA user_version (simple validation)
            conn = sqlite3.connect(tmp.name)
            try:
                conn.execute("SELECT 1 FROM sqlite_master LIMIT 1")
            finally:
                conn.close()
    except sqlite3.Error as e:
        raise ValueError(f"Uploaded file is not a valid SQLite database: {e}")

    # Create backup of current database if it exists
    if backup and DB_PATH.exists():
        backup_path = DB_PATH.with_suffix(f".db.backup.{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        shutil.copy2(DB_PATH, backup_path)
        logger.info("Created backup of current database at %s", backup_path)

    # Write new database file
    try:
        with open(DB_PATH, "wb") as f:
            f.write(file_bytes)
    except Exception as e:
        raise OSError(f"Failed to write database file: {e}")

    # Reset initialization flag to force re-initialization
    global _initialized
    _initialized = False

    logger.info("Database imported successfully")


if __name__ == "__main__":
    # Initialize database if run directly
    init_database()
    print("Database initialized successfully")
    stats = get_stats()
    print(f"Total names: {stats['total_names']}")
    print(f"Classified names: {stats['classified_names']}")
    print(f"Unclassified names: {stats['unclassified_names']}")
    print(f"Rated names: {stats['rated_names']}")
