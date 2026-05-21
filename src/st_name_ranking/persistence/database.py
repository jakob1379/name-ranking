"""SQLite schema/bootstrap facade for the name ranking application.

Focused persistence modules own domain reads and writes. This module keeps
schema initialization plus the historical import surface during migration.
"""

import logging
import sqlite3
from contextlib import AbstractContextManager
from pathlib import Path

from metaphone import doublemetaphone

from st_name_ranking.persistence import (
    connection as db_connection,
)
from st_name_ranking.persistence import (
    database_io,
    name_store,
    preference_stats_store,
    ratings_store,
    region_store,
    settings_store,
    stats_store,
)
from st_name_ranking.persistence.feature_store import (  # noqa: F401
    CorruptFeatureCacheError,
    get_active_feature_set_version,
    get_cached_features,
    get_cached_features_batch,
    get_feature_cache_stats,
    get_feature_set_by_version,
    get_or_create_feature_set,
    is_features_computed,
    set_cached_features,
    set_cached_features_batch,
)
from st_name_ranking.persistence.sync_store import (  # noqa: F401
    get_latest_submodule_version,
    sync_names_with_submodule,
    update_submodule_version,
)
from st_name_ranking.types import (
    DatabaseStats,
    NameDetails,
    PhoneticCodes,
    PreferenceStats,
    UnclassifiedName,
)

logger = logging.getLogger(__name__)

_INIT_STATE = db_connection._INIT_STATE
INITIAL_SCORE = db_connection.INITIAL_SCORE
MAX_SQL_PARAMS = db_connection.MAX_SQL_PARAMS


def get_db_path() -> Path:
    """Return the active SQLite database path."""
    return db_connection.get_db_path()


def set_db_path(path: str | Path) -> None:
    """Set the active SQLite database path and reset initialization state."""
    db_connection.set_db_path(path)


def reset_database_init_state() -> None:
    """Reset cached database-initialization state."""
    db_connection.reset_database_init_state()


def get_connection(timeout: float = 30.0) -> AbstractContextManager[sqlite3.Connection]:
    """Return a database connection using the active connection path."""
    return db_connection.get_connection(timeout)


def export_database() -> bytes:
    """Export the current SQLite database file as bytes."""
    return database_io.export_database()


def import_database(file_bytes: bytes, *, backup: bool = True) -> None:
    """Replace the current SQLite database with uploaded bytes."""
    database_io.import_database(file_bytes, backup=backup)


def _compute_phonetic_codes(name: str) -> tuple[str, str]:
    """Compute Double Metaphone phonetic codes for a name."""
    primary, secondary = doublemetaphone(name)
    return (primary or "", secondary or "")


def update_phonetic_codes(limit: int | None = None, conn: sqlite3.Connection | None = None) -> int:
    """Update phonetic codes for names where codes are missing."""
    return name_store.update_phonetic_codes(limit=limit, conn=conn, compute_codes=_compute_phonetic_codes)


MIN_NAMES_FOR_COMPARISON_TEST = 2
COMPARISON_PROBE_NAME_A = "__st_name_ranking_comparison_probe_a__"
COMPARISON_PROBE_NAME_B = "__st_name_ranking_comparison_probe_b__"
COMPARISON_PROBE_SAVEPOINT = "comparison_preference_probe"


def _comparison_table_supports_both_disliked(conn: sqlite3.Connection) -> bool:
    """Probe preference=2 support without persisting rows or touching real comparisons."""
    conn.execute(f"SAVEPOINT {COMPARISON_PROBE_SAVEPOINT}")
    try:
        conn.execute(
            "INSERT OR IGNORE INTO names (name) VALUES (?), (?)",
            (COMPARISON_PROBE_NAME_A, COMPARISON_PROBE_NAME_B),
        )
        rows = conn.execute(
            "SELECT id FROM names WHERE name IN (?, ?) ORDER BY name",
            (COMPARISON_PROBE_NAME_A, COMPARISON_PROBE_NAME_B),
        ).fetchall()
        if len(rows) < MIN_NAMES_FOR_COMPARISON_TEST:
            msg = "Failed to create comparison migration probe names"
            raise RuntimeError(msg)

        id_a, id_b = rows[0][0], rows[1][0]
        conn.execute(
            "INSERT INTO comparisons (name_a_id, name_b_id, preference) VALUES (?, ?, 2)",
            (id_a, id_b),
        )
    except sqlite3.IntegrityError as e:
        if "CHECK" in str(e):
            return False
        raise
    finally:
        conn.execute(f"ROLLBACK TO SAVEPOINT {COMPARISON_PROBE_SAVEPOINT}")
        conn.execute(f"RELEASE SAVEPOINT {COMPARISON_PROBE_SAVEPOINT}")

    return True


def _migrate_comparisons_table_if_needed(conn: sqlite3.Connection) -> None:
    """Migrate comparisons table to support preference=2 (both disliked) if needed."""
    if _comparison_table_supports_both_disliked(conn):
        logger.debug("Comparisons table already supports preference=2")
        return

    logger.info("Migrating comparisons table to support preference=2")
    conn.execute("DROP TABLE IF EXISTS comparisons_new")
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
    conn.execute("""
        INSERT OR IGNORE INTO comparisons_new (name_a_id, name_b_id, preference, created_at)
        SELECT name_a_id, name_b_id, preference, MAX(created_at) as created_at
        FROM comparisons
        GROUP BY name_a_id, name_b_id, preference
    """)
    conn.execute("DROP TABLE comparisons")
    conn.execute("ALTER TABLE comparisons_new RENAME TO comparisons")
    logger.info("Comparisons table migrated successfully")


def _ensure_schema(conn: sqlite3.Connection) -> None:
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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ratings (
            name_id INTEGER PRIMARY KEY REFERENCES names(id)
            ON DELETE CASCADE,
            rating REAL NOT NULL DEFAULT 1500.0,
            matches INTEGER DEFAULT 0,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS region_mapping (
            nationality TEXT PRIMARY KEY,
            region TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS source_versions (
            id INTEGER PRIMARY KEY,
            commit_hash TEXT NOT NULL,
            synced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS comparisons (
            id INTEGER PRIMARY KEY,
            name_a_id INTEGER NOT NULL REFERENCES names(id),
            name_b_id INTEGER NOT NULL REFERENCES names(id),
            preference INTEGER NOT NULL CHECK(preference IN (-1, 0, 1)),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(name_a_id, name_b_id, preference)
        )
    """)
    _migrate_comparisons_table_if_needed(conn)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feature_sets (
            id INTEGER PRIMARY KEY,
            version TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            feature_names_json TEXT NOT NULL,
            is_active BOOLEAN DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS name_features (
            name_id INTEGER NOT NULL REFERENCES names(id) ON DELETE CASCADE,
            feature_set_id INTEGER NOT NULL REFERENCES feature_sets(id) ON DELETE CASCADE,
            features_json TEXT NOT NULL,
            computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (name_id, feature_set_id)
        )
    """)


def _ensure_name_phonetic_columns(conn: sqlite3.Connection) -> None:
    cursor = conn.execute("PRAGMA table_info(names)")
    columns = [row[1] for row in cursor.fetchall()]
    if "phonetic_primary" not in columns:
        conn.execute("ALTER TABLE names ADD COLUMN phonetic_primary TEXT")
        logger.debug("Added phonetic_primary column")
    if "phonetic_secondary" not in columns:
        conn.execute("ALTER TABLE names ADD COLUMN phonetic_secondary TEXT")
        logger.debug("Added phonetic_secondary column")

    update_phonetic_codes(conn=conn)


def _ensure_indexes(conn: sqlite3.Connection) -> None:
    index_statements = (
        "CREATE INDEX IF NOT EXISTS idx_names_gender ON names(gender)",
        "CREATE INDEX IF NOT EXISTS idx_names_origin ON names(origin_region)",
        "CREATE INDEX IF NOT EXISTS idx_ratings_rating ON ratings(rating)",
        "CREATE INDEX IF NOT EXISTS idx_names_phonetic_primary ON names(phonetic_primary)",
        "CREATE INDEX IF NOT EXISTS idx_names_phonetic_secondary ON names(phonetic_secondary)",
        "CREATE INDEX IF NOT EXISTS idx_comparisons_name_a ON comparisons(name_a_id)",
        "CREATE INDEX IF NOT EXISTS idx_comparisons_name_b ON comparisons(name_b_id)",
        "CREATE INDEX IF NOT EXISTS idx_comparisons_created ON comparisons(created_at)",
        "CREATE INDEX IF NOT EXISTS idx_model_state_updated ON model_state(last_updated)",
        "CREATE INDEX IF NOT EXISTS idx_feature_sets_version ON feature_sets(version)",
        "CREATE INDEX IF NOT EXISTS idx_feature_sets_active ON feature_sets(is_active)",
        "CREATE INDEX IF NOT EXISTS idx_name_features_lookup ON name_features(name_id, feature_set_id)",
        "CREATE INDEX IF NOT EXISTS idx_name_features_computed ON name_features(computed_at)",
    )
    for statement in index_statements:
        conn.execute(statement)


def _insert_default_region_mapping(conn: sqlite3.Connection) -> None:
    """Insert default nationality -> region mapping."""
    region_store.insert_default_region_mapping(conn)


def _ensure_seed_region_mapping(conn: sqlite3.Connection) -> None:
    if conn.execute("SELECT COUNT(*) FROM region_mapping").fetchone()[0] == 0:
        _insert_default_region_mapping(conn)


def init_database() -> None:
    """Initialize database schema if it doesn't exist."""
    db_path = get_db_path()
    if _INIT_STATE["db_initialized"] and _INIT_STATE["db_path"] == db_path:
        logger.debug("Database already initialized, skipping initialization")
        return

    db_exists = db_path.exists()

    try:
        with get_connection() as conn:
            table_exists = False
            if db_exists:
                try:
                    conn.execute("SELECT 1 FROM names LIMIT 1").fetchone()
                    table_exists = True
                except sqlite3.OperationalError:
                    table_exists = False

            if not db_exists:
                logger.info("Creating new database at %s", db_path)
            elif not table_exists:
                logger.info("Initializing database schema for existing file")
            else:
                logger.info("Using existing database, ensuring schema is up to date")

            _ensure_schema(conn)
            _ensure_name_phonetic_columns(conn)
            _ensure_indexes(conn)
            _ensure_seed_region_mapping(conn)

            logger.info("Database schema verified successfully")
    except Exception:
        reset_database_init_state()
        raise
    else:
        _INIT_STATE["db_initialized"] = True
        _INIT_STATE["db_path"] = db_path


def get_unclassified_names(limit: int | None = None) -> list[UnclassifiedName]:
    """Get names that haven't been classified with origin region."""
    return name_store.get_unclassified_names(limit=limit)


def update_name_origin(name_id: int, region: str, confidence: float) -> None:
    """Update a name's origin region and confidence."""
    name_store.update_name_origin(name_id, region, confidence)


def get_names_with_origins(
    confidence_threshold: float = 0.5,
) -> dict[str, tuple[str, float, str, str]]:
    """Get known name -> (region, confidence, phonetic_primary, phonetic_secondary)."""
    return name_store.get_names_with_origins(confidence_threshold=confidence_threshold)


def get_names_by_filters(
    gender: str | None = None,
    origins: list[str] | None = None,
) -> list[str]:
    """Get names filtered by gender and origin regions."""
    return name_store.get_names_by_filters(gender=gender, origins=origins)


def get_names_by_gender() -> dict[str, list[str]]:
    """Get names categorized by gender."""
    return name_store.get_names_by_gender()


def get_all_origin_regions() -> list[str]:
    """Get distinct origin regions from names table, including NULL as International."""
    return name_store.get_all_origin_regions()


def get_name_details_batch(
    names: list[str],
) -> list[NameDetails]:
    """Get gender and origin_region for multiple names in batch."""
    return name_store.get_name_details_batch(names)


def get_phonetic_codes_batch(names: list[str]) -> dict[str, PhoneticCodes]:
    """Get phonetic codes for multiple names in batch."""
    return name_store.get_phonetic_codes_batch(names)


def get_ratings() -> dict[str, float]:
    """Get all ratings as name -> rating dictionary."""
    return ratings_store.get_ratings()


def update_rating(name: str, rating: float) -> list[str]:
    """Update one rating and return the skipped name when it is missing."""
    return ratings_store.update_rating(name, rating)


def update_ratings_batch(ratings_dict: dict[str, float]) -> list[str]:
    """Update multiple ratings in one transaction, incrementing match counts."""
    return ratings_store.update_ratings_batch(ratings_dict)


def update_ratings_batch_values(ratings_dict: dict[str, float]) -> list[str]:
    """Update multiple ratings without incrementing match counts."""
    return ratings_store.update_ratings_batch_values(ratings_dict)


def record_comparison(name_a: str, name_b: str, preference: int) -> None:
    """Record a pairwise comparison in the database."""
    ratings_store.record_comparison(name_a, name_b, preference)


def get_total_comparisons() -> int:
    """Get total number of recorded pairwise comparisons."""
    return ratings_store.get_total_comparisons()


def get_comparison_count(name: str) -> int:
    """Get number of comparisons involving a name."""
    return ratings_store.get_comparison_count(name)


def initialize_ratings(names: list[str]) -> dict[str, float]:
    """Initialize ratings for a list of names with the default score."""
    return ratings_store.initialize_ratings(names)


def save_user_setting(key: str, value: str) -> None:
    """Save a user setting."""
    settings_store.save_user_setting(key, value)


def load_user_setting(key: str, default: str = "") -> str:
    """Load a user setting."""
    return settings_store.load_user_setting(key, default=default)


def get_stats() -> DatabaseStats:
    """Get database statistics."""
    return stats_store.get_stats()


def get_preference_stats_by_gender() -> dict[str, PreferenceStats]:
    """Get preference statistics grouped by gender."""
    return preference_stats_store.get_preference_stats_by_gender()


def get_preference_stats_by_origin() -> dict[str, PreferenceStats]:
    """Get preference statistics grouped by origin region."""
    return preference_stats_store.get_preference_stats_by_origin()


def get_preference_stats_by_phonetic() -> dict[str, PreferenceStats]:
    """Get preference statistics grouped by phonetic primary code."""
    return preference_stats_store.get_preference_stats_by_phonetic()


if __name__ == "__main__":
    init_database()
    print("Database initialized successfully")
    stats = get_stats()
    print(f"Total names: {stats['total_names']}")
    print(f"Classified names: {stats['classified_names']}")
    print(f"Unclassified names: {stats['unclassified_names']}")
    print(f"Rated names: {stats['rated_names']}")
