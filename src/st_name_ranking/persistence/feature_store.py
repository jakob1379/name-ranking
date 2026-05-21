"""Persistence helpers for feature-set and name-feature cache tables."""

import importlib
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, TypedDict

from st_name_ranking.persistence.connection import MAX_SQL_PARAMS, get_connection
from st_name_ranking.types import FeatureSetRecord, FeatureValues, ProgressCallback

logger = logging.getLogger(__name__)

MAX_SQLITE_TABLE_LOOKUP_NAMES = 2


@dataclass(frozen=True)
class FeatureCacheRebuildResult:
    """Summary of a feature-cache rebuild workflow."""

    version: str
    feature_names: list[str]
    feature_set_id: int
    processed: int
    deleted: int = 0


class FeatureSetCacheStats(TypedDict):
    """Coverage counters for one feature-set cache."""

    version: str
    is_active: bool
    cached_count: int
    missing_count: int
    coverage_pct: float


class FeatureCacheStats(TypedDict):
    """Aggregate feature-cache coverage counters."""

    total_names: int
    feature_sets: list[FeatureSetCacheStats]


class FeatureStatusStats(TypedDict):
    """Feature-cache status counters for CLI/status output."""

    feature_sets_count: int
    names_with_features: int
    active_version: str | None


class CorruptFeatureCacheError(RuntimeError):
    """Raised when cached feature JSON cannot be decoded."""

    def __init__(self, *, name_id: int, feature_set_id: int, cause: json.JSONDecodeError) -> None:
        self.name_id = name_id
        self.feature_set_id = feature_set_id
        super().__init__(
            f"Corrupt feature cache row for name_id={name_id}, feature_set_id={feature_set_id}: {cause.msg}",
        )


def get_or_create_feature_set(version: str, feature_names: list[str]) -> int:
    """Get a feature-set ID, creating the row when needed."""
    with get_connection() as conn:
        cursor = conn.execute(
            "SELECT id FROM feature_sets WHERE version = ?",
            (version,),
        )
        row = cursor.fetchone()
        if row:
            return row[0]

        cursor = conn.execute(
            """
            INSERT INTO feature_sets (version, feature_names_json, is_active)
            VALUES (?, ?, 1)
            """,
            (version, json.dumps(feature_names)),
        )
        logger.info("Created feature set version '%s' with %d features", version, len(feature_names))
        return cursor.lastrowid


def create_active_feature_set(version: str, feature_names: list[str]) -> int:
    """Create a new active feature-set row and deactivate older versions."""
    with get_connection() as conn:
        if not _table_exists(conn, "feature_sets"):
            msg = "Database not initialized. Run 'st-name-ranking db init' first."
            raise RuntimeError(msg)

        conn.execute("UPDATE feature_sets SET is_active = 0")
        cursor = conn.execute(
            """
            INSERT INTO feature_sets (version, feature_names_json, is_active)
            VALUES (?, ?, 1)
            """,
            (version, json.dumps(feature_names)),
        )
        return cursor.lastrowid


def rebuild_feature_cache(
    *,
    version: str | None = None,
    batch_size: int = 100,
    clear_existing: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> FeatureCacheRebuildResult:
    """Create an active feature set and cache current features for all names."""
    extractor = _new_feature_extractor()
    feature_names = extractor.get_feature_names()
    feature_set_version = version or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    deleted = clear_all_features() if clear_existing else 0
    feature_set_id = create_active_feature_set(feature_set_version, feature_names)
    processed = extract_and_cache_features(feature_set_id, batch_size=batch_size, progress_callback=progress_callback)

    return FeatureCacheRebuildResult(
        version=feature_set_version,
        feature_names=feature_names,
        feature_set_id=feature_set_id,
        processed=processed,
        deleted=deleted,
    )


def get_active_feature_set_version() -> str | None:
    """Get the currently active feature-set version."""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT version FROM feature_sets
            WHERE is_active = 1
            ORDER BY created_at DESC
            LIMIT 1
            """,
        )
        row = cursor.fetchone()
        return row[0] if row else None


def get_feature_set_by_version(version: str) -> FeatureSetRecord | None:
    """Get feature-set details by version."""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT id, version, feature_names_json, is_active, created_at
            FROM feature_sets
            WHERE version = ?
            """,
            (version,),
        )
        row = cursor.fetchone()
        if not row:
            return None

        feature_names = json.loads(row[2])
        return {
            "id": row[0],
            "version": row[1],
            "feature_names": [str(name) for name in feature_names],
            "is_active": bool(row[3]),
            "created_at": row[4],
        }


def get_cached_features_batch(
    name_ids: list[int],
    feature_set_id: int,
) -> dict[int, FeatureValues]:
    """Get cached feature dictionaries for multiple names."""
    if not name_ids:
        return {}

    result: dict[int, FeatureValues] = {}
    chunk_size = MAX_SQL_PARAMS // 2

    with get_connection() as conn:
        for i in range(0, len(name_ids), chunk_size):
            chunk = name_ids[i : i + chunk_size]
            placeholders = ", ".join(["?"] * len(chunk))
            query = f"""
                SELECT name_id, features_json
                FROM name_features
                WHERE feature_set_id = ? AND name_id IN ({placeholders})
            """
            cursor = conn.execute(query, [feature_set_id, *chunk])
            for row in cursor:
                result[row[0]] = _decode_features_json(
                    row[1],
                    name_id=row[0],
                    feature_set_id=feature_set_id,
                )

    return result


def get_cached_features(name_id: int, feature_set_id: int) -> FeatureValues | None:
    """Get cached features for a single name."""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT features_json FROM name_features
            WHERE name_id = ? AND feature_set_id = ?
            """,
            (name_id, feature_set_id),
        )
        row = cursor.fetchone()
        if not row:
            return None

        return _decode_features_json(
            row[0],
            name_id=name_id,
            feature_set_id=feature_set_id,
        )


def set_cached_features_batch(
    features_data: list[tuple[int, int, FeatureValues]],
) -> int:
    """Cache computed features for multiple names."""
    if not features_data:
        return 0

    inserted = 0
    chunk_size = MAX_SQL_PARAMS // 4

    with get_connection() as conn:
        for i in range(0, len(features_data), chunk_size):
            chunk = features_data[i : i + chunk_size]
            conn.executemany(
                """
                INSERT OR REPLACE INTO name_features
                (name_id, feature_set_id, features_json, computed_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """,
                [(name_id, fs_id, json.dumps(features)) for name_id, fs_id, features in chunk],
            )
            inserted += len(chunk)

    return inserted


def extract_and_cache_features(
    feature_set_id: int,
    batch_size: int = 100,
    progress_callback: ProgressCallback | None = None,
) -> int:
    """Extract features for all names and cache them for a feature set."""
    extractor = _new_feature_extractor()
    feature_names = extractor.get_feature_names()

    with get_connection() as conn:
        cursor = conn.execute("""
            SELECT id, name, gender, origin_region
            FROM names
            ORDER BY id
        """)
        names_data = cursor.fetchall()

    total = len(names_data)
    processed = 0

    for i in range(0, total, batch_size):
        batch = names_data[i : i + batch_size]
        features_data: list[tuple[int, int, FeatureValues]] = []

        for name_id, name, gender, origin_region in batch:
            features = extractor.extract(name, gender, origin_region)
            features_data.append(
                (
                    name_id,
                    feature_set_id,
                    {
                        feature_name: float(value)
                        for feature_name, value in zip(feature_names, features.tolist(), strict=True)
                    },
                ),
            )

        set_cached_features_batch(features_data)
        processed += len(batch)
        if progress_callback:
            progress_callback(processed, total)

    return processed


def clear_all_features() -> int:
    """Clear all cached name features and return the number of deleted rows."""
    with get_connection() as conn:
        if not _table_exists(conn, "name_features"):
            return 0
        cursor = conn.execute("DELETE FROM name_features")
        return cursor.rowcount


def has_feature_cache() -> bool:
    """Return whether feature tables exist and contain cached rows."""
    try:
        with get_connection() as conn:
            existing_tables = _existing_tables(conn, {"feature_sets", "name_features"})
            if "feature_sets" not in existing_tables or "name_features" not in existing_tables:
                return False

            cursor = conn.execute("SELECT COUNT(*) FROM name_features LIMIT 1")
            return cursor.fetchone()[0] > 0
    except sqlite3.OperationalError:
        logger.exception("Failed to inspect feature cache")
        return False


def get_feature_stats() -> FeatureStatusStats:
    """Get feature-cache summary statistics for CLI/status output."""
    with get_connection() as conn:
        existing_tables = _existing_tables(conn, {"feature_sets", "name_features"})
        if "feature_sets" not in existing_tables or "name_features" not in existing_tables:
            return {
                "feature_sets_count": 0,
                "names_with_features": 0,
                "active_version": None,
            }

        feature_sets_count = conn.execute("SELECT COUNT(*) FROM feature_sets").fetchone()[0]
        names_with_features = conn.execute("SELECT COUNT(DISTINCT name_id) FROM name_features").fetchone()[0]
        row = conn.execute(
            """
            SELECT version FROM feature_sets
            WHERE is_active = 1
            ORDER BY created_at DESC
            LIMIT 1
            """,
        ).fetchone()

        return {
            "feature_sets_count": feature_sets_count,
            "names_with_features": names_with_features,
            "active_version": row[0] if row else None,
        }


def _existing_tables(conn: sqlite3.Connection, table_names: set[str]) -> set[str]:
    sorted_names = tuple(sorted(table_names))
    if not sorted_names:
        return set()

    if len(sorted_names) == 1:
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name = ?"
    elif len(sorted_names) == MAX_SQLITE_TABLE_LOOKUP_NAMES:
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name IN (?, ?)"
    else:
        msg = "_existing_tables supports at most two table names"
        raise ValueError(msg)

    cursor = conn.execute(query, sorted_names)
    return {row[0] for row in cursor.fetchall()}


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    return cursor.fetchone() is not None


def _new_feature_extractor() -> Any:
    features_module = importlib.import_module("st_name_ranking.learning.features")
    return features_module.FeatureExtractor()


def _decode_features_json(features_json: str, *, name_id: int, feature_set_id: int) -> FeatureValues:
    try:
        decoded: dict[str, Any] = json.loads(features_json)
    except json.JSONDecodeError as e:
        logger.exception(
            "Corrupt JSON in name_features for name_id=%s, feature_set_id=%s",
            name_id,
            feature_set_id,
        )
        raise CorruptFeatureCacheError(name_id=name_id, feature_set_id=feature_set_id, cause=e) from e
    return {str(name): float(value) for name, value in decoded.items()}


def set_cached_features(
    name_id: int,
    feature_set_id: int,
    features: FeatureValues,
) -> None:
    """Cache computed features for a single name."""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO name_features
            (name_id, feature_set_id, features_json, computed_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (name_id, feature_set_id, json.dumps(features)),
        )


def is_features_computed(name_id: int, feature_set_id: int) -> bool:
    """Return whether a cache row exists for a name and feature set."""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT 1 FROM name_features
            WHERE name_id = ? AND feature_set_id = ?
            """,
            (name_id, feature_set_id),
        )
        return cursor.fetchone() is not None


def get_feature_cache_stats() -> FeatureCacheStats:
    """Get aggregate feature-cache coverage statistics."""
    with get_connection() as conn:
        total_names = conn.execute("SELECT COUNT(*) FROM names").fetchone()[0]

        cursor = conn.execute("""
            SELECT fs.version, fs.is_active, COUNT(nf.name_id) as cached_count
            FROM feature_sets fs
            LEFT JOIN name_features nf ON fs.id = nf.feature_set_id
            GROUP BY fs.id
            ORDER BY fs.created_at DESC
        """)

        feature_set_stats: list[FeatureSetCacheStats] = []
        for row in cursor:
            cached = row[2]
            feature_set_stats.append(
                {
                    "version": row[0],
                    "is_active": bool(row[1]),
                    "cached_count": cached,
                    "missing_count": total_names - cached,
                    "coverage_pct": (cached / total_names * 100) if total_names > 0 else 0,
                },
            )

        return {
            "total_names": total_names,
            "feature_sets": feature_set_stats,
        }
