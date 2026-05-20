"""Persistence helpers for feature-set and name-feature cache tables."""

import json
import logging
from typing import Any

from st_name_ranking.persistence import database
from st_name_ranking.types import FeatureSetRecord, FeatureValues

logger = logging.getLogger(__name__)


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
    with database.get_connection() as conn:
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


def get_active_feature_set_version() -> str | None:
    """Get the currently active feature-set version."""
    with database.get_connection() as conn:
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
    with database.get_connection() as conn:
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
    chunk_size = database.MAX_SQL_PARAMS // 2

    with database.get_connection() as conn:
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
    with database.get_connection() as conn:
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
    chunk_size = database.MAX_SQL_PARAMS // 4

    with database.get_connection() as conn:
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
    with database.get_connection() as conn:
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
    with database.get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT 1 FROM name_features
            WHERE name_id = ? AND feature_set_id = ?
            """,
            (name_id, feature_set_id),
        )
        return cursor.fetchone() is not None


def get_feature_cache_stats() -> dict:
    """Get aggregate feature-cache coverage statistics."""
    with database.get_connection() as conn:
        total_names = conn.execute("SELECT COUNT(*) FROM names").fetchone()[0]

        cursor = conn.execute("""
            SELECT fs.version, fs.is_active, COUNT(nf.name_id) as cached_count
            FROM feature_sets fs
            LEFT JOIN name_features nf ON fs.id = nf.feature_set_id
            GROUP BY fs.id
            ORDER BY fs.created_at DESC
        """)

        feature_set_stats = []
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
