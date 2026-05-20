"""Submodule sync persistence for approved-name data."""

import logging
import shutil
import subprocess
from pathlib import Path

from metaphone import doublemetaphone

from st_name_ranking.name_normalization import is_valid_name, strip_name_notes
from st_name_ranking.persistence import database
from st_name_ranking.types import SourceVersion

logger = logging.getLogger(__name__)


def sync_names_with_submodule(submodule_path: Path = Path("godkendtefornavne")) -> int:
    """Sync names from the approved-names submodule into the database."""
    logger.debug("Syncing names from submodule")
    json_path = submodule_path / "allenavne.json"
    if not json_path.exists():
        _msg = f"Submodule JSON not found: {json_path}"
        raise FileNotFoundError(_msg)

    current_commit = _current_submodule_commit(submodule_path)
    with database.get_connection() as conn:
        last_sync = conn.execute(
            "SELECT commit_hash FROM source_versions ORDER BY id DESC LIMIT 1",
        ).fetchone()

        if last_sync and last_sync[0] == current_commit:
            logger.debug("Already synced with current commit")
            return 0

    valid_names = _load_valid_names(json_path)
    logger.debug("Filtered %d valid names", len(valid_names))

    inserted_count = 0
    with database.get_connection() as conn:
        if valid_names:
            before = conn.total_changes
            conn.executemany(
                "INSERT OR IGNORE INTO names (name, gender, phonetic_primary, phonetic_secondary) VALUES (?, ?, ?, ?)",
                valid_names,
            )
            inserted_count = conn.total_changes - before
            logger.debug("Bulk insert attempted, %d new rows", inserted_count)

        conn.execute(
            "INSERT INTO source_versions (commit_hash) VALUES (?)",
            (current_commit,),
        )

    logger.info("Inserted %d new names", inserted_count)
    return inserted_count


def get_latest_submodule_version() -> SourceVersion | None:
    """Get the latest synced submodule commit."""
    with database.get_connection() as conn:
        cursor = conn.execute(
            "SELECT commit_hash FROM source_versions ORDER BY id DESC LIMIT 1",
        )
        row = cursor.fetchone()
        if row:
            return SourceVersion(commit_hash=row[0])
        return None


def update_submodule_version(commit_hash: str) -> None:
    """Record a source-version commit hash."""
    with database.get_connection() as conn:
        conn.execute(
            "INSERT INTO source_versions (commit_hash) VALUES (?)",
            (commit_hash,),
        )


def _current_submodule_commit(submodule_path: Path) -> str:
    try:
        git_executable = shutil.which("git")
        if not git_executable:
            msg = "Git executable not found"
            raise RuntimeError(msg)

        result = subprocess.run(  # noqa: S603
            [git_executable, "-C", str(submodule_path), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except subprocess.SubprocessError as e:
        _msg = f"Failed to get submodule commit hash: {e}"
        raise RuntimeError(_msg) from e
    else:
        current_commit = result.stdout.strip()
        logger.debug("Submodule commit hash: %s", current_commit)
        return current_commit


def _load_valid_names(json_path: Path) -> list[tuple[str, str, str, str]]:
    import polars as pl  # noqa: PLC0415

    df = pl.read_json(json_path)
    logger.info("Loaded %d rows from JSON", df.height)

    if df.is_empty():
        logger.debug("Empty JSON, nothing to sync")
        return []

    if not all(col in df.columns for col in ["name", "gender"]):
        _msg = "JSON missing required columns 'name' and/or 'gender'"
        raise ValueError(_msg)

    valid_names = []
    for row in df.iter_rows(named=True):
        name = strip_name_notes(str(row.get("name", "")))
        gender_raw = str(row.get("gender", "")).strip()
        gender = _normalize_gender(gender_raw)
        if not gender:
            logger.warning(
                "Invalid gender '%s' for name '%s', skipping",
                gender_raw,
                name,
            )
            continue
        if is_valid_name(name):
            primary, secondary = doublemetaphone(name)
            valid_names.append((name, gender, primary or "", secondary or ""))

    return valid_names


def _normalize_gender(gender_raw: str) -> str | None:
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
    return gender_map.get(gender_raw) or gender_map.get(gender_raw.lower())
