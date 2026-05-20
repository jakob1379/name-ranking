"""Data loading and persistence functions."""

import logging
import sqlite3
from pathlib import Path

import polars as pl

from st_name_ranking.name_normalization import is_valid_name, strip_name_notes
from st_name_ranking.persistence import database
from st_name_ranking.persistence.database import INITIAL_SCORE, initialize_ratings

logger = logging.getLogger(__name__)

# Constants for validation and logging
MAX_INVALID_NAME_LOG = 5
LARGE_DATASET_THRESHOLD = 1000


class DataLoaderError(RuntimeError):
    """Base class for data-loading failures."""


class DatabaseLoadError(DataLoaderError):
    """Raised when database-backed data cannot be loaded or saved."""


class NameDataLoadError(DataLoaderError):
    """Raised when local name data files cannot be loaded."""


class InvalidNameDataSchemaError(NameDataLoadError):
    """Raised when a name data file is missing required columns."""


def load_ratings() -> dict[str, float]:
    """Load saved ratings from database.
    Raises DatabaseLoadError if database access fails.
    """
    try:
        database.init_database()
        return database.get_ratings()
    except sqlite3.Error as e:
        msg = f"Could not load ratings from database: {e}"
        raise DatabaseLoadError(msg) from e


def save_ratings(
    ratings: dict[str, float],
    names_to_update: list[str] | None = None,
) -> bool:
    """Save ratings to database.

    Args:
        ratings: Dictionary mapping name -> rating
        names_to_update: Optional list of names to update. If None, updates all.

    Returns:
        True if successful.

    """
    try:
        database.init_database()

        # Filter ratings to update only specified names
        if names_to_update is not None:
            ratings_to_save = {name: rating for name, rating in ratings.items() if name in names_to_update}
        else:
            ratings_to_save = ratings

        if ratings_to_save:
            # Use batch update for efficiency
            database.update_ratings_batch(ratings_to_save)

            logger.info("Updated %d ratings in database", len(ratings_to_save))
    except sqlite3.Error as e:
        msg = f"Failed to save ratings to database: {e}"
        raise DatabaseLoadError(msg) from e
    else:
        return True


def initialize_or_load_ratings(names: list[str]) -> dict[str, float]:
    """Initialize ratings for names, loading existing ratings from file.
    Merges saved ratings with new names (new names get INITIAL_SCORE).
    """
    saved = load_ratings()
    if saved is None:
        # No saved ratings, initialize fresh
        return initialize_ratings(names)

    # Merge: use saved ratings for existing names, initialize new names
    ratings = saved.copy()
    new_names_added = 0
    for name in names:
        if name not in ratings:
            ratings[name] = INITIAL_SCORE
            new_names_added += 1

    if new_names_added > 0:
        logger.info("Added %d new names with initial rating %s", new_names_added, INITIAL_SCORE)

    return ratings


def load_submodule_json() -> list[dict[str, str]]:
    """Load name-gender pairs from local git submodule JSON file.
    Returns list of dicts with 'name' and 'gender' keys.
    """
    json_path = Path("godkendtefornavne") / "allenavne.json"
    try:
        df = pl.read_json(json_path)

        # Ensure we have the expected columns
        if not all(col in df.columns for col in ["name", "gender"]):
            msg = f"JSON missing required columns. Found: {list(df.columns)}"
            raise InvalidNameDataSchemaError(msg)

        # Validate structure and filter out invalid names
        valid_items = []
        invalid_count = 0

        for row in df.iter_rows(named=True):
            name = strip_name_notes(str(row.get("name", "")))
            gender = str(row.get("gender", "")).strip()

            if is_valid_name(name):
                valid_items.append({"name": name, "gender": gender})
            else:
                invalid_count += 1
                if invalid_count <= MAX_INVALID_NAME_LOG:  # Log first few invalid names
                    logger.warning("Skipping invalid name entry: %r", name)

        if invalid_count > 0:
            logger.info("Filtered out %d invalid name entries", invalid_count)

        logger.info("Loaded %d name-gender pairs from JSON", len(valid_items))
    except (FileNotFoundError, ValueError, RuntimeError, pl.exceptions.PolarsError) as e:
        msg = f"Failed to load submodule JSON: {e}"
        raise NameDataLoadError(msg) from e
    else:
        return valid_items


def load_names_by_gender(
    *,
    sync_with_submodule: bool = False,
) -> dict[str, list[str]]:
    """Load names from database, categorized by gender.
    Unisex names are included in both 'Male' and 'Female' categories.

    Args:
        sync_with_submodule: If True, sync with submodule before loading.
                             Default False for faster startup.

    """
    try:
        database.init_database()

        # Only sync if explicitly requested
        if sync_with_submodule:
            inserted = database.sync_names_with_submodule()
            if inserted > 0:
                logger.info("Synced %d new names from submodule to database", inserted)
        else:
            # Check if database is empty and warn user
            with database.get_connection() as conn:
                count = conn.execute("SELECT COUNT(*) FROM names").fetchone()[0]
                if count == 0:
                    logger.info("Database empty; no names loaded")
                    return {}

        # Get names categorized by gender from database
        gender_data = database.get_names_by_gender()
        if not gender_data:
            logger.info("No names found in database")
            return {}

        # Log counts (but only if we have many names to avoid toast spam)
        total_names = sum(len(names) for names in gender_data.values())
        if total_names > LARGE_DATASET_THRESHOLD:  # Only show toast for large datasets
            logger.info("Loaded %d names from database", total_names)

    except sqlite3.Error as e:
        msg = f"Failed to load names by gender from database: {e}"
        raise DatabaseLoadError(msg) from e
    else:
        return gender_data


def load_submodule_csv_fallback() -> list[str]:
    """Fallback: Load names from local git submodule CSV files.
    Used when JSON is not available.
    """
    submodule_path = Path("godkendtefornavne")
    csv_files = ["drengenavne.csv", "pigenavne.csv", "unisexnavne.csv"]

    all_names = []
    invalid_count = 0
    try:
        for csv_file in csv_files:
            file_path = submodule_path / csv_file
            if file_path.exists():
                with file_path.open(encoding="utf-8") as f:
                    for line in f:
                        name = strip_name_notes(line)
                        if name:  # Skip empty lines
                            if is_valid_name(name):
                                all_names.append(name)
                            else:
                                invalid_count += 1
                                if invalid_count <= MAX_INVALID_NAME_LOG:
                                    logger.warning("Skipping invalid CSV entry: %r", name)
            else:
                logger.warning("Submodule CSV file not found: %s", file_path)

        if not all_names:
            msg = "No names found in submodule files"
            raise NameDataLoadError(msg)

        if invalid_count > 0:
            logger.info("Filtered out %d invalid CSV entries", invalid_count)

        names = sorted(set(all_names))
        logger.info("Loaded %d names from CSV fallback", len(names))
    except OSError as e:
        msg = f"Failed to load from CSV fallback: {e}"
        raise NameDataLoadError(msg) from e
    else:
        return names
