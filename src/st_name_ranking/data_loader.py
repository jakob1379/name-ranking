"""Data loading and persistence functions."""

import logging
import re
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from st_name_ranking import database
from st_name_ranking.database import INITIAL_SCORE, initialize_ratings

logger = logging.getLogger(__name__)

# Constants for validation and logging
MIN_NAME_LENGTH = 2
MAX_INVALID_NAME_LOG = 5
LARGE_DATASET_THRESHOLD = 1000


def strip_name_notes(name: str) -> str:
    """Strip note suffixes from raw name strings.

    Example:
        "Matteos - variant af godkendt fornavn" -> "Matteos"
    """
    if not isinstance(name, str):
        return ""
    return name.split(" - ", 1)[0].strip()


def is_valid_name(name: str) -> bool:
    """Check if a string is a valid name (not a header or placeholder).
    Filters out strings like 'name1', 'Navn', 'name', etc.
    """
    if not name or not isinstance(name, str):
        return False

    name_lower = name.strip().lower()

    # Common header/placeholder patterns to exclude
    invalid_patterns = [
        "name",
        "navn",
        "fornavn",
        "firstname",
        "køn",
        "gender",
        "kjønn",
        "id",
        "nummer",
        "number",
        # Pattern like 'name1', 'name 1', 'navn1', etc.
        r"^name\s*\d+$",
        r"^navn\s*\d+$",
        r"^fornavn\s*\d+$",
    ]

    # Check exact matches
    if name_lower in [
        "name",
        "navn",
        "fornavn",
        "firstname",
        "køn",
        "gender",
        "kjønn",
    ]:
        return False

    # Check pattern matches

    for pattern in invalid_patterns[-3:]:  # The regex patterns
        if re.match(pattern, name_lower, re.IGNORECASE):
            return False

    # Name should have at least MIN_NAME_LENGTH characters
    return not len(name_lower) < MIN_NAME_LENGTH


def load_ratings() -> dict[str, float] | None:
    """Load saved ratings from database.
    Returns ratings dict or None if database not initialized.
    """
    try:
        database.init_database()
        return database.get_ratings()
    except sqlite3.Error as e:
        st.toast(
            f"Could not load ratings from database: {e}",
            icon="⚠️",
        )
        return None


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

            st.toast(
                f"Updated {len(ratings_to_save)} ratings in database",
                icon="ℹ️",
            )
    except sqlite3.Error as e:
        st.toast(
            f"Failed to save ratings to database: {e}",
            icon="❌",
            duration="long",
        )
        return False
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
        st.toast(
            f"Added {new_names_added} new names with initial rating {INITIAL_SCORE}",
            icon="ℹ️",
        )

    return ratings


def load_submodule_json() -> list[dict[str, str]]:
    """Load name-gender pairs from local git submodule JSON file.
    Returns list of dicts with 'name' and 'gender' keys.
    """
    json_path = Path("godkendtefornavne") / "allenavne.json"
    try:
        # Use pandas to read JSON for better performance and error handling
        df = pd.read_json(json_path, encoding="utf-8")

        # Ensure we have the expected columns
        if not all(col in df.columns for col in ["name", "gender"]):
            st.toast(
                f"JSON missing required columns. Found: {df.columns.tolist()}",
                icon="❌",
                duration="long",
            )
            return []

        # Validate structure and filter out invalid names
        valid_items = []
        invalid_count = 0

        for _, row in df.iterrows():
            name = strip_name_notes(str(row["name"]))
            gender = str(row["gender"]).strip()

            if is_valid_name(name):
                valid_items.append({"name": name, "gender": gender})
            else:
                invalid_count += 1
                if invalid_count <= MAX_INVALID_NAME_LOG:  # Log first few invalid names
                    st.toast(
                        f"Skipping invalid name entry: '{name}'",
                        icon="⚠️",
                    )

        if invalid_count > 0:
            st.toast(
                f"Filtered out {invalid_count} invalid name entries",
                icon="ℹ️",
            )

        st.toast(
            f"Loaded {len(valid_items)} name-gender pairs from JSON",
            icon="✅",
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        st.toast(
            f"Failed to load submodule JSON: {e}",
            icon="❌",
            duration="long",
        )
        return []
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
                st.toast(
                    f"Synced {inserted} new names from submodule to database",
                    icon="ℹ️",
                )
        else:
            # Check if database is empty and warn user
            with database.get_connection() as conn:
                count = conn.execute("SELECT COUNT(*) FROM names").fetchone()[0]
                if count == 0:
                    st.toast(
                        "Database empty. Click 'Sync Names' to load names.",
                        icon="⚠️",
                    )
                    return {}

        # Get names categorized by gender from database
        gender_data = database.get_names_by_gender()
        if not gender_data:
            st.toast(
                "No names found in database",
                icon="❌",
                duration="long",
            )
            return {}

        # Log counts (but only if we have many names to avoid toast spam)
        total_names = sum(len(names) for names in gender_data.values())
        if total_names > LARGE_DATASET_THRESHOLD:  # Only show toast for large datasets
            st.toast(
                f"Loaded {total_names} names from database",
                icon="✅",
            )

    except sqlite3.Error as e:
        st.toast(
            f"Failed to load names by gender from database: {e}",
            icon="❌",
            duration="long",
        )
        return {}
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
                                    st.toast(
                                        f"Skipping invalid CSV entry: '{name}'",
                                        icon="⚠️",
                                    )
            else:
                st.toast(
                    f"Submodule CSV file not found: {file_path}",
                    icon="⚠️",
                )

        if not all_names:
            st.toast(
                "No names found in submodule files",
                icon="❌",
                duration="long",
            )
            return []

        if invalid_count > 0:
            st.toast(
                f"Filtered out {invalid_count} invalid CSV entries",
                icon="ℹ️",
            )

        names = sorted(set(all_names))
        st.toast(
            f"Loaded {len(names)} names from CSV fallback",
            icon="✅",
        )
    except OSError as e:
        st.toast(
            f"Failed to load from CSV fallback: {e}",
            icon="❌",
            duration="long",
        )
        return []
    else:
        return names
