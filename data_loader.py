"""
Data loading and persistence functions.
"""

import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional

import httpx
import pandas as pd
import streamlit as st

from elo import INITIAL_RATING, initialize_ratings
import database


def is_valid_name(name: str) -> bool:
    """
    Check if a string is a valid name (not a header or placeholder).
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

    # Name should have at least 2 characters
    if len(name_lower) < 2:
        return False

    return True


def load_ratings(file_path: str = "ratings.json") -> Optional[Dict[str, float]]:
    """
    Load saved ratings from database.
    Returns ratings dict or None if database not initialized.
    """
    try:
        database.init_database()
        return database.get_ratings()
    except Exception as e:
        st.toast(
            f"Could not load ratings from database: {e}",
            icon="⚠️",
        )
        return None


def save_ratings(
    ratings: Dict[str, float], file_path: str = "ratings.json"
) -> bool:
    """
    Save ratings to database.
    Returns True if successful.
    """
    try:
        database.init_database()
        updated = 0
        for name, rating in ratings.items():
            try:
                database.update_rating(name, rating)
                updated += 1
            except Exception as e:
                st.toast(
                    f"Failed to update rating for {name}: {e}",
                    icon="⚠️",
                )
                # Continue with other ratings
        
        if updated > 0:
            st.toast(
                f"Updated {updated} ratings in database",
                icon="ℹ️",
            )
        return True
    except Exception as e:
        st.toast(
            f"Failed to save ratings to database: {e}",
            icon="❌",
            duration="long",
        )
        return False


def initialize_or_load_ratings(names: List[str]) -> Dict[str, float]:
    """
    Initialize ratings for names, loading existing ratings from file.
    Merges saved ratings with new names (new names get INITIAL_RATING).
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
            ratings[name] = INITIAL_RATING
            new_names_added += 1

    if new_names_added > 0:
        st.toast(
            f"Added {new_names_added} new names with initial rating "
            f"{INITIAL_RATING}",
            icon="ℹ️",
        )

    return ratings


def load_submodule_json() -> List[Dict[str, str]]:
    """
    Load name-gender pairs from local git submodule JSON file.
    Returns list of dicts with 'name' and 'gender' keys.
    """
    json_path = os.path.join("godkendtefornavne", "allenavne.json")
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
            name = str(row["name"]).strip()
            gender = str(row["gender"]).strip()

            if is_valid_name(name):
                valid_items.append({"name": name, "gender": gender})
            else:
                invalid_count += 1
                if invalid_count <= 5:  # Log first few invalid names
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
        return valid_items
    except Exception as e:
        st.toast(
            f"Failed to load submodule JSON: {e}",
            icon="❌",
            duration="long",
        )
        return []


def load_local_csv(csv_path: str = "alle-navne.csv") -> List[str]:
    """
    Load names from local CSV file (Navn column).
    Returns list of names.
    """
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        if "Navn" not in df.columns:
            st.toast(
                "CSV file missing 'Navn' column. "
                f"Columns found: {df.columns.tolist()}",
                icon="❌",
                duration="long",
            )
            return []
        names = df["Navn"].astype(str).str.strip().tolist()
        # Filter out empty strings and deduplicate
        names = sorted(list(set([name for name in names if name])))
        st.toast(
            f"Loaded {len(names)} names from local CSV",
            icon="✅",
        )
        return names
    except Exception as e:
        st.toast(
            f"Failed to load local CSV: {e}",
            icon="❌",
            duration="long",
        )
        return []


def load_names_by_gender() -> Dict[str, List[str]]:
    """
    Load names from database, categorized by gender.
    Returns dict mapping gender to list of names.
    Unisex names are included in both 'Male' and 'Female' categories.
    """
    try:
        # Ensure database is initialized and synced with submodule
        database.init_database()
        inserted = database.sync_names_with_submodule()
        if inserted > 0:
            st.toast(
                f"Synced {inserted} new names from submodule to database",
                icon="ℹ️",
            )
        
        # Get names categorized by gender from database
        gender_data = database.get_names_by_gender()
        if not gender_data:
            st.toast(
                "No names found in database",
                icon="❌",
                duration="long",
            )
            return {}
        
        # Log counts
        for gender, names in gender_data.items():
            st.toast(
                f"Loaded {len(names)} {gender.lower()} names",
                icon="ℹ️",
            )

        return gender_data
    except Exception as e:
        st.toast(
            f"Failed to load names by gender from database: {e}",
            icon="❌",
            duration="long",
        )
        return {}


def load_submodule_names(gender: str = "All") -> List[str]:
    """
    Load names from local git submodule JSON file.
    Returns list of names for the specified gender.
    gender: "Male", "Female", "Unisex", or "All"
    """
    try:
        gender_data = load_names_by_gender()
        if not gender_data:
            return []

        if gender not in gender_data:
            st.toast(
                f"Unknown gender filter: {gender}. Using 'All'",
                icon="⚠️",
            )
            gender = "All"

        names = gender_data.get(gender, [])
        st.toast(
            f"Loaded {len(names)} names for {gender}",
            icon="✅",
        )
        return names
    except Exception as e:
        st.toast(
            f"Failed to load from submodule JSON: {e}",
            icon="❌",
            duration="long",
        )
        # Fallback to CSV files for compatibility
        return load_submodule_csv_fallback()


def load_submodule_csv_fallback() -> List[str]:
    """
    Fallback: Load names from local git submodule CSV files.
    Used when JSON is not available.
    """
    submodule_path = "godkendtefornavne"
    csv_files = ["drengenavne.csv", "pigenavne.csv", "unisexnavne.csv"]

    all_names = []
    invalid_count = 0
    try:
        for csv_file in csv_files:
            file_path = os.path.join(submodule_path, csv_file)
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        name = line.strip()
                        if name:  # Skip empty lines
                            if is_valid_name(name):
                                all_names.append(name)
                            else:
                                invalid_count += 1
                                if (
                                    invalid_count <= 5
                                ):  # Log first few invalid names
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

        names = sorted(list(set(all_names)))
        st.toast(
            f"Loaded {len(names)} names from CSV fallback",
            icon="✅",
        )
        return names
    except Exception as e:
        st.toast(
            f"Failed to load from CSV fallback: {e}",
            icon="❌",
            duration="long",
        )
        return []


def fetch_danish_names() -> List[str]:
    """
    Attempts to fetch the approved names from the mirror repository.
    """
    urls = [
        "https://koldfront.dk/git/godkendtefornavne/plain/pigenavne.txt",
        "https://koldfront.dk/git/godkendtefornavne/plain/drengenavne.txt",
    ]

    all_names = []
    try:
        with httpx.Client(timeout=10.0) as client:
            for url in urls:
                response = client.get(url)
                response.raise_for_status()
                # Split by newline and filter
                names = [
                    line.strip()
                    for line in response.text.splitlines()
                    if line.strip()
                ]
                all_names.extend(names)
        return sorted(list(set(all_names)))
    except Exception as e:
        st.toast(
            f"Failed to fetch remote names: {e}",
            icon="❌",
            duration="long",
        )
        return []


def fetch_koldfront_csv() -> List[str]:
    """
    Fetch names from Koldfront CSV files (boy/girl/unisex).
    Returns list of names.
    """
    urls = [
        "https://koldfront.dk/git/godkendtefornavne/plain/drengenavne.csv",
        "https://koldfront.dk/git/godkendtefornavne/plain/pigenavne.csv",
        "https://koldfront.dk/git/godkendtefornavne/plain/unisexnavne.csv",
    ]

    all_names = []
    try:
        with httpx.Client(timeout=15.0) as client:
            for url in urls:
                response = client.get(url)
                response.raise_for_status()
                # CSV files have one name per line (no header)
                names = [
                    line.strip()
                    for line in response.text.splitlines()
                    if line.strip()
                ]
                all_names.extend(names)
        names = sorted(list(set(all_names)))
        st.toast(
            f"Fetched {len(names)} names from Koldfront",
            icon="✅",
        )
        return names
    except Exception as e:
        st.toast(
            f"Failed to fetch from Koldfront: {e}",
            icon="❌",
            duration="long",
        )
        return []
