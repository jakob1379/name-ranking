import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import httpx
import numpy as np
import pandas as pd
import streamlit as st
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer

# Add parent directory to path for database module
sys.path.insert(0, str(Path(__file__).parent))
import database

K_FACTOR: float = 32.0
INITIAL_RATING: float = 1500.0

# -----------------------------------------------------------------------------
# Data Validation Helpers
# -----------------------------------------------------------------------------


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
    import re

    for pattern in invalid_patterns[-3:]:  # The regex patterns
        if re.match(pattern, name_lower, re.IGNORECASE):
            return False

    # Name should have at least 2 characters
    if len(name_lower) < 2:
        return False

    return True


# -----------------------------------------------------------------------------
# Display Helpers
# -----------------------------------------------------------------------------


def display_name_with_rating(
    name: str, rating: float, delta: Optional[Union[int, float, str]] = None
) -> None:
    """
    Display name much larger than rating using st.metric with custom styling.
    CSS is injected in render_tournament to make the value (name) larger
    and label (rating) smaller.
    delta: difference in Elo compared to opponent (positive if higher,
    negative if lower).
    """
    # Format delta for display
    if delta is not None:
        # Convert numeric delta to string with sign
        if isinstance(delta, (int, float)):
            delta_str = f"{delta:+.0f}"
        else:
            delta_str = str(delta)
    else:
        delta_str = None

    # Use st.metric with swapped label/value to get desired visual hierarchy
    # Value will be large (name), label will be smaller (rating)
    st.metric(value=name, label=f"{rating:.0f}", delta=delta_str, border=True)


# -----------------------------------------------------------------------------
# Core Logic: Elo & Math
# -----------------------------------------------------------------------------


def expected_score(rating_a: float, rating_b: float) -> float:
    """
    Calculate expected score for A.
    Formula: E_A = 1 / (1 + 10 ^ ((R_B - R_A) / 400))
    """
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_elo_and_save(
    ratings: Dict[str, float], winner: str, loser: str, k: float = K_FACTOR
) -> Dict[str, float]:
    """
    Update Elo ratings and save in background.
    """
    updated_ratings = update_elo(ratings, winner, loser, k)

    # Save in background (non-blocking)
    try:
        # Use a simple thread or just save synchronously for now
        # In production, we might use asyncio or a proper background task
        save_ratings(updated_ratings)
    except Exception as e:
        # Don't break the UI if save fails
        st.warning(f"Failed to save ratings: {e}")

    return updated_ratings


def update_elo_draw_and_save(
    ratings: Dict[str, float], player_a: str, player_b: str, k: float = K_FACTOR
) -> Dict[str, float]:
    """
    Update Elo ratings for a draw and save in background.
    """
    updated_ratings = update_elo_draw(ratings, player_a, player_b, k)

    # Save in background (non-blocking)
    try:
        save_ratings(updated_ratings)
    except Exception as e:
        st.warning(f"Failed to save ratings: {e}")

    return updated_ratings


def update_elo_generic(
    ratings: Dict[str, float],
    player_a: str,
    player_b: str,
    score_a: float,
    score_b: float,
    k: float = K_FACTOR,
) -> Dict[str, float]:
    """
    Generic Elo update for any outcome.
    score_a is the result for player_a (typically 1.0 for win,
    0.5 for draw, 0.0 for loss).
    """
    if player_a not in ratings or player_b not in ratings:
        return ratings

    r_a = ratings[player_a]
    r_b = ratings[player_b]

    e_a = expected_score(r_a, r_b)
    e_b = 1.0 - e_a

    ratings[player_a] = r_a + k * (score_a - e_a)
    ratings[player_b] = r_b + k * (score_b - e_b)

    return ratings


def update_elo(
    ratings: Dict[str, float], winner: str, loser: str, k: float = K_FACTOR
) -> Dict[str, float]:
    """
    Update Elo ratings based on a binary outcome (1 for winner, 0 for loser).
    """
    return update_elo_generic(ratings, winner, loser, 1.0, 0.0, k)


def update_elo_draw(
    ratings: Dict[str, float], player_a: str, player_b: str, k: float = K_FACTOR
) -> Dict[str, float]:
    """
    Update Elo ratings for a draw (0.5 points each).
    """
    return update_elo_generic(ratings, player_a, player_b, 0.5, 0.5, k)


def initialize_ratings(names: List[str]) -> Dict[str, float]:
    return {name.strip(): INITIAL_RATING for name in names if name.strip()}


# -----------------------------------------------------------------------------
# Similarity Logic: String & Vector
# -----------------------------------------------------------------------------


def get_string_similarity_scores(
    target: str, candidates: List[str], limit: int = 10
) -> List[Tuple[str, float]]:
    """
    Uses RapidFuzz (Levenshtein) to find similar names.
    Returns list of (name, score).
    """
    if not candidates:
        return []

    # process.extract returns (match, score, index)
    results = process.extract(
        target, candidates, scorer=fuzz.ratio, limit=limit
    )
    return [(item[0], item[1]) for item in results]


@st.cache_resource
def load_embedding_model() -> SentenceTransformer:
    # 'paraphrase-multilingual-MiniLM-L12-v2' is good for Danish/English
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def get_vector_similarity_scores(
    model: SentenceTransformer,
    target: str,
    candidates: List[str],
    limit: int = 10,
) -> List[Tuple[str, float]]:
    """
    Uses LLM embeddings to find semantic similarity.
    Returns list of (name, score).
    """
    if not candidates:
        return []

    # Encode target and all candidates
    target_embedding = model.encode([target])
    candidate_embeddings = model.encode(candidates)

    # Compute Cosine Similarity
    # (N, 1) dot (1, N) -> (1, N)
    scores = np.dot(candidate_embeddings, target_embedding.T).flatten()

    # Get indices of top scores
    top_indices = np.argsort(scores)[::-1][:limit]

    return [(candidates[i], float(scores[i])) for i in top_indices]


# -----------------------------------------------------------------------------
# Data Source
# -----------------------------------------------------------------------------


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
        st.error(f"Failed to fetch remote names: {e}")
        return []


def load_local_csv(csv_path: str = "alle-navne.csv") -> List[str]:
    """
    Load names from local CSV file (Navn column).
    Returns list of names.
    """
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        if "Navn" not in df.columns:
            st.error(
                "CSV file missing 'Navn' column. "
                f"Columns found: {df.columns.tolist()}"
            )
            return []
        names = df["Navn"].astype(str).str.strip().tolist()
        # Filter out empty strings and deduplicate
        names = sorted(list(set([name for name in names if name])))
        st.success(f"Loaded {len(names)} names from local CSV")
        return names
    except Exception as e:
        st.error(f"Failed to load local CSV: {e}")
        return []


def load_names_by_gender() -> Dict[str, List[str]]:
    """
    Load names from database, categorized by gender.
    Returns dict mapping gender to list of names.
    Unisex names are included in both 'Male' and 'Female' categories.
    """
    try:
        # Initialize database if needed
        database.init_database()
        
        # Query all names with gender from database
        with database.get_connection() as conn:
            cursor = conn.execute("SELECT name, gender FROM names")
            rows = cursor.fetchall()
        
        if not rows:
            st.warning("No names found in database. Syncing with submodule...")
            inserted = database.sync_names_with_submodule()
            if inserted > 0:
                st.info(f"Synced {inserted} names from submodule")
                # Query again
                with database.get_connection() as conn:
                    cursor = conn.execute("SELECT name, gender FROM names")
                    rows = cursor.fetchall()
            else:
                st.error("Failed to sync names from submodule")
                return {}
        
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
        
        # Log counts
        for gender, names in result.items():
            st.info(f"Loaded {len(names)} {gender.lower()} names")
        
        return result
    except Exception as e:
        st.error(f"Failed to load names from database: {e}")
        return {}


def get_filtered_names(
    gender: Optional[str] = None, 
    origins: Optional[List[str]] = None
) -> List[str]:
    """
    Get names filtered by gender and origin regions.
    Returns list of names.
    """
    try:
        database.init_database()
        return database.get_names_by_filters(gender, origins)
    except Exception as e:
        st.error(f"Failed to get filtered names: {e}")
        return []


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
            st.warning(f"Unknown gender filter: {gender}. Using 'All'")
            gender = "All"

        names = gender_data.get(gender, [])
        st.success(f"Loaded {len(names)} names for {gender}")
        return names
    except Exception as e:
        st.error(f"Failed to load from submodule JSON: {e}")
        # Fallback to CSV files for compatibility
        return load_submodule_csv_fallback()


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
            st.error(
                f"JSON missing required columns. Found: {df.columns.tolist()}"
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
                    st.warning(f"Skipping invalid name entry: '{name}'")

        if invalid_count > 0:
            st.info(f"Filtered out {invalid_count} invalid name entries")

        st.success(f"Loaded {len(valid_items)} name-gender pairs from JSON")
        return valid_items
    except Exception as e:
        st.error(f"Failed to load submodule JSON: {e}")
        return []


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
                                    st.warning(
                                        f"Skipping invalid CSV entry: '{name}'"
                                    )
            else:
                st.warning(f"Submodule CSV file not found: {file_path}")

        if not all_names:
            st.error("No names found in submodule files")
            return []

        if invalid_count > 0:
            st.info(f"Filtered out {invalid_count} invalid CSV entries")

        names = sorted(list(set(all_names)))
        st.success(f"Loaded {len(names)} names from CSV fallback")
        return names
    except Exception as e:
        st.error(f"Failed to load from CSV fallback: {e}")
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
        st.success(f"Fetched {len(names)} names from Koldfront")
        return names
    except Exception as e:
        st.error(f"Failed to fetch from Koldfront: {e}")
        return []


def load_ratings(file_path: str = "ratings.json") -> Optional[Dict[str, float]]:
    """
    Load saved ratings from database.
    Returns ratings dict or empty dict if no ratings exist.
    """
    try:
        database.init_database()
        return database.get_ratings()
    except Exception as e:
        st.warning(f"Could not load ratings from database: {e}")
        return {}


def save_ratings(
    ratings: Dict[str, float], file_path: str = "ratings.json"
) -> bool:
    """
    Save ratings to database.
    Returns True if successful.
    """
    try:
        database.init_database()
        for name, rating in ratings.items():
            database.update_rating(name, rating)
        return True
    except Exception as e:
        st.error(f"Failed to save ratings to database: {e}")
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
        st.info(
            f"Added {new_names_added} new names with initial rating "
            f"{INITIAL_RATING}"
        )

    return ratings


def pull_submodule_updates() -> bool:
    """
    Pull latest updates from the git submodule.
    Returns True if successful.
    """
    try:
        import subprocess  # nosec: B404 - git commands are safe, no user input
        import time

        with st.spinner("Pulling latest name data from git submodule..."):
            result = subprocess.run(  # nosec
                ["git", "-C", "godkendtefornavne", "pull"],
                capture_output=True,
                text=True,
                timeout=30,
            )

        if result.returncode == 0:
            st.success("✅ Submodule updated successfully")
            if result.stdout:
                st.text(f"Output: {result.stdout[:200]}")

            # Show reload message with slight delay
            reload_info = st.info("⏳ Reloading names in 2 seconds...")
            time.sleep(2)
            reload_info.empty()

            return True
        else:
            st.error(f"Failed to pull submodule: {result.stderr}")
            return False
    except Exception as e:
        st.error(f"Error pulling submodule: {e}")
        return False


# -----------------------------------------------------------------------------
# UI Layout
# -----------------------------------------------------------------------------


def setup_session_state(names: List[str]) -> None:
    if "ratings" not in st.session_state:
        st.session_state.ratings = initialize_or_load_ratings(names)

    if "candidate_a" not in st.session_state:
        st.session_state.candidate_a = ""

    if "candidate_b" not in st.session_state:
        st.session_state.candidate_b = ""

    if "names" not in st.session_state:
        st.session_state.names = names


def select_candidates(names: List[str]) -> Tuple[str, str]:
    """
    Simple random selection for candidates.
    In a more complex app, we'd pick names with low comparison counts or
    close ratings (TrueSkill style).
    """
    if len(names) < 2:
        return "", ""

    return tuple(np.random.choice(names, size=2, replace=False))


def render_tournament(names: List[str]) -> None:
    st.header("Elo Rating Tournament")
    st.write(f"Comparing {len(names)} names")
    st.caption(
        "💡 **Keyboard shortcuts**: Left arrow (←) for left name, "
        "Right arrow (→) for right name, Up arrow (↑) for draw"
    )

    # Pick candidates if not set or just reset
    if not st.session_state.candidate_a or not st.session_state.candidate_b:
        c_a, c_b = select_candidates(names)
        st.session_state.candidate_a = c_a
        st.session_state.candidate_b = c_b

    # Inject CSS for metric styling and equal column heights
    st.markdown(
        """
    <style>
    /* Style for st.metric display */
    div[data-testid="stMetricValue"] p {
        font-size: 48px !important;
        font-weight: bold !important;
        text-align: center !important;
        margin-bottom: 5px !important;
    }
    div[data-testid="stMetricLabel"] p {
        font-size: 24px !important;
        color: #666 !important;
        text-align: center !important;
        margin-top: 0 !important;
    }
    div[data-testid="stMetric"] {
        text-align: center !important;
    }
    div[data-testid="stMetricDelta"] svg {
        width: 20px !important;
        height: 20px !important;
    }

    /* Ensure columns have equal height */
    div[data-testid="column"] {
        display: flex !important;
        flex-direction: column !important;
        min-height: 300px !important;
    }
    div[data-testid="column"] > div[data-testid="stVerticalBlock"] {
        flex-grow: 1 !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    _, col_left, _, col_right, _ = st.columns([0.8, 1, 0.4, 1, 0.8])

    # Get both ratings before displaying
    rating_a = st.session_state.ratings.get(
        st.session_state.candidate_a, INITIAL_RATING
    )
    rating_b = st.session_state.ratings.get(
        st.session_state.candidate_b, INITIAL_RATING
    )

    # Calculate rating differences for delta display
    delta_a = rating_a - rating_b  # Positive if A is higher rated
    delta_b = rating_b - rating_a  # Positive if B is higher rated

    with col_left:
        display_name_with_rating(
            st.session_state.candidate_a, rating_a, delta=delta_a
        )

        # Centered button container
        button_container = st.container()
        with button_container:
            st.markdown(
                "<div style='text-align: center'>", unsafe_allow_html=True
            )
            button_clicked = st.button(
                f"👈 Prefer {st.session_state.candidate_a}",
                key="vote_a",
                width="stretch",
                type="primary",
                shortcut="Left",
                help=(
                    f"Select {st.session_state.candidate_a} as preferred "
                    "(Left arrow key)"
                ),
            )
            st.markdown("</div>", unsafe_allow_html=True)

        if button_clicked:
            update_elo_and_save(
                st.session_state.ratings,
                st.session_state.candidate_a,
                st.session_state.candidate_b,
            )
            st.session_state.candidate_a, st.session_state.candidate_b = (
                select_candidates(names)
            )
            st.rerun()

    with col_right:
        display_name_with_rating(
            st.session_state.candidate_b, rating_b, delta=delta_b
        )

        # Centered button container (same as left side)
        button_container = st.container()
        with button_container:
            st.markdown(
                "<div style='text-align: center'>", unsafe_allow_html=True
            )
            button_clicked = st.button(
                f"Prefer {st.session_state.candidate_b} 👉",
                key="vote_b",
                width="stretch",
                type="primary",
                shortcut="Right",
                help=(
                    f"Select {st.session_state.candidate_b} as preferred "
                    "(Right arrow key)"
                ),
            )
            st.markdown("</div>", unsafe_allow_html=True)

        if button_clicked:
            update_elo_and_save(
                st.session_state.ratings,
                st.session_state.candidate_b,
                st.session_state.candidate_a,
            )
            st.session_state.candidate_a, st.session_state.candidate_b = (
                select_candidates(names)
            )
            st.rerun()

    # Draw button centered below both names
    _, col_mid, _ = st.columns([1, 0.4, 1])
    with col_mid:
        draw_clicked = st.button(
            "🤝 Draw / Equal Preference",
            key="vote_draw",
            width="stretch",
            type="secondary",
            shortcut="Up",
            help="Mark both names as equally preferred (Up arrow key)",
        )

    if draw_clicked:
        st.toast("you chose a draw!", duration="long")
        update_elo_draw_and_save(
            st.session_state.ratings,
            st.session_state.candidate_a,
            st.session_state.candidate_b,
        )
        st.session_state.candidate_a, st.session_state.candidate_b = (
            select_candidates(names)
        )
        st.rerun()

    st.divider()
    st.subheader("Current Top 10")
    sorted_ratings = sorted(
        st.session_state.ratings.items(), key=lambda x: x[1], reverse=True
    )
    df = pd.DataFrame(sorted_ratings[:10], columns=["Name", "Rating"])
    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        column_config={
            "ratings": st.column_config.NumberColumn(
                "Rating",
                help="Higher is better",
                format="%d",
                pinned=True,
                width="small",
            )
        },
    )


def render_similarity(names: List[str]) -> None:
    st.header("Similarity Search")

    search_type: Literal["String", "Vector"] = st.radio(
        "Search Method", ["String (Levenshtein)", "Vector (LLM Embedding)"]
    )

    query = st.text_input("Reference Name", value="Alma")

    if st.button("Find Similar") and query:
        if search_type == "String (Levenshtein)":
            results = get_string_similarity_scores(query, names, limit=10)
            st.dataframe(
                pd.DataFrame(results, columns=["Name", "Similarity Score"]),
                width="stretch",
                hide_index=True,
            )
        else:
            with st.spinner("Loading Embedding Model (first run is slow)..."):
                model = load_embedding_model()

            with st.spinner("Computing Similarities..."):
                results = get_vector_similarity_scores(
                    model, query, names, limit=10
                )

            st.dataframe(
                pd.DataFrame(results, columns=["Name", "Cosine Similarity"]),
                width="stretch",
                hide_index=True,
            )


def main() -> None:
    st.set_page_config(page_title="Name Ranker", layout="wide")
    st.title("Name Preference Ranker")

    # Data Loading - Only from submodule
    with st.sidebar:
        # Auto-load from submodule on first run
        if "all_names_data" not in st.session_state:
            with st.spinner("Loading names from submodule..."):
                # Load gender-categorized data
                gender_data = load_names_by_gender()
                if gender_data and "All" in gender_data:
                    # Store the full dataset
                    st.session_state.all_names_data = gender_data
                    st.session_state.all_names = gender_data["All"]

                    # Initialize with all names for ratings
                    setup_session_state(gender_data["All"])
                    st.success(f"Loaded {len(gender_data['All'])} total names")

                    # Show breakdown
                    for gender in ["Male", "Female", "Unisex"]:
                        if gender in gender_data:
                            st.info(
                                f"  • {gender}: "
                                f"{len(gender_data[gender])} names"
                            )

                    # Auto-save ratings after loading new dataset
                    if save_ratings(st.session_state.ratings):
                        st.info("Ratings saved automatically")
                else:
                    st.error("Failed to load names from submodule")
                    return

        # Submodule management
        st.subheader("Submodule Management")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Reload Names"):
                with st.spinner("Reloading..."):
                    # Reload gender-categorized data
                    gender_data = load_names_by_gender()
                    if gender_data and "All" in gender_data:
                        # Update the full dataset
                        st.session_state.all_names_data = gender_data
                        st.session_state.all_names = gender_data["All"]

                        # Reinitialize ratings with all names
                        setup_session_state(gender_data["All"])
                        st.success(
                            f"Reloaded {len(gender_data['All'])} total names"
                        )

                        # Show breakdown
                        for gender in ["Male", "Female", "Unisex"]:
                            if gender in gender_data:
                                st.info(
                                    f"  • {gender}: "
                                    f"{len(gender_data[gender])} names"
                                )
                        st.rerun()

        with col2:
            if st.button("Check for Updates"):
                with st.spinner("Checking for updates..."):
                    if pull_submodule_updates():
                        st.rerun()

        # Show submodule status
        if os.path.exists("godkendtefornavne"):
            try:
                import subprocess  # nosec: B404 - git commands are safe, no user input

                result = subprocess.run(  # nosec
                    [
                        "git",
                        "-C",
                        "godkendtefornavne",
                        "log",
                        "-1",
                        "--format=%cd",
                        "--date=short",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    last_update = result.stdout.strip()
                    st.caption(f"Last update: {last_update}")
            except Exception:  # nosec: B110 - non-critical, can safely ignore git log failure
                pass

        st.divider()

        # Gender Filtering
        st.subheader("Gender Filter")
        if "gender_filter" not in st.session_state:
            st.session_state.gender_filter = "All"

        # Use pills for gender selection - modern, tab-like appearance
        gender_option = st.pills(
            "Filter names by gender:",
            ["All", "Male", "Female", "Unisex"],
            default=st.session_state.gender_filter,
            help=(
                "Select which gender of names to compare. "
                "Click or use left/right arrow keys to navigate."
            ),
        )

        if gender_option != st.session_state.gender_filter:
            st.session_state.gender_filter = gender_option
            st.info(f"Filter set to: {gender_option}")
            # When changing filter, we need to update the displayed names
            # but keep all ratings
            st.rerun()

        # Origin Filtering
        st.subheader("Origin Filter")
        
        # Get available origin regions from database
        available_regions = database.get_all_origin_regions()
        
        # Load saved origin filter from database
        if "origin_filter" not in st.session_state:
            saved_origins_json = database.load_user_setting("selected_origins", "[]")
            try:
                saved_origins = json.loads(saved_origins_json)
                # Validate that saved origins are still available
                saved_origins = [o for o in saved_origins if o in available_regions]
                st.session_state.origin_filter = saved_origins
            except:
                # Default: all regions selected
                st.session_state.origin_filter = available_regions.copy()
        
        # Multiselect for origin filter
        selected_origins = st.multiselect(
            "Filter names by origin region:",
            options=available_regions,
            default=st.session_state.origin_filter,
            help="Select one or more origin regions. Leave empty to show all names."
        )
        
        # Save to session state and persist to database if changed
        if selected_origins != st.session_state.origin_filter:
            st.session_state.origin_filter = selected_origins
            # Save to database
            database.save_user_setting("selected_origins", json.dumps(selected_origins))
            st.info(f"Origin filter updated: {selected_origins if selected_origins else 'All regions'}")
            st.rerun()
        
        st.divider()

        # Ratings management
        st.subheader("Ratings Management")

        if "names" in st.session_state and st.session_state.names:
            st.caption(f"Active Dataset: {len(st.session_state.names)} names")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Save Ratings"):
                    if save_ratings(st.session_state.ratings):
                        st.success("Ratings saved!")
            with col2:
                if st.button(
                    "Reset Ratings",
                    help=(
                        "Reset all ratings to initial values. "
                        "This action cannot be undone."
                    ),
                    type="secondary",
                ):
                    # Open confirmation dialog
                    with st.dialog("Confirm Reset Ratings"):
                        st.markdown("### ⚠️ Reset All Ratings?")
                        st.write(
                            "This will reset **all** ratings to their "
                            "initial values."
                        )
                        st.write("**This action cannot be undone.**")

                        col_confirm, col_cancel = st.columns(2)
                        with col_confirm:
                            if st.button("Yes, Reset Ratings", type="primary"):
                                st.session_state.ratings = initialize_ratings(
                                    st.session_state.names
                                )
                                st.success("✅ Ratings reset to initial values")
                                st.rerun()
                        with col_cancel:
                            if st.button("Cancel", type="secondary"):
                                st.rerun()

            # Export ratings
            st.subheader("Export")
            if st.button("Export Ratings as JSON"):
                import json as json_module

                ratings_json = json_module.dumps(
                    {
                        "ratings": st.session_state.ratings,
                        "export_date": datetime.now().isoformat(),
                        "total_names": len(st.session_state.ratings),
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                st.download_button(
                    label="Download Ratings JSON",
                    data=ratings_json,
                    file_name=f"name_ratings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                )

    # Main Content
    if "all_names_data" not in st.session_state:
        st.warning("Please load names from the sidebar.")
        return

    # Get current gender filter
    current_gender = st.session_state.get("gender_filter", "All")

    # Get current origin filter
    current_origins = st.session_state.get("origin_filter", [])
    # Empty list means no origin filtering (show all regions)
    origins_to_filter = current_origins if current_origins else None
    
    # Get filtered names using database
    filtered_names = get_filtered_names(
        gender=current_gender if current_gender != "All" else None,
        origins=origins_to_filter
    )
    
    if not filtered_names:
        if current_origins:
            st.warning(
                f"No names found for gender: {current_gender}, "
                f"origins: {current_origins}"
            )
        else:
            st.warning(f"No names found for gender filter: {current_gender}")
        return
    
    # Get total names count for reference (all names in database)
    total_names_count = len(st.session_state.all_names)
    
    # Show filter info
    if current_origins:
        st.info(
            f"Showing {len(filtered_names)} names "
            f"(gender: {current_gender.lower()}, "
            f"origins: {', '.join(current_origins)}) "
            f"out of {total_names_count} total"
        )
    else:
        st.info(
            f"Showing {len(filtered_names)} {current_gender.lower()} names "
            f"out of {total_names_count} total"
        )

    tab1, tab2 = st.tabs(["Tournament", "Similarity Search"])

    with tab1:
        render_tournament(filtered_names)

    with tab2:
        render_similarity(filtered_names)


if __name__ == "__main__":
    main()
