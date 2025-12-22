import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Literal, Optional
import httpx
import json
import os
from datetime import datetime
from rapidfuzz import fuzz, process

from sentence_transformers import SentenceTransformer

K_FACTOR: float = 32.0
INITIAL_RATING: float = 1500.0

# -----------------------------------------------------------------------------
# Display Helpers
# -----------------------------------------------------------------------------

def display_name_with_rating(name: str, rating: float) -> None:
    """
    Display name much larger than rating using st.metric with custom styling.
    Uses CSS injection to make the value (name) larger and label (rating) smaller.
    """
    # Inject CSS to override st.metric styles
    st.markdown("""
    <style>
    /* Make metric value (name) much larger */
    div[data-testid="stMetricValue"] p {
        font-size: 48px !important;
        font-weight: bold !important;
        text-align: center !important;
        margin-bottom: 5px !important;
    }
    /* Make metric label (rating) smaller */
    div[data-testid="stMetricLabel"] p {
        font-size: 24px !important;
        color: #666 !important;
        text-align: center !important;
        margin-top: 0 !important;
    }
    /* Center the metric container */
    div[data-testid="stMetric"] {
        text-align: center !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Use st.metric with swapped label/value to get desired visual hierarchy
    # Value will be large (name), label will be smaller (rating)
    st.metric(value=name, label=f"{rating:.0f}")

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
    ratings: Dict[str, float],
    winner: str,
    loser: str,
    k: float = K_FACTOR
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

def update_elo(
    ratings: Dict[str, float],
    winner: str,
    loser: str,
    k: float = K_FACTOR
) -> Dict[str, float]:
    """
    Update Elo ratings based on a binary outcome (1 for winner, 0 for loser).
    """
    if winner not in ratings or loser not in ratings:
        return ratings

    r_winner = ratings[winner]
    r_loser = ratings[loser]

    e_winner = expected_score(r_winner, r_loser)
    e_loser = 1.0 - e_winner

    ratings[winner] = r_winner + k * (1.0 - e_winner)
    ratings[loser] = r_loser + k * (0.0 - e_loser)

    return ratings

def initialize_ratings(names: List[str]) -> Dict[str, float]:
    return {name.strip(): INITIAL_RATING for name in names if name.strip()}

# -----------------------------------------------------------------------------
# Similarity Logic: String & Vector
# -----------------------------------------------------------------------------

def get_string_similarity_scores(
    target: str,
    candidates: List[str],
    limit: int = 10
) -> List[Tuple[str, float]]:
    """
    Uses RapidFuzz (Levenshtein) to find similar names.
    Returns list of (name, score).
    """
    if not candidates:
        return []

    # process.extract returns (match, score, index)
    results = process.extract(target, candidates, scorer=fuzz.ratio, limit=limit)
    return [(item[0], item[1]) for item in results]

@st.cache_resource
def load_embedding_model() -> SentenceTransformer:
    # 'paraphrase-multilingual-MiniLM-L12-v2' is good for Danish/English
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def get_vector_similarity_scores(
    model: SentenceTransformer,
    target: str,
    candidates: List[str],
    limit: int = 10
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
                names = [line.strip() for line in response.text.splitlines() if line.strip()]
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
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        if "Navn" not in df.columns:
            st.error(f"CSV file missing 'Navn' column. Columns found: {df.columns.tolist()}")
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
    Load names from local git submodule JSON file, categorized by gender.
    Returns dict mapping gender to list of names.
    Unisex names are included in both 'Male' and 'Female' categories.
    """
    try:
        data = load_submodule_json()
        if not data:
            return {}

        # Initialize gender categories
        gender_lists = {
            "Female": set(),
            "Male": set(),
            "Unisex": set(),
            "All": set()
        }

        # Categorize names
        for item in data:
            name = item["name"]
            gender = item["gender"]

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
        st.error(f"Failed to load names by gender: {e}")
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
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            st.error(f"Expected JSON array, got {type(data)}")
            return []
        # Validate structure
        valid_items = []
        for item in data:
            if isinstance(item, dict) and "name" in item and "gender" in item:
                valid_items.append({
                    "name": str(item["name"]).strip(),
                    "gender": str(item["gender"]).strip()
                })
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
    csv_files = [
        "drengenavne.csv",
        "pigenavne.csv",
        "unisexnavne.csv"
    ]

    all_names = []
    try:
        for csv_file in csv_files:
            file_path = os.path.join(submodule_path, csv_file)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    names = [line.strip() for line in f.readlines() if line.strip()]
                    all_names.extend(names)
            else:
                st.warning(f"Submodule CSV file not found: {file_path}")

        if not all_names:
            st.error("No names found in submodule files")
            return []

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
                names = [line.strip() for line in response.text.splitlines() if line.strip()]
                all_names.extend(names)
        names = sorted(list(set(all_names)))
        st.success(f"Fetched {len(names)} names from Koldfront")
        return names
    except Exception as e:
        st.error(f"Failed to fetch from Koldfront: {e}")
        return []


def load_ratings(file_path: str = "ratings.json") -> Optional[Dict[str, float]]:
    """
    Load saved ratings from JSON file.
    Returns ratings dict or None if file doesn't exist.
    """
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Check if it's the new format with metadata
                if isinstance(data, dict) and "ratings" in data:
                    return data["ratings"]
                # Old format: just ratings dict
                elif isinstance(data, dict):
                    return data
        return None
    except Exception as e:
        st.warning(f"Could not load ratings: {e}")
        return None


def save_ratings(ratings: Dict[str, float], file_path: str = "ratings.json") -> bool:
    """
    Save ratings to JSON file with metadata.
    Returns True if successful.
    """
    try:
        data = {
            "ratings": ratings,
            "last_saved": datetime.now().isoformat(),
            "total_names": len(ratings)
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"Failed to save ratings: {e}")
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
        st.info(f"Added {new_names_added} new names with initial rating {INITIAL_RATING}")

    return ratings


def pull_submodule_updates() -> bool:
    """
    Pull latest updates from the git submodule.
    Returns True if successful.
    """
    try:
        import subprocess
        result = subprocess.run(
            ["git", "-C", "godkendtefornavne", "pull"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            st.success("Submodule updated successfully")
            if result.stdout:
                st.text(f"Output: {result.stdout[:200]}")
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

    # Pick candidates if not set or just reset
    if not st.session_state.candidate_a or not st.session_state.candidate_b:
        c_a, c_b = select_candidates(names)
        st.session_state.candidate_a = c_a
        st.session_state.candidate_b = c_b

    col_a, col_b = st.columns(2)

    with col_a:
        rating_a = st.session_state.ratings.get(st.session_state.candidate_a, INITIAL_RATING)
        display_name_with_rating(st.session_state.candidate_a, rating_a)
        if st.button(
            f"👈 Prefer {st.session_state.candidate_a}",
            key="vote_a",
            use_container_width=True,
            type="primary"
        ):
            update_elo_and_save(
                st.session_state.ratings,
                st.session_state.candidate_a,
                st.session_state.candidate_b
            )
            st.session_state.candidate_a, st.session_state.candidate_b = select_candidates(names)
            st.rerun()

    with col_b:
        rating_b = st.session_state.ratings.get(st.session_state.candidate_b, INITIAL_RATING)
        display_name_with_rating(st.session_state.candidate_b, rating_b)
        if st.button(
            f"Prefer {st.session_state.candidate_b} 👉",
            key="vote_b",
            use_container_width=True,
            type="primary"
        ):
            update_elo_and_save(
                st.session_state.ratings,
                st.session_state.candidate_b,
                st.session_state.candidate_a
            )
            st.session_state.candidate_a, st.session_state.candidate_b = select_candidates(names)
            st.rerun()

    st.divider()
    st.subheader("Current Top 10")
    sorted_ratings = sorted(st.session_state.ratings.items(), key=lambda x: x[1], reverse=True)
    df = pd.DataFrame(sorted_ratings[:10], columns=["Name", "Rating"])
    st.dataframe(df, use_container_width=True, hide_index=True)

def render_similarity(names: List[str]) -> None:
    st.header("Similarity Search")

    search_type: Literal["String", "Vector"] = st.radio(
        "Search Method",
        ["String (Levenshtein)", "Vector (LLM Embedding)"]
    )

    query = st.text_input("Reference Name", value="Alma")

    if st.button("Find Similar") and query:
        if search_type == "String (Levenshtein)":
            results = get_string_similarity_scores(query, names, limit=10)
            st.dataframe(
                pd.DataFrame(results, columns=["Name", "Similarity Score"]),
                use_container_width=True,
                hide_index=True
            )
        else:
            with st.spinner("Loading Embedding Model (first run is slow)..."):
                model = load_embedding_model()

            with st.spinner("Computing Similarities..."):
                results = get_vector_similarity_scores(model, query, names, limit=10)

            st.dataframe(
                pd.DataFrame(results, columns=["Name", "Cosine Similarity"]),
                use_container_width=True,
                hide_index=True
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
                            st.info(f"  • {gender}: {len(gender_data[gender])} names")

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
                        st.success(f"Reloaded {len(gender_data['All'])} total names")

                        # Show breakdown
                        for gender in ["Male", "Female", "Unisex"]:
                            if gender in gender_data:
                                st.info(f"  • {gender}: {len(gender_data[gender])} names")
                        st.rerun()

        with col2:
            if st.button("Check for Updates"):
                with st.spinner("Checking for updates..."):
                    if pull_submodule_updates():
                        st.rerun()

        # Show submodule status
        if os.path.exists("godkendtefornavne"):
            try:
                import subprocess
                result = subprocess.run(
                    ["git", "-C", "godkendtefornavne", "log", "-1", "--format=%cd", "--date=short"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    last_update = result.stdout.strip()
                    st.caption(f"Last update: {last_update}")
            except:
                pass

        st.divider()

        # Gender Filtering
        st.subheader("Gender Filter")
        if "gender_filter" not in st.session_state:
            st.session_state.gender_filter = "All"

        gender_option = st.selectbox(
            "Filter names by gender:",
            ["All", "Male", "Female", "Unisex"],
            index=["All", "Male", "Female", "Unisex"].index(st.session_state.gender_filter)
        )

        if gender_option != st.session_state.gender_filter:
            st.session_state.gender_filter = gender_option
            st.info(f"Filter set to: {gender_option}")
            # When changing filter, we need to update the displayed names
            # but keep all ratings
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
                if st.button("Reset Ratings"):
                    st.session_state.ratings = initialize_ratings(st.session_state.names)
                    st.success("Ratings reset to initial values")
                    st.rerun()

            # Export ratings
            st.subheader("Export")
            if st.button("Export Ratings as JSON"):
                import io
                import json as json_module
                ratings_json = json_module.dumps(
                    {
                        "ratings": st.session_state.ratings,
                        "export_date": datetime.now().isoformat(),
                        "total_names": len(st.session_state.ratings)
                    },
                    indent=2,
                    ensure_ascii=False
                )
                st.download_button(
                    label="Download Ratings JSON",
                    data=ratings_json,
                    file_name=f"name_ratings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

    # Main Content
    if "all_names_data" not in st.session_state:
        st.warning("Please load names from the sidebar.")
        return

    # Get current gender filter
    current_gender = st.session_state.get("gender_filter", "All")

    # Get filtered names for the current gender
    if current_gender in st.session_state.all_names_data:
        filtered_names = st.session_state.all_names_data[current_gender]
    else:
        filtered_names = st.session_state.all_names_data.get("All", [])

    if not filtered_names:
        st.warning(f"No names found for gender filter: {current_gender}")
        return

    # Show filter info
    st.info(f"Showing {len(filtered_names)} {current_gender.lower()} names (out of {len(st.session_state.all_names)} total)")

    tab1, tab2 = st.tabs(["Tournament", "Similarity Search"])

    with tab1:
        render_tournament(filtered_names)

    with tab2:
        render_similarity(filtered_names)

if __name__ == "__main__":
    main()
