"""Utility functions for the name ranking application."""

import functools
import logging
import sqlite3
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import streamlit as st

from st_name_ranking import database
from st_name_ranking.data_loader import initialize_or_load_ratings

# K_FACTOR no longer used - model uses Bayesian updates
from st_name_ranking.features import FeatureExtractor
from st_name_ranking.model import BradleyTerryModel, initialize_model_if_needed
from st_name_ranking.phonetic_similarity import phonetic_similarity

logger = logging.getLogger(__name__)

# Minimum names required for pair selection
MIN_NAMES_FOR_PAIR_SELECTION = 2

# Thread pool for background model updates
_model_update_lock = threading.Lock()


@functools.lru_cache
def get_thread_executor() -> ThreadPoolExecutor:
    """Shared thread pool for background tasks."""
    return ThreadPoolExecutor(max_workers=2)


def update_model_async(name_a: str, name_b: str, preference: int) -> None:
    """Queue model update and run in background thread.

    Thread-safe: model updates are queued and executed sequentially.
    """

    def _update():
        # Acquire lock to ensure sequential model updates
        with _model_update_lock:
            model = get_active_learning_model()
            features_a = get_name_features(name_a)
            features_b = get_name_features(name_b)

            # IRLS update (slow, but in background)
            model.update(features_a, features_b, preference)
            model.save_to_db()

    # Submit to thread pool
    executor = get_thread_executor()
    executor.submit(_update)


def update_ratings_async() -> None:
    """Queue ratings update to run in background thread."""

    def _update():
        _update_ratings_from_model()

    executor = get_thread_executor()
    executor.submit(_update)


def record_comparison_instant(
    name_a: str,
    name_b: str,
    preference: int,
    blocking: bool = False,
) -> None:
    """Instant comparison recording - just records, no waiting.

    Args:
        name_a: First name in comparison
        name_b: Second name in comparison
        preference: -1 (name_a preferred), 0 (draw), 1 (name_b preferred), 2 (both disliked)
        blocking: If True, wait for model update to complete (legacy behavior)
    """
    from concurrent.futures import wait

    # 1. Fast DB insert (always synchronous)
    database.record_comparison(name_a, name_b, preference)

    # 2. Queue async model update (non-blocking)
    future = get_thread_executor().submit(_update_model_sync, name_a, name_b, preference)

    # 3. Queue async ratings update (non-blocking)
    ratings_future = get_thread_executor().submit(_update_ratings_from_model)

    if blocking:
        # Wait for background tasks to complete
        wait([future, ratings_future])


def _update_model_sync(name_a: str, name_b: str, preference: int) -> None:
    """Synchronous model update - called from thread pool.

    Thread-safe: acquires lock to ensure sequential updates.
    """
    with _model_update_lock:
        try:
            model = get_active_learning_model()
            features_a = get_name_features(name_a)
            features_b = get_name_features(name_b)

            if preference == 2:
                # Both disliked
                model.update_both_disliked(features_a, features_b)
            else:
                # Standard preference (-1, 0, 1)
                model.update(features_a, features_b, preference)

            model.save_to_db()
        except (RuntimeError, ValueError, AttributeError):
            logger.exception("Failed to update model in background")


def get_active_learning_model() -> BradleyTerryModel:
    """Get or initialize the active learning model."""
    if get_active_learning_model._cache is None:
        # Initialize feature extractor first to get feature names
        extractor = get_feature_extractor()
        feature_names = extractor.get_feature_names()
        get_active_learning_model._cache = initialize_model_if_needed(feature_names)
    return get_active_learning_model._cache


get_active_learning_model._cache = None


def get_feature_extractor() -> FeatureExtractor:
    """Get or initialize the feature extractor."""
    if get_feature_extractor._cache is None:
        get_feature_extractor._cache = FeatureExtractor()
    return get_feature_extractor._cache


get_feature_extractor._cache = None


def get_name_features(name: str) -> np.ndarray:
    """Extract features for a name by querying gender and origin from database."""
    extractor = get_feature_extractor()

    # Get name details from database
    with database.get_connection() as conn:
        cursor = conn.execute(
            "SELECT gender, origin_region FROM names WHERE name = ?",
            (name,),
        )
        row = cursor.fetchone()

        if row:
            gender, origin_region = row
        else:
            # Name not in database (shouldn't happen)
            gender, origin_region = None, None

    return extractor.extract(name, gender, origin_region)


def get_names_features(names: list[str]) -> np.ndarray:
    """Extract features for multiple names in batch.
    Returns feature matrix of shape (n_names, n_features).
    """
    extractor = get_feature_extractor()

    # Get name details from database in batch
    details = database.get_name_details_batch(names)
    genders = [d.gender for d in details]
    origins = [d.origin_region for d in details]

    return extractor.batch_extract(names, genders, origins)


def _compute_rating_for_name(name: str) -> float:
    """Compute rating for a single name using current model weights.
    Returns rating = 1500 + utility * 500.
    """
    model = get_active_learning_model()
    features = get_name_features(name)
    utility = model.get_utility(features.reshape(1, -1))[0]  # get_utility expects 2D
    rating = 1500 + utility * 500
    return rating


def pull_submodule_updates(*, classify_origins: bool = False) -> bool:
    """Pull latest updates from the git submodule and sync with database.
    If classify_origins is True and ethnidata is available, classify origins.
    Returns True if successful.
    """
    logger.debug(
        "Pulling submodule updates, classify_origins=%s",
        classify_origins,
    )
    try:
        with st.spinner("Pulling latest name data from git submodule..."):
            result = subprocess.run(  # nosec
                ["git", "-C", "godkendtefornavne", "pull"],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
    except subprocess.SubprocessError as e:
        st.toast(
            f"Error pulling submodule: {e}",
            icon="❌",
            duration="long",
        )
        return False
    else:
        if result.returncode == 0:
            st.toast(
                "✅ Submodule updated successfully",
                icon="✅",
            )
            if result.stdout:
                st.text(f"Output: {result.stdout[:200]}")

            # Sync new names with database
            with st.spinner("Syncing new names with database..."):
                try:
                    # Ensure database is initialized
                    database.init_database()
                    inserted = database.sync_names_with_submodule()
                    if inserted > 0:
                        st.toast(
                            f"✅ Added {inserted} new names to database",
                            icon="✅",
                        )
                    else:
                        st.toast(
                            "No new names to add",
                            icon="ℹ️",
                        )
                except (RuntimeError, ValueError, subprocess.SubprocessError) as sync_error:
                    st.toast(
                        f"Failed to sync names: {sync_error}",
                        icon="❌",
                        duration="long",
                    )
                    # Continue anyway - names will be synced on next load

            # Classify origins if requested and ethnidata is available
            if classify_origins:
                try:
                    # Ensure database is initialized before classification
                    database.init_database()
                    from st_name_ranking.classify_origins import classify_all_names

                    with st.spinner("Classifying name origins..."):
                        # Classify only unclassified names
                        classified = classify_all_names(limit=None)
                        if classified > 0:
                            st.toast(
                                f"✅ Classified {classified} name origins",
                                icon="✅",
                            )
                        else:
                            st.toast(
                                "No names needed classification",
                                icon="ℹ️",
                            )
                except ImportError:
                    st.toast(
                        "ethnidata not installed. Run: pip install ethnidata",
                        icon="⚠️",
                    )
                except (RuntimeError, ValueError) as classify_error:
                    st.toast(
                        f"Failed to classify origins: {classify_error}",
                        icon="❌",
                        duration="long",
                    )

            # Show reload message with slight delay
            st.toast("⏳ Reloading names in 2 seconds...", icon="⏳")
            time.sleep(2)

            return True
        st.toast(
            f"Failed to pull submodule: {result.stderr}",
            icon="❌",
            duration="long",
        )
        return False


def setup_session_state(names: list[str]) -> None:
    if "ratings" not in st.session_state:
        st.session_state["ratings"] = initialize_or_load_ratings(names)

    if "candidate_a" not in st.session_state:
        st.session_state["candidate_a"] = ""

    if "candidate_b" not in st.session_state:
        st.session_state["candidate_b"] = ""

    if "names" not in st.session_state:
        st.session_state["names"] = names


def select_candidates(
    names: list[str],
    features: np.ndarray | None = None,
    sample_size: int | None = None,
) -> tuple[str, str]:
    """Active learning candidate selection using Bradley-Terry model.
    Uses Thompson sampling to select maximally informative pairs.

    Args:
        names: List of candidate names
        features: Optional precomputed feature matrix of shape (len(names), n_features).
                 If None, features will be computed on the fly.

    """
    if len(names) < MIN_NAMES_FOR_PAIR_SELECTION:
        return "", ""

    rng = np.random.default_rng()
    # Sample a subset of names for efficiency
    effective_sample_size = min(max(sample_size if sample_size is not None else 50, 2), len(names))
    if len(names) == effective_sample_size:
        sampled = names
        sampled_indices = list(range(len(names)))
    else:
        sampled_indices = list(
            rng.choice(len(names), size=effective_sample_size, replace=False),
        )
        sampled = [names[i] for i in sampled_indices]

    try:
        # Get model
        model = get_active_learning_model()

        # Get features for sampled names
        if features is not None:
            # Use precomputed features, subset by sampled indices
            sampled_features = features[sampled_indices]
        else:
            # Compute features on the fly
            sampled_features = get_names_features(sampled)

        # Use model's Thompson sampling for pair selection
        pair = model.select_pair(
            sampled_features,
            sampled,
        )
    except (RuntimeError, ValueError, AttributeError) as e:
        logger.warning(
            "Active learning pair selection failed: %s. Falling back to basic selection.",
            e,
        )
        # Fallback to basic selection (doesn't use features)
        return _select_candidates_fallback(sampled)
    else:
        return pair.name_a, pair.name_b


def select_candidate_batch(
    names: list[str],
    features: np.ndarray | None = None,
    batch_size: int = 3,
    sample_size: int | None = None,
) -> list[tuple[str, str]]:
    """Select a batch of candidate pairs for active learning.
    Returns list of (name_a, name_b) pairs.
    """
    if len(names) < MIN_NAMES_FOR_PAIR_SELECTION:
        return []
    batch_size = max(batch_size, 1)
    rng = np.random.default_rng()

    # Sample a subset of names for efficiency
    effective_sample_size = min(max(sample_size if sample_size is not None else 50, 2), len(names))
    if len(names) == effective_sample_size:
        sampled = names
        sampled_indices = list(range(len(names)))
    else:
        sampled_indices = list(
            rng.choice(len(names), size=effective_sample_size, replace=False),
        )
        sampled = [names[i] for i in sampled_indices]

    try:
        model = get_active_learning_model()
        if features is not None:
            sampled_features = features[sampled_indices]
        else:
            sampled_features = get_names_features(sampled)

        pairs = model.select_top_k_pairs(
            sampled_features,
            sampled,
            k=batch_size,
        )
        # Convert to list of (name_a, name_b)
        return [(pair.name_a, pair.name_b) for pair in pairs]
    except (RuntimeError, ValueError, AttributeError) as e:
        logger.warning(
            "Active learning batch selection failed: %s. Falling back to single pair selection.",
            e,
        )
        # Fallback to single pair selection
        if sample_size is None:
            pair = select_candidates(names, features)
        else:
            pair = select_candidates(names, features, sample_size=sample_size)
        return [pair] if pair[0] else []


def _select_candidates_fallback(names: list[str]) -> tuple[str, str]:
    """Fallback candidate selection using comparison counts and phonetic similarity.
    Used when active learning model fails.
    """
    if len(names) < MIN_NAMES_FOR_PAIR_SELECTION:
        return "", ""
    rng = np.random.default_rng()

    # Get comparison counts for names
    counts = {}
    for name in names:
        counts[name] = database.get_comparison_count(name)

    # Compute utility based on inverse comparison frequency
    utilities = {}
    for name, count in counts.items():
        utilities[name] = 1.0 / (count + 1)

    # Evaluate pairs among names (limit to 100 random pairs for efficiency)
    n_pairs = min(100, len(names) * (len(names) - 1) // 2)
    best_pair = ("", "")
    best_score = -1.0

    for _ in range(n_pairs):
        i, j = rng.choice(len(names), size=2, replace=False)
        a = names[i]
        b = names[j]
        # Pair score = sum of utilities + phonetic similarity (0-1 scale)
        phonetic_score = phonetic_similarity(a, b)
        pair_score = utilities[a] + utilities[b] + phonetic_score

        if pair_score > best_score:
            best_score = pair_score
            best_pair = (a, b)

    # Fallback to random if something went wrong
    if best_pair == ("", ""):
        return tuple(rng.choice(names, size=2, replace=False))

    return best_pair


def update_model_and_save(winner: str, loser: str) -> None:
    """Update Bradley-Terry model with comparison result and save."""
    try:
        model = get_active_learning_model()

        # Extract features for both names
        features_a = get_name_features(winner)
        features_b = get_name_features(loser)

        # Update model (preference: -1 means winner preferred, which is name_a)
        # Actually, winner is name_a, so preference = -1 (a preferred)
        model.update(features_a, features_b, -1)

        # Save model state to database
        model.save_to_db()
        # Ratings are updated separately via update_preference_and_save

    except (RuntimeError, ValueError, AttributeError):
        logger.exception("Failed to update model")
        # Don't crash the UI, but log the error


def update_model_draw_and_save(player_a: str, player_b: str) -> None:
    """Update Bradley-Terry model with draw result and save."""
    try:
        model = get_active_learning_model()

        # Extract features for both names
        features_a = get_name_features(player_a)
        features_b = get_name_features(player_b)

        # Update model with draw (preference: 0)
        model.update(features_a, features_b, 0)

        # Save model state to database
        model.save_to_db()
        # Ratings are updated separately via update_preference_draw_and_save

    except (RuntimeError, ValueError, AttributeError):
        logger.exception("Failed to update model for draw")


def update_model_down_and_save(player_a: str, player_b: str) -> None:
    """Update Bradley-Terry model with both names disliked result and save."""
    try:
        model = get_active_learning_model()

        # Extract features for both names
        features_a = get_name_features(player_a)
        features_b = get_name_features(player_b)

        # Update model with both disliked
        model.update_both_disliked(features_a, features_b)

        # Save model state to database
        model.save_to_db()
        # Ratings are updated separately via update_preference_down_and_save

    except (RuntimeError, ValueError, AttributeError):
        logger.exception("Failed to update model for both disliked")


def _update_ratings_from_model() -> None:
    """Update ratings table with current model utilities for all names.
    This maintains backward compatibility with UI that displays ratings.
    """
    try:
        model = get_active_learning_model()

        # Get all names from database
        with database.get_connection() as conn:
            cursor = conn.execute("SELECT name FROM names")
            names = [row[0] for row in cursor.fetchall()]

        if not names:
            return

        # Extract features for all names (batch)
        features = get_names_features(names)

        # Get model utilities
        utilities = model.get_utility(features)

        # Update ratings table in batch
        ratings_dict = {}
        for name, utility in zip(names, utilities):
            # Convert utility to preference score scale (1500 ± 500)
            # This is just for display purposes
            rating = 1500 + utility * 500
            ratings_dict[name] = rating
        database.update_ratings_batch_values(ratings_dict)

    except (RuntimeError, ValueError) as e:
        logger.warning("Failed to update ratings from model: %s", e)


def sync_names_from_submodule() -> int:
    """Sync names from submodule JSON to database.
    Returns number of new names added.
    """
    try:
        database.init_database()
        with st.spinner("Syncing names from submodule..."):
            inserted = database.sync_names_with_submodule()
            if inserted > 0:
                st.toast(
                    f"✅ Added {inserted} new names to database",
                    icon="✅",
                )
            else:
                st.toast(
                    "Database already up to date with submodule",
                    icon="ℹ️",
                )
            return inserted
    except (RuntimeError, ValueError, subprocess.SubprocessError) as e:
        st.toast(
            f"Failed to sync names: {e}",
            icon="❌",
            duration="long",
        )
        return 0


def record_comparison_fast(name_a: str, name_b: str, preference: int) -> None:
    """Record a comparison in the database without triggering model updates.

    This is the fast path for UI responsiveness - only does a single DB insert
    without any model retraining or rating recomputation.

    Args:
        name_a: First name in comparison
        name_b: Second name in comparison
        preference: -1 (name_a preferred), 0 (draw), 1 (name_b preferred), 2 (both disliked)
    """
    try:
        database.record_comparison(name_a, name_b, preference)
        logger.debug("Fast recorded comparison: %s vs %s (pref=%d)", name_a, name_b, preference)
    except (sqlite3.Error, ValueError) as e:
        logger.warning("Failed to record comparison: %s", e)


def queue_model_update(names: list[str] | None = None) -> None:
    """Queue a model update to be processed in batch.

    Stores pending comparison count in session state. When threshold is reached,
    triggers actual model update.

    Args:
        names: Optional list of names for batch rating update after model training
    """
    import streamlit as st

    # Initialize pending updates counter
    if "pending_model_updates" not in st.session_state:
        st.session_state.pending_model_updates = 0

    st.session_state.pending_model_updates += 1

    # Trigger batch update every 10 comparisons (configurable)
    batch_threshold = st.session_state.get("model_update_batch_threshold", 10)

    if st.session_state.pending_model_updates >= batch_threshold:
        logger.info("Batch model update triggered (%d comparisons)", st.session_state.pending_model_updates)
        try:
            # Import here to avoid circular dependency
            from st_name_ranking.data_loader import initialize_or_load_ratings

            # Re-initialize ratings which loads from database and retrains model
            new_ratings = initialize_or_load_ratings(names or [])
            st.session_state.ratings = new_ratings
            st.session_state.pending_model_updates = 0
            st.session_state.model_last_updated = time.time()
            logger.info("Batch model update completed")
        except Exception:
            logger.exception("Batch model update failed")


def select_candidates_fast(
    names: list[str],
    features: np.ndarray | None = None,
) -> tuple[str, str]:
    """Fast candidate selection using random sampling.

    Falls back to random pair selection without model inference.
    This is O(1) vs O(n) for model-based selection.

    Args:
        names: List of candidate names
        features: Ignored, kept for API compatibility

    Returns:
        Tuple of (name_a, name_b)
    """
    if len(names) < MIN_NAMES_FOR_PAIR_SELECTION:
        return "", ""

    rng = np.random.default_rng()

    # Simple random selection - O(1)
    idx_a, idx_b = rng.choice(len(names), size=2, replace=False)
    return names[idx_a], names[idx_b]


def should_use_model_selection() -> bool:
    """Check if model-based selection should be used.

    Returns False if model is being updated or hasn't been updated recently.
    """
    import streamlit as st

    # If there are pending updates, use fast path
    if st.session_state.get("pending_model_updates", 0) > 0:
        return False

    # If model was never updated or last update > 60 seconds ago, use fast path
    last_update = st.session_state.get("model_last_updated", 0)
    if time.time() - last_update > 60:
        return False

    return True


def update_preference_and_save(
    ratings: dict[str, float],
    winner: str,
    loser: str,
    *,
    blocking: bool = False,
) -> dict[str, float]:
    """Update Bayesian preference model with comparison result.
    Returns updated ratings dict derived from model weights.

    Args:
        ratings: Current ratings dictionary
        winner: Winning name (preferred)
        loser: Losing name
        blocking: If True, wait for model update to complete (legacy behavior)
                  If False, queue async update and return immediately
    """
    # Use instant recording (async by default)
    record_comparison_instant(winner, loser, -1, blocking=blocking)

    # Return updated ratings (may be stale if non-blocking)
    if not blocking:
        # Non-blocking: return current ratings immediately
        return ratings.copy()

    # Blocking: compute fresh ratings
    try:
        winner_rating = _compute_rating_for_name(winner)
        loser_rating = _compute_rating_for_name(loser)
        ratings[winner] = winner_rating
        ratings[loser] = loser_rating
        database.update_rating_value(winner, winner_rating)
        database.update_rating_value(loser, loser_rating)
        return ratings.copy()
    except (RuntimeError, ValueError) as e:
        logger.warning("Failed to compute updated ratings: %s", e)
        return ratings.copy()


def update_preference_draw_and_save(
    ratings: dict[str, float],
    player_a: str,
    player_b: str,
    *,
    blocking: bool = False,
) -> dict[str, float]:
    """Update Bayesian preference model with draw result.
    Returns updated ratings dict derived from model weights.

    Args:
        ratings: Current ratings dictionary
        player_a: First name
        player_b: Second name
        blocking: If True, wait for model update to complete
    """
    record_comparison_instant(player_a, player_b, 0, blocking=blocking)

    if not blocking:
        return ratings.copy()

    try:
        rating_a = _compute_rating_for_name(player_a)
        rating_b = _compute_rating_for_name(player_b)
        ratings[player_a] = rating_a
        ratings[player_b] = rating_b
        database.update_rating_value(player_a, rating_a)
        database.update_rating_value(player_b, rating_b)
        return ratings.copy()
    except (RuntimeError, ValueError) as e:
        logger.warning("Failed to compute updated ratings: %s", e)
        return ratings.copy()


def update_preference_down_and_save(
    ratings: dict[str, float],
    player_a: str,
    player_b: str,
    *,
    blocking: bool = False,
) -> dict[str, float]:
    """Update Bayesian preference model with both disliked result.
    Returns updated ratings dict derived from model weights.

    Args:
        ratings: Current ratings dictionary
        player_a: First name
        player_b: Second name
        blocking: If True, wait for model update to complete
    """
    record_comparison_instant(player_a, player_b, 2, blocking=blocking)

    if not blocking:
        return ratings.copy()

    try:
        rating_a = _compute_rating_for_name(player_a)
        rating_b = _compute_rating_for_name(player_b)
        ratings[player_a] = rating_a
        ratings[player_b] = rating_b
        database.update_rating_value(player_a, rating_a)
        database.update_rating_value(player_b, rating_b)
        return ratings.copy()
    except (RuntimeError, ValueError) as e:
        logger.warning("Failed to compute updated ratings: %s", e)
        return ratings.copy()
