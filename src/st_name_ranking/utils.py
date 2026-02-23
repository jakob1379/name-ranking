"""Utility functions for the name ranking application."""

import logging
import sqlite3
import subprocess
import time

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

# Global instances for model and feature extractor
_model = None
_feature_extractor = None


def get_active_learning_model() -> BradleyTerryModel:
    """Get or initialize the active learning model."""
    global _model
    if _model is None:
        # Initialize feature extractor first to get feature names
        extractor = get_feature_extractor()
        feature_names = extractor.get_feature_names()
        _model = initialize_model_if_needed(feature_names)
    return _model


def get_feature_extractor() -> FeatureExtractor:
    """Get or initialize the feature extractor."""
    global _feature_extractor
    if _feature_extractor is None:
        _feature_extractor = FeatureExtractor()
    return _feature_extractor


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


def pull_submodule_updates(classify_origins: bool = False) -> bool:
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
                    from st_name_ranking.classify_origins import classify_all_names  # noqa: PLC0415

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
    except subprocess.SubprocessError as e:
        st.toast(
            f"Error pulling submodule: {e}",
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
    # Sample a subset of names for efficiency (max 50)
    sample_size = min(50, len(names))
    if len(names) == sample_size:
        sampled = names
        sampled_indices = list(range(len(names)))
    else:
        sampled_indices = list(
            rng.choice(len(names), size=sample_size, replace=False),
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
        return pair.name_a, pair.name_b
    except (RuntimeError, ValueError, AttributeError) as e:
        logger.warning(
            "Active learning pair selection failed: %s. Falling back to basic selection.",
            e,
        )
        # Fallback to basic selection (doesn't use features)
        return _select_candidates_fallback(sampled)


def select_candidate_batch(
    names: list[str],
    features: np.ndarray | None = None,
    batch_size: int = 3,
) -> list[tuple[str, str]]:
    """Select a batch of candidate pairs for active learning.
    Returns list of (name_a, name_b) pairs.
    """
    if len(names) < MIN_NAMES_FOR_PAIR_SELECTION:
        return []
    batch_size = max(batch_size, 1)
    rng = np.random.default_rng()

    # Sample a subset of names for efficiency (max 50)
    sample_size = min(50, len(names))
    if len(names) == sample_size:
        sampled = names
        sampled_indices = list(range(len(names)))
    else:
        sampled_indices = list(
            rng.choice(len(names), size=sample_size, replace=False),
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
        pair = select_candidates(names, features)
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


def update_preference_and_save(
    ratings: dict[str, float],
    winner: str,
    loser: str,
) -> dict[str, float]:
    """Update Bayesian preference model with comparison result.
    Returns updated ratings dict derived from model weights.
    """
    # Update the model
    update_model_and_save(winner, loser)

    # Record comparison in database
    try:
        database.record_comparison(winner, loser, -1)
    except sqlite3.Error as e:
        logger.warning("Failed to record comparison: %s", e)

    # Compute new ratings only for the two names involved
    try:
        winner_rating = _compute_rating_for_name(winner)
        loser_rating = _compute_rating_for_name(loser)

        # Update in-memory ratings dict
        ratings[winner] = winner_rating
        ratings[loser] = loser_rating

        # Update database for these two names
        database.update_rating_value(winner, winner_rating)
        database.update_rating_value(loser, loser_rating)

        logger.debug(
            "Updated ratings: %s=%.1f, %s=%.1f",
            winner,
            winner_rating,
            loser,
            loser_rating,
        )
        return ratings.copy()
    except (RuntimeError, ValueError) as e:
        logger.warning("Failed to compute updated ratings: %s", e)
        # Return original ratings as fallback
        return ratings.copy()


def update_preference_draw_and_save(
    ratings: dict[str, float],
    player_a: str,
    player_b: str,
) -> dict[str, float]:
    """Update Bayesian preference model with draw result.
    Returns updated ratings dict derived from model weights.
    """
    # Update the model
    update_model_draw_and_save(player_a, player_b)

    # Record comparison in database
    try:
        database.record_comparison(player_a, player_b, 0)
    except sqlite3.Error as e:
        logger.warning("Failed to record draw comparison: %s", e)

    # Compute new ratings only for the two names involved
    try:
        rating_a = _compute_rating_for_name(player_a)
        rating_b = _compute_rating_for_name(player_b)

        # Update in-memory ratings dict
        ratings[player_a] = rating_a
        ratings[player_b] = rating_b

        # Update database for these two names
        database.update_rating_value(player_a, rating_a)
        database.update_rating_value(player_b, rating_b)

        logger.debug(
            "Updated draw ratings: %s=%.1f, %s=%.1f",
            player_a,
            rating_a,
            player_b,
            rating_b,
        )
        return ratings.copy()
    except (RuntimeError, ValueError) as e:
        logger.warning("Failed to compute updated ratings: %s", e)
        # Return original ratings as fallback
        return ratings.copy()


def update_preference_down_and_save(
    ratings: dict[str, float],
    player_a: str,
    player_b: str,
) -> dict[str, float]:
    """Update Bayesian preference model with both disliked result.
    Returns updated ratings dict derived from model weights.
    """
    # Update the model
    update_model_down_and_save(player_a, player_b)

    # Record down comparison in database (preference=2)
    try:
        database.record_comparison(player_a, player_b, 2)
    except sqlite3.Error as e:
        logger.warning("Failed to record down comparison: %s", e)

    # Compute new ratings only for the two names involved
    try:
        rating_a = _compute_rating_for_name(player_a)
        rating_b = _compute_rating_for_name(player_b)

        # Update in-memory ratings dict
        ratings[player_a] = rating_a
        ratings[player_b] = rating_b

        # Update database for these two names
        database.update_rating_value(player_a, rating_a)
        database.update_rating_value(player_b, rating_b)

        logger.debug(
            "Updated down ratings: %s=%.1f, %s=%.1f",
            player_a,
            rating_a,
            player_b,
            rating_b,
        )
        return ratings.copy()
    except (RuntimeError, ValueError) as e:
        logger.warning("Failed to compute updated ratings: %s", e)
        # Return original ratings as fallback
        return ratings.copy()
