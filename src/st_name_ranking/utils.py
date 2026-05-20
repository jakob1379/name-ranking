"""Utility functions for the name ranking application."""

import logging
import subprocess
import time

import numpy as np
import streamlit as st

from st_name_ranking import database
from st_name_ranking.data_loader import initialize_or_load_ratings
from st_name_ranking.model_service import (
    _compute_rating_for_name,
    _update_model_sync,
    _update_ratings_from_model,
    get_thread_executor,
    record_comparison_instant,
    update_model_and_save,
    update_model_async,
    update_model_down_and_save,
    update_model_draw_and_save,
    update_preference_and_save,
    update_preference_down_and_save,
    update_preference_draw_and_save,
)
from st_name_ranking.pair_selection import (
    PairSelectionDependencies,
    PairSelectionOptions,
    get_active_learning_model,
    get_feature_extractor,
    get_name_features,
    get_names_features,
)
from st_name_ranking.pair_selection import (
    select_candidate_pairs as _select_candidate_pairs,
)
from st_name_ranking.pair_selection import (
    select_random_batch as _select_random_batch,
)
from st_name_ranking.pair_selection import (
    select_random_pair as _select_random_pair,
)
from st_name_ranking.phonetic_similarity import phonetic_similarity

logger = logging.getLogger(__name__)

__all__ = [
    "PairSelectionOptions",
    "_compute_rating_for_name",
    "_update_model_sync",
    "_update_ratings_from_model",
    "get_active_learning_model",
    "get_feature_extractor",
    "get_name_features",
    "get_names_features",
    "get_thread_executor",
    "record_comparison_instant",
    "update_model_and_save",
    "update_model_async",
    "update_model_down_and_save",
    "update_model_draw_and_save",
    "update_preference_and_save",
    "update_preference_down_and_save",
    "update_preference_draw_and_save",
]

# Minimum names required for pair selection
MIN_NAMES_FOR_PAIR_SELECTION = 2


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
    """Select one active-learning candidate pair.

    Args:
        names: List of candidate names
        features: Optional precomputed feature matrix of shape (len(names), n_features).
                 If None, features will be computed on the fly.
        sample_size: Optional model-ranking subset size. None uses the pair-selection default capped to the number of names.

    """
    pairs = _select_candidate_pairs(
        names,
        features,
        PairSelectionOptions(batch_size=1, sample_size=sample_size),
        _pair_selection_dependencies(),
    )
    return pairs[0] if pairs else ("", "")


def select_candidate_batch(
    names: list[str],
    features: np.ndarray | None = None,
    batch_size: int = 3,
    sample_size: int | None = None,
    options: PairSelectionOptions | None = None,
) -> list[tuple[str, str]]:
    """Select a batch of candidate pairs for active learning.

    sample_size limits model ranking to a random subset unless options is supplied.
    Returns list of (name_a, name_b) pairs.
    """
    resolved_options = options or PairSelectionOptions(
        batch_size=batch_size,
        sample_size=sample_size,
    )
    return _select_candidate_pairs(
        names,
        features,
        resolved_options,
        _pair_selection_dependencies(),
    )


def select_random_pair(names: list[str]) -> tuple[str, str]:
    """Select a random pair of names."""
    return _select_random_pair(names)


def select_random_batch(names: list[str], batch_size: int) -> list[tuple[str, str]]:
    """Select distinct random name pairs."""
    return _select_random_batch(names, batch_size)


def _pair_selection_dependencies() -> PairSelectionDependencies:
    return PairSelectionDependencies(
        model_provider=get_active_learning_model,
        features_provider=get_names_features,
        comparison_count_provider=database.get_comparison_count,
        heuristic_pair_provider=_select_candidates_fallback,
        single_pair_provider=select_candidates,
        warning_logger=logger.warning,
    )


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
