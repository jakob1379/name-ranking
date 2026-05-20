"""Utility functions for the name ranking application."""

import logging

import numpy as np

from st_name_ranking import database
from st_name_ranking.active_learning.lazy_updates import record_comparison_instant
from st_name_ranking.active_learning.selection import (
    PairSelectionDependencies,
    PairSelectionOptions,
    get_active_learning_model,
    get_feature_extractor,
    get_name_features,
    get_names_features,
)
from st_name_ranking.active_learning.selection import (
    select_candidate_pairs as _select_candidate_pairs,
)
from st_name_ranking.active_learning.selection import (
    select_random_batch as _select_random_batch,
)
from st_name_ranking.active_learning.selection import (
    select_random_pair as _select_random_pair,
)
from st_name_ranking.interface.app_actions import (
    pull_submodule_updates,
    setup_session_state,
    sync_names_from_submodule,
)
from st_name_ranking.phonetic_similarity import phonetic_similarity

logger = logging.getLogger(__name__)

__all__ = [
    "PairSelectionOptions",
    "get_active_learning_model",
    "get_feature_extractor",
    "get_name_features",
    "get_names_features",
    "pull_submodule_updates",
    "record_comparison_instant",
    "setup_session_state",
    "sync_names_from_submodule",
    "try_select_candidates",
]

# Minimum names required for pair selection
MIN_NAMES_FOR_PAIR_SELECTION = 2


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
    pair = try_select_candidates(names, features, sample_size)
    if pair is None:
        msg = f"Need at least {MIN_NAMES_FOR_PAIR_SELECTION} names"
        raise ValueError(msg)
    return pair


def try_select_candidates(
    names: list[str],
    features: np.ndarray | None = None,
    sample_size: int | None = None,
) -> tuple[str, str] | None:
    """Select one candidate pair, or return None when no pair is available."""
    pairs = _select_candidate_pairs(
        names,
        features,
        PairSelectionOptions(batch_size=1, sample_size=sample_size),
        _pair_selection_dependencies(),
    )
    return pairs[0] if pairs else None


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


def _select_candidates_fallback(names: list[str]) -> tuple[str, str] | None:
    """Fallback candidate selection using comparison counts and phonetic similarity.
    Used when active learning model fails.
    """
    if len(names) < MIN_NAMES_FOR_PAIR_SELECTION:
        return None
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
    best_pair: tuple[str, str] | None = None
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

    if best_pair is None:
        return tuple(rng.choice(names, size=2, replace=False))

    return best_pair
