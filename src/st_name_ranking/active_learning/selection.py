"""Pair selection and active-learning model access."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from st_name_ranking.learning.features import FeatureExtractor
from st_name_ranking.learning.model import BradleyTerryModel, initialize_model_if_needed
from st_name_ranking.persistence import database
from st_name_ranking.phonetic_similarity import phonetic_similarity

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable

MIN_NAMES_FOR_PAIR_SELECTION = 2
DEFAULT_PAIR_SAMPLE_SIZE = 50
_ACTIVE_LEARNING_STATE_LOCK = threading.RLock()


@dataclass(frozen=True)
class PairSelectionOptions:
    """Policy for tournament pair selection.

    sample_size limits model ranking to a random subset; None uses
    DEFAULT_PAIR_SAMPLE_SIZE capped to the number of candidate names.
    """

    batch_size: int = 1
    sample_size: int | None = None
    min_training_samples: int = 0
    fallback: str = "heuristic"


def get_active_learning_model() -> BradleyTerryModel:
    """Get or initialize the active learning model."""
    if get_active_learning_model._cache is None:
        with _ACTIVE_LEARNING_STATE_LOCK:
            if get_active_learning_model._cache is None:
                extractor = get_feature_extractor()
                feature_names = extractor.get_feature_names()
                get_active_learning_model._cache = initialize_model_if_needed(feature_names)
    return get_active_learning_model._cache


get_active_learning_model._cache = None


def get_feature_extractor() -> FeatureExtractor:
    """Get or initialize the feature extractor."""
    if get_feature_extractor._cache is None:
        with _ACTIVE_LEARNING_STATE_LOCK:
            if get_feature_extractor._cache is None:
                get_feature_extractor._cache = FeatureExtractor()
    return get_feature_extractor._cache


get_feature_extractor._cache = None


def reset_active_learning_state() -> None:
    """Clear active-learning singletons in one synchronized lifecycle step."""
    with _ACTIVE_LEARNING_STATE_LOCK:
        get_active_learning_model._cache = None
        get_feature_extractor._cache = None


def get_name_features(name: str) -> np.ndarray:
    """Extract features for a name by querying gender and origin from database."""
    extractor = get_feature_extractor()

    with database.get_connection() as conn:
        cursor = conn.execute(
            "SELECT gender, origin_region FROM names WHERE name = ?",
            (name,),
        )
        row = cursor.fetchone()

    gender, origin_region = row or (None, None)
    return extractor.extract(name, gender, origin_region)


def get_names_features(names: list[str]) -> np.ndarray:
    """Extract features for multiple names in batch."""
    extractor = get_feature_extractor()
    details = database.get_name_details_batch(names)
    genders = [detail.gender for detail in details]
    origins = [detail.origin_region for detail in details]
    return extractor.batch_extract(names, genders, origins)


@dataclass(frozen=True)
class PairSelectionDependencies:
    """Injectable dependencies for compatibility wrappers and tests."""

    model_provider: Callable[[], BradleyTerryModel] = get_active_learning_model
    features_provider: Callable[[list[str]], np.ndarray] = get_names_features
    comparison_count_provider: Callable[[str], int] = database.get_comparison_count
    heuristic_pair_provider: Callable[[list[str]], tuple[str, str] | None] | None = None
    single_pair_provider: Callable[[list[str], np.ndarray | None], tuple[str, str] | None] | None = None
    warning_logger: Callable[..., None] = logger.warning


def select_candidates(
    names: list[str],
    features: np.ndarray | None = None,
    sample_size: int | None = None,
    dependencies: PairSelectionDependencies | None = None,
) -> tuple[str, str]:
    """Select one active-learning candidate pair.

    sample_size limits model ranking to a random subset; None uses
    DEFAULT_PAIR_SAMPLE_SIZE capped to the number of candidate names.
    """
    pair = try_select_candidates(names, features, sample_size, dependencies)
    if pair is None:
        msg = f"Need at least {MIN_NAMES_FOR_PAIR_SELECTION} names"
        raise ValueError(msg)
    return pair


def try_select_candidates(
    names: list[str],
    features: np.ndarray | None = None,
    sample_size: int | None = None,
    dependencies: PairSelectionDependencies | None = None,
) -> tuple[str, str] | None:
    """Select one candidate pair, or return None when no pair is available."""
    pairs = select_candidate_pairs(
        names,
        features,
        PairSelectionOptions(batch_size=1, sample_size=sample_size),
        dependencies,
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

    sample_size limits model ranking to a random subset; None uses
    DEFAULT_PAIR_SAMPLE_SIZE capped to the number of candidate names.
    """
    resolved_options = options or PairSelectionOptions(
        batch_size=batch_size,
        sample_size=sample_size,
    )
    return select_candidate_pairs(names, features, resolved_options)


def select_candidate_pairs(
    names: list[str],
    features: np.ndarray | None = None,
    options: PairSelectionOptions | None = None,
    dependencies: PairSelectionDependencies | None = None,
) -> list[tuple[str, str]]:
    """Select one or more tournament pairs through the active-learning model."""
    if len(names) < MIN_NAMES_FOR_PAIR_SELECTION:
        return []

    options = options or PairSelectionOptions()
    dependencies = dependencies or PairSelectionDependencies()
    batch_size = max(options.batch_size, 1)
    sampled, sampled_indices = _sample_names(names, options.sample_size)

    try:
        model = dependencies.model_provider()
        training_samples = _model_training_samples(model)
        if options.min_training_samples > 0 and training_samples < options.min_training_samples:
            logger.info(
                "Active learning model has %d training samples; using %s fallback until %d samples",
                training_samples,
                options.fallback,
                options.min_training_samples,
            )
            return _fallback_pairs(names, features, batch_size, options.fallback, dependencies)

        sampled_features = (
            features[sampled_indices] if features is not None else dependencies.features_provider(sampled)
        )

        if batch_size == 1:
            pair = model.select_pair(sampled_features, sampled)
            return [(pair.name_a, pair.name_b)]

        pairs = model.select_top_k_pairs(sampled_features, sampled, k=batch_size)
        return [(pair.name_a, pair.name_b) for pair in pairs]
    except (RuntimeError, ValueError, AttributeError) as e:
        dependencies.warning_logger(
            "Active learning pair selection failed: %s. Falling back to %s selection.",
            e,
            options.fallback,
        )
        return _fallback_pairs(names, features, batch_size, options.fallback, dependencies)


def select_random_pair(names: list[str]) -> tuple[str, str]:
    """Select a random pair of names."""
    pairs = select_random_batch(names, 1)
    if not pairs:
        msg = f"Need at least {MIN_NAMES_FOR_PAIR_SELECTION} names"
        raise ValueError(msg)
    return pairs[0]


def select_random_batch(names: list[str], batch_size: int) -> list[tuple[str, str]]:
    """Select distinct random name pairs."""
    if len(names) < MIN_NAMES_FOR_PAIR_SELECTION:
        return []

    target_size = max(batch_size, 1)
    max_pairs = len(names) * (len(names) - 1) // 2
    target_size = min(target_size, max_pairs)
    rng = np.random.default_rng()
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    while len(pairs) < target_size:
        idx_a, idx_b = rng.choice(len(names), size=2, replace=False)
        pair = (names[idx_a], names[idx_b])
        normalized = (min(pair[0], pair[1]), max(pair[0], pair[1]))
        if normalized in seen:
            continue
        seen.add(normalized)
        pairs.append(pair)

    return pairs


def _sample_names(names: list[str], sample_size: int | None) -> tuple[list[str], list[int]]:
    effective_sample_size = min(
        max(sample_size if sample_size is not None else DEFAULT_PAIR_SAMPLE_SIZE, 2),
        len(names),
    )
    if len(names) == effective_sample_size:
        return names, list(range(len(names)))

    rng = np.random.default_rng()
    sampled_indices = list(
        rng.choice(len(names), size=effective_sample_size, replace=False),
    )
    return [names[i] for i in sampled_indices], sampled_indices


def _model_training_samples(model: BradleyTerryModel) -> int:
    training_samples = getattr(model.state, "training_samples", 0)
    return training_samples if isinstance(training_samples, int) else 0


def _fallback_pairs(
    names: list[str],
    features: np.ndarray | None,
    batch_size: int,
    fallback: str,
    dependencies: PairSelectionDependencies,
) -> list[tuple[str, str]]:
    if fallback == "random":
        return select_random_batch(names, batch_size)

    if batch_size > 1:
        if dependencies.single_pair_provider is not None:
            try:
                pair = dependencies.single_pair_provider(names, features)
            except ValueError:
                return []
            return [pair] if _has_pair(pair) else []

        pairs = select_candidate_pairs(
            names,
            features,
            PairSelectionOptions(batch_size=1),
            dependencies,
        )
        return pairs[:1]

    pair = _select_candidates_fallback(names, dependencies)
    return [pair] if _has_pair(pair) else []


def _has_pair(pair: tuple[str, str] | None) -> bool:
    return pair is not None and bool(pair[0]) and bool(pair[1])


def _select_candidates_fallback(
    names: list[str],
    dependencies: PairSelectionDependencies | None = None,
) -> tuple[str, str] | None:
    dependencies = dependencies or PairSelectionDependencies()
    if dependencies.heuristic_pair_provider is not None:
        return dependencies.heuristic_pair_provider(names)

    if len(names) < MIN_NAMES_FOR_PAIR_SELECTION:
        return None

    rng = np.random.default_rng()
    utilities = {name: 1.0 / (dependencies.comparison_count_provider(name) + 1) for name in names}
    n_pairs = min(100, len(names) * (len(names) - 1) // 2)
    best_pair: tuple[str, str] | None = None
    best_score = -1.0

    for _ in range(n_pairs):
        i, j = rng.choice(len(names), size=2, replace=False)
        a = names[i]
        b = names[j]
        pair_score = utilities[a] + utilities[b] + phonetic_similarity(a, b)

        if pair_score > best_score:
            best_score = pair_score
            best_pair = (a, b)

    if best_pair is None:
        return select_random_pair(names)

    return best_pair
