"""Model update and rating persistence services without Streamlit dependencies."""

from __future__ import annotations

import functools
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, wait

from st_name_ranking import database
from st_name_ranking.pair_selection import (
    get_active_learning_model,
    get_name_features,
    get_names_features,
)

logger = logging.getLogger(__name__)

_model_update_lock = threading.Lock()
BOTH_DISLIKED_PREFERENCE = 2


@functools.lru_cache
def get_thread_executor() -> ThreadPoolExecutor:
    """Shared thread pool for background model/rating updates."""
    return ThreadPoolExecutor(max_workers=2)


def update_model_async(name_a: str, name_b: str, preference: int) -> None:
    """Queue a model update in the background."""

    def _update() -> None:
        with _model_update_lock:
            model = get_active_learning_model()
            features_a = get_name_features(name_a)
            features_b = get_name_features(name_b)
            model.update(features_a, features_b, preference)
            model.save_to_db()

    get_thread_executor().submit(_update)


def record_comparison_instant(
    name_a: str,
    name_b: str,
    preference: int,
    *,
    blocking: bool = False,
) -> None:
    """Record a comparison synchronously and update model/ratings in the background."""
    database.record_comparison(name_a, name_b, preference)

    future = get_thread_executor().submit(_update_model_sync, name_a, name_b, preference)
    ratings_future = get_thread_executor().submit(_update_ratings_from_model)

    if blocking:
        wait([future, ratings_future])


def _update_model_sync(name_a: str, name_b: str, preference: int) -> None:
    """Update the preference model while holding the model update lock."""
    with _model_update_lock:
        try:
            model = get_active_learning_model()
            features_a = get_name_features(name_a)
            features_b = get_name_features(name_b)

            if preference == BOTH_DISLIKED_PREFERENCE:
                model.update_both_disliked(features_a, features_b)
            else:
                model.update(features_a, features_b, preference)

            model.save_to_db()
        except (RuntimeError, ValueError, AttributeError):
            logger.exception("Failed to update model in background")


def _compute_rating_for_name(name: str) -> float:
    """Compute a display rating for one name from current model utility."""
    model = get_active_learning_model()
    features = get_name_features(name)
    utility = model.get_utility(features.reshape(1, -1))[0]
    return 1500 + utility * 500


def update_model_and_save(winner: str, loser: str) -> None:
    """Update the model for a winner/loser preference and persist it."""
    try:
        model = get_active_learning_model()
        features_a = get_name_features(winner)
        features_b = get_name_features(loser)
        model.update(features_a, features_b, -1)
        model.save_to_db()
    except (RuntimeError, ValueError, AttributeError):
        logger.exception("Failed to update model")


def update_model_draw_and_save(player_a: str, player_b: str) -> None:
    """Update the model for an equal-preference vote and persist it."""
    try:
        model = get_active_learning_model()
        features_a = get_name_features(player_a)
        features_b = get_name_features(player_b)
        model.update(features_a, features_b, 0)
        model.save_to_db()
    except (RuntimeError, ValueError, AttributeError):
        logger.exception("Failed to update model for draw")


def update_model_down_and_save(player_a: str, player_b: str) -> None:
    """Update the model for a both-disliked vote and persist it."""
    try:
        model = get_active_learning_model()
        features_a = get_name_features(player_a)
        features_b = get_name_features(player_b)
        model.update_both_disliked(features_a, features_b)
        model.save_to_db()
    except (RuntimeError, ValueError, AttributeError):
        logger.exception("Failed to update model for both disliked")


def _update_ratings_from_model() -> None:
    """Refresh stored display ratings from current model utilities."""
    try:
        model = get_active_learning_model()

        with database.get_connection() as conn:
            cursor = conn.execute("SELECT name FROM names")
            names = [row[0] for row in cursor.fetchall()]

        if not names:
            return

        features = get_names_features(names)
        utilities = model.get_utility(features)
        ratings_dict = {name: 1500 + utility * 500 for name, utility in zip(names, utilities, strict=False)}
        database.update_ratings_batch_values(ratings_dict)

    except (RuntimeError, ValueError) as e:
        logger.warning("Failed to update ratings from model: %s", e)


def update_preference_and_save(
    ratings: dict[str, float],
    winner: str,
    loser: str,
    *,
    blocking: bool = False,
) -> dict[str, float]:
    """Record a winner/loser preference and return updated display ratings."""
    record_comparison_instant(winner, loser, -1, blocking=blocking)

    if not blocking:
        return ratings.copy()

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
    """Record an equal-preference vote and return updated display ratings."""
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
    """Record a both-disliked vote and return updated display ratings."""
    record_comparison_instant(player_a, player_b, BOTH_DISLIKED_PREFERENCE, blocking=blocking)

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
