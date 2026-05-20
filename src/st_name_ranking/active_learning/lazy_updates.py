"""Model update and rating persistence services without Streamlit dependencies."""

from __future__ import annotations

import functools
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from st_name_ranking.active_learning.selection import (
    get_active_learning_model,
    get_name_features,
    get_names_features,
)
from st_name_ranking.persistence import database

logger = logging.getLogger(__name__)

_model_update_lock = threading.Lock()
BOTH_DISLIKED_PREFERENCE = 2


@dataclass(frozen=True)
class ModelUpdateStatus:
    """Outcome of recording a comparison and refreshing model-derived state."""

    recorded: bool
    model_updated: bool | None
    ratings_fresh: bool | None
    fallback_used: bool = False
    error: str | None = None


@functools.lru_cache
def get_thread_executor() -> ThreadPoolExecutor:
    """Shared thread pool for background model/rating updates."""
    return ThreadPoolExecutor(max_workers=2)


def record_comparison_instant(
    name_a: str,
    name_b: str,
    preference: int,
    *,
    blocking: bool = False,
) -> ModelUpdateStatus:
    """Record a comparison synchronously and update model/ratings."""
    try:
        database.record_comparison(name_a, name_b, preference)
    except (RuntimeError, ValueError) as e:
        logger.exception("Failed to record comparison")
        return ModelUpdateStatus(
            recorded=False,
            model_updated=False,
            ratings_fresh=False,
            fallback_used=True,
            error=str(e),
        )

    if not blocking:
        get_thread_executor().submit(_update_model_then_refresh_ratings, name_a, name_b, preference)
        return ModelUpdateStatus(recorded=True, model_updated=None, ratings_fresh=None)

    model_updated, ratings_fresh = _update_model_then_refresh_ratings(name_a, name_b, preference)
    return ModelUpdateStatus(
        recorded=True,
        model_updated=model_updated,
        ratings_fresh=ratings_fresh,
        fallback_used=not (model_updated and ratings_fresh),
        error=None if model_updated and ratings_fresh else "model or rating refresh failed",
    )


def _update_model_then_refresh_ratings(name_a: str, name_b: str, preference: int) -> tuple[bool, bool]:
    """Update the model first, then refresh ratings from the persisted model state."""
    model_updated = _update_model_sync(name_a, name_b, preference)
    if not model_updated:
        return False, False
    return True, _update_ratings_from_model()


def _update_model_sync(name_a: str, name_b: str, preference: int) -> bool:
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
            return False
        else:
            return True


def _update_ratings_from_model() -> bool:
    """Refresh stored display ratings from current model utilities."""
    try:
        model = get_active_learning_model()

        with database.get_connection() as conn:
            cursor = conn.execute("SELECT name FROM names")
            names = [row[0] for row in cursor.fetchall()]

        if not names:
            return True

        features = get_names_features(names)
        utilities = model.get_utility(features)
        ratings_dict = {name: 1500 + utility * 500 for name, utility in zip(names, utilities, strict=False)}
        database.update_ratings_batch_values(ratings_dict)

    except (RuntimeError, ValueError) as e:
        logger.warning("Failed to update ratings from model: %s", e)
        return False
    else:
        return True
