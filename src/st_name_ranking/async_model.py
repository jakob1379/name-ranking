"""Lazy/asynchronous model updates for tournament optimization.

This module provides lazy model updating to fix tournament delays. Instead of
updating the model synchronously after each comparison, comparisons are batched
and the model is updated asynchronously in the background.

Key features:
- LazyModelUpdater: Batches comparisons and updates model asynchronously
- Fast pair selection: Falls back to random selection when model is stale
- Session state integration: Works seamlessly with Streamlit's session state
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import streamlit as st

from st_name_ranking import database
from st_name_ranking.model import BradleyTerryModel, initialize_model_if_needed
from st_name_ranking.types import NamePair
from st_name_ranking.utils import get_feature_extractor, get_name_features

logger = logging.getLogger(__name__)

# Constants for lazy model updating
LAZY_MODEL_UPDATER_KEY: Final[str] = "st_name_ranking_lazy_model_updater"
DEFAULT_BATCH_SIZE: Final[int] = 5
DEFAULT_MAX_PENDING: Final[int] = 20
DEFAULT_STALE_THRESHOLD_SECONDS: Final[int] = 30
MIN_NAMES_FOR_PAIR_SELECTION: Final[int] = 2


def select_random_pair(names: list[str]) -> tuple[str, str]:
    """Select a random pair of names. O(1) random selection, no model needed.

    Args:
        names: List of names to select from

    Returns:
        Tuple of (name_a, name_b) selected randomly

    Raises:
        ValueError: If fewer than 2 names are provided
    """
    if len(names) < MIN_NAMES_FOR_PAIR_SELECTION:
        msg = f"Need at least {MIN_NAMES_FOR_PAIR_SELECTION} names for pair selection"
        raise ValueError(msg)

    rng = np.random.default_rng()
    idx_a, idx_b = rng.choice(len(names), size=2, replace=False)
    return names[idx_a], names[idx_b]


def select_random_batch(names: list[str], batch_size: int) -> list[tuple[str, str]]:
    """Select a batch of random name pairs. O(n) batch random selection.

    Args:
        names: List of names to select from
        batch_size: Number of pairs to select

    Returns:
        List of (name_a, name_b) tuples

    Raises:
        ValueError: If fewer than 2 names are provided
    """
    if len(names) < MIN_NAMES_FOR_PAIR_SELECTION:
        msg = f"Need at least {MIN_NAMES_FOR_PAIR_SELECTION} names for pair selection"
        raise ValueError(msg)

    rng = np.random.default_rng()
    pairs: list[tuple[str, str]] = []
    attempts = 0
    max_attempts = batch_size * 10  # Prevent infinite loops

    while len(pairs) < batch_size and attempts < max_attempts:
        attempts += 1
        idx_a, idx_b = rng.choice(len(names), size=2, replace=False)
        pair = (names[idx_a], names[idx_b])
        # Normalize order to avoid duplicates
        normalized = tuple(sorted(pair))  # type: ignore[type-var]
        if normalized not in pairs:
            pairs.append(normalized)  # type: ignore[arg-type]

    return pairs


@dataclass
class ComparisonRecord:
    """A single comparison record waiting to be processed.

    Attributes:
        name_a: First name in comparison
        name_b: Second name in comparison
        preference: Preference value (-1=a wins, 0=draw, 1=b wins, 2=both disliked)
        timestamp: When the comparison was recorded
    """

    name_a: str
    name_b: str
    preference: int
    timestamp: float = field(default_factory=time.time)


class LazyModelUpdater:
    """Batches comparisons and updates model asynchronously.

    This class solves the 10s tournament delay problem by:
    1. Recording comparisons immediately without blocking
    2. Batching comparisons for efficient model updates
    3. Updating the model asynchronously in the background
    4. Falling back to fast random selection when model is stale

    Attributes:
        batch_size: Number of comparisons before triggering model update
        max_pending: Maximum pending comparisons before forced update
        stale_threshold_seconds: How long before model is considered stale
        pending_comparisons: Queue of comparisons waiting to be processed
        last_update_time: Timestamp of last model update
        update_lock: Thread lock for safe async updates
        _model: Cached model instance
    """

    def __init__(
        self,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_pending: int = DEFAULT_MAX_PENDING,
        stale_threshold_seconds: int = DEFAULT_STALE_THRESHOLD_SECONDS,
    ) -> None:
        """Initialize the lazy model updater.

        Args:
            batch_size: Number of comparisons to batch before updating (default 5)
            max_pending: Maximum pending comparisons before forced update (default 20)
            stale_threshold_seconds: Seconds before model is considered stale (default 30)
        """
        self.batch_size = batch_size
        self.max_pending = max_pending
        self.stale_threshold_seconds = stale_threshold_seconds
        self.pending_comparisons: deque[ComparisonRecord] = deque()
        self.last_update_time = time.time()
        self.update_lock = threading.Lock()
        self._model: BradleyTerryModel | None = None
        self._background_thread: threading.Thread | None = None

    def record_comparison(self, name_a: str, name_b: str, preference: int) -> None:
        """Fast - just stores comparison, no model update.

        This method is designed to be called from UI button handlers
        and returns immediately without blocking.

        Args:
            name_a: First name in comparison
            name_b: Second name in comparison
            preference: Preference value (-1=a wins, 0=draw, 1=b wins, 2=both disliked)
        """
        record = ComparisonRecord(name_a, name_b, preference)
        self.pending_comparisons.append(record)

        # Record to database immediately (fast operation)
        try:
            # Normalize preference for database: 2 (both disliked) is stored as-is
            db_preference = preference
            database.record_comparison(name_a, name_b, db_preference)
        except Exception:
            logger.exception("Failed to record comparison to database")
            # Don't crash the UI, just log

        logger.debug(
            "Recorded comparison %s vs %s (pref=%d), pending=%d",
            name_a,
            name_b,
            preference,
            len(self.pending_comparisons),
        )

    def should_update(self) -> bool:
        """Check if we should update model now.

        Returns:
            True if model should be updated (batch size reached or max pending exceeded)
        """
        if len(self.pending_comparisons) >= self.batch_size:
            return True
        if len(self.pending_comparisons) >= self.max_pending:
            logger.warning(
                "Forced update: %d pending comparisons exceeds max %d",
                len(self.pending_comparisons),
                self.max_pending,
            )
            return True
        return False

    def is_model_stale(self) -> bool:
        """Check if model is stale (hasn't been updated recently).

        Returns:
            True if model hasn't been updated in stale_threshold_seconds
        """
        return time.time() - self.last_update_time > self.stale_threshold_seconds

    def update_model(self, *, force: bool = False) -> bool:
        """Actually update the model with pending comparisons.

        This method can be called synchronously or asynchronously.
        Uses a thread lock to prevent concurrent updates.

        Args:
            force: If True, update even if batch size not reached

        Returns:
            True if update was performed, False otherwise
        """
        with self.update_lock:
            if not force and not self.should_update():
                return False

            if not self.pending_comparisons:
                return False

            # Copy and clear pending comparisons
            comparisons = list(self.pending_comparisons)
            self.pending_comparisons.clear()

        # Perform update outside the lock to allow new comparisons to be recorded
        try:
            self._perform_update(comparisons)
            self.last_update_time = time.time()
            logger.info(
                "Model updated with %d comparisons at %.2f",
                len(comparisons),
                self.last_update_time,
            )
            return True
        except Exception:
            logger.exception("Failed to update model with %d comparisons", len(comparisons))
            # Restore comparisons for retry
            with self.update_lock:
                self.pending_comparisons.extendleft(comparisons)
            return False

    def _perform_update(self, comparisons: list[ComparisonRecord]) -> None:
        """Perform the actual model update.

        Args:
            comparisons: List of comparison records to process
        """
        # Get or initialize model
        if self._model is None:
            extractor = get_feature_extractor()
            feature_names = extractor.get_feature_names()
            self._model = initialize_model_if_needed(feature_names)

        model = self._model

        # Prepare batch for model update
        batch: list[tuple[np.ndarray, np.ndarray, int]] = []
        for record in comparisons:
            features_a = get_name_features(record.name_a)
            features_b = get_name_features(record.name_b)

            # Convert preference to model format
            if record.preference == 2:  # Both disliked
                # Handle both disliked separately
                d = len(features_a)
                neutral = np.zeros(d, dtype=features_a.dtype)
                batch.append((features_a, neutral, 1))  # neutral preferred over a
                batch.append((features_b, neutral, 1))  # neutral preferred over b
            else:
                # Normal comparison: -1=a wins, 0=draw, 1=b wins
                batch.append((features_a, features_b, record.preference))

        # Update model with batch
        model.update_batch(batch)

        # Save to database
        model.save_to_db()

        # Update ratings table from model
        self._update_ratings_from_model()

    def _update_ratings_from_model(self) -> None:
        """Update ratings table with current model utilities for all names."""
        try:
            model = self._model
            if model is None:
                return

            # Get all names from database
            with database.get_connection() as conn:
                cursor = conn.execute("SELECT name FROM names")
                names = [row[0] for row in cursor.fetchall()]

            if not names:
                return

            # Extract features for all names (batch)
            from st_name_ranking.utils import get_names_features

            features = get_names_features(names)

            # Get model utilities
            utilities = model.get_utility(features)

            # Update ratings table in batch
            ratings_dict: dict[str, float] = {}
            for name, utility in zip(names, utilities):
                # Convert utility to preference score scale (1500 ± 500)
                rating = 1500 + utility * 500
                ratings_dict[name] = rating

            database.update_ratings_batch_values(ratings_dict)

        except Exception:
            logger.exception("Failed to update ratings from model")

    def update_model_async(self, *, force: bool = False) -> None:
        """Update model asynchronously in a background thread.

        Args:
            force: If True, update even if batch size not reached
        """
        if self._background_thread is not None and self._background_thread.is_alive():
            logger.debug("Background update already in progress, skipping")
            return

        if not force and not self.should_update():
            return

        def update_worker() -> None:
            try:
                self.update_model(force=force)
            except Exception:
                logger.exception("Background model update failed")

        self._background_thread = threading.Thread(target=update_worker, daemon=True)
        self._background_thread.start()
        logger.debug("Started background model update thread")

    def get_pair_selection_mode(self) -> str:
        """Return 'random' or 'model' based on freshness.

        Returns:
            'random' if model is stale or too many pending comparisons
            'model' if model is fresh and up-to-date
        """
        # If too many pending comparisons, model is behind
        if len(self.pending_comparisons) >= self.batch_size:
            return "random"

        # If model hasn't been updated recently, use random
        if self.is_model_stale():
            return "random"

        return "model"

    def select_pair(
        self,
        names: list[str],
        features: np.ndarray | None = None,
    ) -> tuple[str, str]:
        """Select a pair using appropriate strategy based on model freshness.

        Args:
            names: List of candidate names
            features: Optional precomputed feature matrix

        Returns:
            Tuple of (name_a, name_b)
        """
        mode = self.get_pair_selection_mode()

        if mode == "random":
            return select_random_pair(names)

        # Use model-based selection
        try:
            if self._model is None:
                extractor = get_feature_extractor()
                feature_names = extractor.get_feature_names()
                self._model = initialize_model_if_needed(feature_names)

            model = self._model

            # Sample a subset for efficiency
            rng = np.random.default_rng()
            sample_size = min(50, len(names))
            if len(names) == sample_size:
                sampled = names
                sampled_indices = list(range(len(names)))
            else:
                sampled_indices = list(rng.choice(len(names), size=sample_size, replace=False))
                sampled = [names[i] for i in sampled_indices]

            # Get features for sampled names
            if features is not None:
                sampled_features = features[sampled_indices]
            else:
                from st_name_ranking.utils import get_names_features

                sampled_features = get_names_features(sampled)

            # Use model's Thompson sampling for pair selection
            pair: NamePair = model.select_pair(sampled_features, sampled)
            return pair.name_a, pair.name_b

        except Exception:
            logger.exception("Model-based pair selection failed, falling back to random")
            return select_random_pair(names)

    def select_batch(
        self,
        names: list[str],
        batch_size: int,
        features: np.ndarray | None = None,
    ) -> list[tuple[str, str]]:
        """Select a batch of pairs using appropriate strategy.

        Args:
            names: List of candidate names
            batch_size: Number of pairs to select
            features: Optional precomputed feature matrix

        Returns:
            List of (name_a, name_b) tuples
        """
        mode = self.get_pair_selection_mode()

        if mode == "random":
            return select_random_batch(names, batch_size)

        # Use model-based selection
        try:
            if self._model is None:
                extractor = get_feature_extractor()
                feature_names = extractor.get_feature_names()
                self._model = initialize_model_if_needed(feature_names)

            model = self._model

            # Sample a subset for efficiency
            rng = np.random.default_rng()
            sample_size = min(50, len(names))
            if len(names) == sample_size:
                sampled = names
                sampled_indices = list(range(len(names)))
            else:
                sampled_indices = list(rng.choice(len(names), size=sample_size, replace=False))
                sampled = [names[i] for i in sampled_indices]

            # Get features for sampled names
            if features is not None:
                sampled_features = features[sampled_indices]
            else:
                from st_name_ranking.utils import get_names_features

                sampled_features = get_names_features(sampled)

            # Use model's top-k selection
            pairs = model.select_top_k_pairs(sampled_features, sampled, k=batch_size)
            return [(pair.name_a, pair.name_b) for pair in pairs]

        except Exception:
            logger.exception("Model-based batch selection failed, falling back to random")
            return select_random_batch(names, batch_size)

    def wait_for_update(self, timeout: float = 5.0) -> bool:
        """Wait for any pending background update to complete.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if update completed, False if timed out
        """
        if self._background_thread is None or not self._background_thread.is_alive():
            return True

        self._background_thread.join(timeout=timeout)
        return not self._background_thread.is_alive()

    def get_stats(self) -> dict[str, int | float]:
        """Get current updater statistics.

        Returns:
            Dictionary with pending count, last update time, etc.
        """
        return {
            "pending_comparisons": len(self.pending_comparisons),
            "batch_size": self.batch_size,
            "last_update_time": self.last_update_time,
            "seconds_since_update": time.time() - self.last_update_time,
            "is_stale": self.is_model_stale(),
            "has_background_thread": self._background_thread is not None and self._background_thread.is_alive(),
        }


def get_lazy_updater(
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_pending: int = DEFAULT_MAX_PENDING,
) -> LazyModelUpdater:
    """Get or create LazyModelUpdater from session state.

    Args:
        batch_size: Number of comparisons to batch before updating
        max_pending: Maximum pending comparisons before forced update

    Returns:
        LazyModelUpdater instance from session state
    """
    if LAZY_MODEL_UPDATER_KEY in st.session_state:
        return st.session_state[LAZY_MODEL_UPDATER_KEY]

    updater = LazyModelUpdater(batch_size=batch_size, max_pending=max_pending)
    st.session_state[LAZY_MODEL_UPDATER_KEY] = updater
    logger.info("Created new LazyModelUpdater (batch_size=%d)", batch_size)
    return updater


def record_comparison_lazy(
    name_a: str,
    name_b: str,
    preference: int,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    """Fast record without blocking.

    Records a comparison immediately and triggers async model update if needed.
    Designed to be called from UI button handlers for instant response.

    Args:
        name_a: First name in comparison
        name_b: Second name in comparison
        preference: Preference value (-1=a wins, 0=draw, 1=b wins, 2=both disliked)
        batch_size: Batch size for the lazy updater
    """
    updater = get_lazy_updater(batch_size=batch_size)
    updater.record_comparison(name_a, name_b, preference)

    # Trigger async update if batch is full
    if updater.should_update():
        updater.update_model_async()


def force_model_update(*, wait: bool = False, timeout: float = 5.0) -> bool:
    """Force an immediate model update.

    Args:
        wait: If True, wait for the update to complete
        timeout: Maximum time to wait if wait=True

    Returns:
        True if update was performed/completed
    """
    if LAZY_MODEL_UPDATER_KEY not in st.session_state:
        return False

    updater: LazyModelUpdater = st.session_state[LAZY_MODEL_UPDATER_KEY]

    if wait:
        return updater.update_model(force=True)

    updater.update_model_async(force=True)
    return True


def select_lazy_pair(
    names: list[str],
    features: np.ndarray | None = None,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[str, str]:
    """Select a pair using lazy updater's strategy.

    This is a convenience function that uses the lazy updater
    to select pairs with automatic fallback to random when model is stale.

    Args:
        names: List of candidate names
        features: Optional precomputed feature matrix
        batch_size: Batch size for the lazy updater

    Returns:
        Tuple of (name_a, name_b)
    """
    updater = get_lazy_updater(batch_size=batch_size)
    return updater.select_pair(names, features)


def select_lazy_batch(
    names: list[str],
    batch_size: int,
    features: np.ndarray | None = None,
) -> list[tuple[str, str]]:
    """Select a batch of pairs using lazy updater's strategy.

    Args:
        names: List of candidate names
        batch_size: Number of pairs to select
        features: Optional precomputed feature matrix

    Returns:
        List of (name_a, name_b) tuples
    """
    updater = get_lazy_updater()
    return updater.select_batch(names, batch_size, features)


def get_lazy_updater_stats() -> dict[str, int | float] | None:
    """Get statistics for the current lazy updater.

    Returns:
        Dictionary with statistics, or None if no updater exists
    """
    if LAZY_MODEL_UPDATER_KEY not in st.session_state:
        return None

    updater: LazyModelUpdater = st.session_state[LAZY_MODEL_UPDATER_KEY]
    return updater.get_stats()


def clear_lazy_updater() -> None:
    """Clear the lazy updater from session state.

    This is useful when switching between different datasets or
    when you want to reset the model state.
    """
    if LAZY_MODEL_UPDATER_KEY in st.session_state:
        updater: LazyModelUpdater = st.session_state[LAZY_MODEL_UPDATER_KEY]
        # Wait for any pending updates
        updater.wait_for_update(timeout=2.0)
        del st.session_state[LAZY_MODEL_UPDATER_KEY]
        logger.info("LazyModelUpdater cleared from session state")
