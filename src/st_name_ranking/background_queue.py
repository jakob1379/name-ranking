"""Background thread continuously filling the pair queue.

This module provides a QueueManager class that maintains a queue of name pairs
using a background thread. The main thread can instantly pop pairs while the
background thread continuously refills the queue using model-based or random
pair selection.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Final

import numpy as np
import streamlit as st

from st_name_ranking.async_model import select_random_batch
from st_name_ranking.utils import get_active_learning_model, get_names_features

if TYPE_CHECKING:
    from st_name_ranking.model import BradleyTerryModel

logger = logging.getLogger(__name__)

# Session state key for storing the queue manager
QUEUE_MANAGER_KEY: Final[str] = "st_name_ranking_queue_manager"

# Minimum names required for pair selection
MIN_NAMES_FOR_PAIR_SELECTION: Final[int] = 2

# Minimum training samples before using model-based selection
MIN_TRAINING_SAMPLES: Final[int] = 10


class QueueManager:
    """Thread-safe queue manager with background filler thread.

    Maintains a queue of preloaded (name_a, name_b) pairs. A background thread
    continuously monitors the queue size and refills it when it falls below
    the threshold. The main thread can pop pairs instantly without blocking.

    Attributes:
        names: List of available names to generate pairs from
        queue: Deque of (name_a, name_b) pairs waiting to be displayed
        target_size: Target number of pairs to maintain in queue
        refill_threshold: Queue size below which to trigger refill
        _lock: Thread lock for safe queue access
        _stop_event: Event to signal background thread to stop
        _worker_thread: Background thread that fills the queue
    """

    def __init__(
        self,
        names: list[str],
        target_size: int = 15,
        refill_threshold: int = 5,
    ) -> None:
        """Initialize the queue manager.

        Args:
            names: List of available names to generate pairs from
            target_size: Target number of pairs to maintain in queue (default 15)
            refill_threshold: Queue size below which to trigger refill (default 5)
        """
        if not names:
            raise ValueError("names list cannot be empty")
        if len(names) < MIN_NAMES_FOR_PAIR_SELECTION:
            raise ValueError(f"Need at least {MIN_NAMES_FOR_PAIR_SELECTION} names")

        self.names: list[str] = names
        self.queue: deque[tuple[str, str]] = deque()
        self.target_size: int = max(target_size, 1)
        self.refill_threshold: int = max(refill_threshold, 1)
        self._lock: threading.Lock = threading.Lock()
        self._stop_event: threading.Event = threading.Event()
        self._worker_thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background filler thread."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            logger.debug("Background thread already running")
            return

        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._fill_queue_continuously,
            daemon=True,
        )
        self._worker_thread.start()
        logger.info("Started QueueManager background thread (target_size=%d)", self.target_size)

    def stop(self) -> None:
        """Stop the background thread and wait for it to finish."""
        self._stop_event.set()

        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
            if self._worker_thread.is_alive():
                logger.warning("Background thread did not stop within timeout")
            else:
                logger.info("Stopped QueueManager background thread")
        else:
            logger.debug("No background thread to stop")

    def get_pair(self) -> tuple[str, str] | None:
        """Thread-safe pop from queue (instant, never blocks).

        Returns:
            The next (name_a, name_b) pair, or None if queue is empty
        """
        with self._lock:
            if self.queue:
                pair = self.queue.popleft()
                logger.debug("Popped pair from queue, %d remaining", len(self.queue))
                return pair
            return None

    def get_queue_size(self) -> int:
        """Get the current queue size (thread-safe).

        Returns:
            Number of pairs currently in the queue
        """
        with self._lock:
            return len(self.queue)

    def _fill_queue_continuously(self) -> None:
        """Background thread that continuously keeps the queue full."""
        logger.debug("Background filler thread started")

        while not self._stop_event.is_set():
            try:
                # Check if we need to refill
                with self._lock:
                    current_size = len(self.queue)

                if current_size < self.refill_threshold:
                    self._refill_queue()

                # Small sleep to prevent CPU spinning
                time.sleep(0.1)

            except Exception:
                logger.exception("Error in background queue filler")
                time.sleep(0.5)  # Longer sleep on error

        logger.debug("Background filler thread stopped")

    def _refill_queue(self) -> None:
        """Refill the queue with new pairs.

        Gets the latest model, calculates variances, and fills the queue
        with either model-based or random pairs.
        """
        import time

        start_time = time.perf_counter()
        logger.info("🔄 Queue refill started (current_size=%d, target=%d)", self.get_queue_size(), self.target_size)

        # Get latest model (might be updating or None)
        try:
            model: BradleyTerryModel | None = get_active_learning_model()
            logger.info("✅ Model loaded (training_samples=%d)", model.state.training_samples if model else 0)
        except Exception as e:
            logger.warning("⚠️ Failed to get active learning model: %s", e)
            model = None

        # Get features for all names
        try:
            logger.info("📊 Extracting features for %d names...", len(self.names))
            features = get_names_features(self.names)
            logger.info("✅ Features extracted: %s", features.shape if features is not None else "None")
        except Exception as e:
            logger.warning("⚠️ Failed to get name features: %s", e)
            features = None

        # Calculate how many pairs we need
        with self._lock:
            current_size = len(self.queue)
        needed = self.target_size - current_size

        if needed <= 0:
            return

        # Generate pairs based on model state
        pairs: list[tuple[str, str]] = []

        if model is not None and features is not None and model.state.training_samples >= MIN_TRAINING_SAMPLES:
            # Model-based selection with variance calculation
            try:
                logger.info("🎯 Using model-based selection (training_samples=%d)", model.state.training_samples)
                # Sample a subset for efficiency (max 50 names)
                rng = np.random.default_rng()
                sample_size = min(50, len(self.names))
                if len(self.names) == sample_size:
                    sampled_names = self.names
                    sampled_indices = list(range(len(self.names)))
                else:
                    sampled_indices = list(rng.choice(len(self.names), size=sample_size, replace=False))
                    sampled_names = [self.names[i] for i in sampled_indices]

                sampled_features = features[sampled_indices]

                # Use model's top-k selection
                name_pairs = model.select_top_k_pairs(
                    sampled_features,
                    sampled_names,
                    k=needed,
                )
                pairs = [(p.name_a, p.name_b) for p in name_pairs]
                logger.info("🎯 Generated %d model-based pairs", len(pairs))
            except Exception as e:
                logger.warning("⚠️ Model-based pair selection failed: %s", e)
                pairs = []

        # Fallback to random selection if model selection failed or not enough training samples
        if not pairs:
            try:
                logger.info("🎲 Using random selection (needed=%d)", needed)
                pairs = select_random_batch(self.names, needed)
                logger.info("🎲 Generated %d random pairs", len(pairs))
            except Exception as e:
                logger.error("❌ Random pair selection failed: %s", e)
                return

        # Add pairs to queue, avoiding duplicates
        with self._lock:
            # Get existing pairs as a set for O(1) lookup
            existing_pairs = set(self.queue)

            added = 0
            for pair in pairs:
                # Normalize pair order for duplicate checking
                normalized = (min(pair[0], pair[1]), max(pair[0], pair[1]))
                if normalized not in existing_pairs:
                    self.queue.append(pair)
                    existing_pairs.add(normalized)
                    added += 1

            elapsed = time.perf_counter() - start_time
            if added > 0:
                logger.info("✅ Added %d pairs to queue (total: %d) in %.2fs", added, len(self.queue), elapsed)
            else:
                logger.info("⚠️ No new pairs added (all duplicates) in %.2fs", elapsed)

    def __del__(self) -> None:
        """Ensure background thread is stopped on garbage collection."""
        try:
            self.stop()
        except Exception:
            # Ignore errors during garbage collection
            pass

    def __repr__(self) -> str:
        """Return a string representation of the queue manager state."""
        with self._lock:
            queue_size = len(self.queue)
        thread_status = "running" if self._worker_thread and self._worker_thread.is_alive() else "stopped"
        return f"QueueManager(target_size={self.target_size}, current_size={queue_size}, thread={thread_status})"


def get_queue_manager(
    names: list[str],
    queue_size: int = 15,
    refill_threshold: int = 5,
) -> QueueManager:
    """Get or create a QueueManager from Streamlit session state.

    If a queue manager already exists for the same names, returns it.
    Otherwise, stops the old manager if present and creates a new one.

    Args:
        names: List of available names to generate pairs from
        queue_size: Target number of pairs to maintain in queue (default 15)
        refill_threshold: Queue size below which to trigger refill (default 5)

    Returns:
        QueueManager instance from session state (existing or newly created)
    """
    # Check if we have an existing manager with the same names and settings
    if QUEUE_MANAGER_KEY in st.session_state:
        existing_manager: QueueManager = st.session_state[QUEUE_MANAGER_KEY]

        # Check if names and queue_size are the same
        # Quick checks first: length and first/last elements
        names_match = (
            len(existing_manager.names) == len(names)
            and len(existing_manager.names) > 0
            and existing_manager.names[0] == names[0]
            and existing_manager.names[-1] == names[-1]
        )
        size_match = existing_manager.target_size == queue_size

        if names_match and size_match:
            logger.debug("Reusing existing QueueManager from session state")
            return existing_manager

        # Settings changed - stop the old manager
        logger.info("Settings changed (names=%s, size=%s), stopping old QueueManager", not names_match, not size_match)
        existing_manager.stop()
        del st.session_state[QUEUE_MANAGER_KEY]

    # Create new manager and start it
    manager = QueueManager(names, target_size=queue_size, refill_threshold=refill_threshold)
    manager.start()
    st.session_state[QUEUE_MANAGER_KEY] = manager
    logger.info("Created and started new QueueManager (queue_size=%d)", queue_size)

    return manager


def stop_queue_manager() -> None:
    """Stop and clear the QueueManager from session state.

    Safe to call even if no queue manager exists. This is useful when
    switching between different datasets or when shutting down.
    """
    if QUEUE_MANAGER_KEY in st.session_state:
        manager: QueueManager = st.session_state[QUEUE_MANAGER_KEY]
        manager.stop()
        del st.session_state[QUEUE_MANAGER_KEY]
        logger.info("QueueManager stopped and cleared from session state")
    else:
        logger.debug("No QueueManager found in session state to stop")


def get_queue_manager_stats() -> dict[str, int | float | str] | None:
    """Get statistics for the current queue manager.

    Returns:
        Dictionary with statistics, or None if no manager exists
    """
    if QUEUE_MANAGER_KEY not in st.session_state:
        return None

    manager: QueueManager = st.session_state[QUEUE_MANAGER_KEY]
    return {
        "queue_size": manager.get_queue_size(),
        "target_size": manager.target_size,
        "refill_threshold": manager.refill_threshold,
        "num_names": len(manager.names),
        "thread_alive": manager._worker_thread is not None and manager._worker_thread.is_alive(),
    }
