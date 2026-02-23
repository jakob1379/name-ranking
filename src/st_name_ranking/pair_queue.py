"""FIFO queue for preloading tournament pairs.

Provides a PairQueue class that maintains a queue of preloaded (name_a, name_b) pairs
with smart refill capabilities and Streamlit session state integration.
"""

from collections import deque
import logging
from typing import Final

import numpy as np
import streamlit as st

from st_name_ranking.utils import select_candidate_batch

logger = logging.getLogger(__name__)

# Session state key for storing the pair queue
PAIR_QUEUE_KEY: Final[str] = "st_name_ranking_pair_queue"


class PairQueue:
    """FIFO queue for preloading tournament pairs.

    Maintains a queue of preloaded (name_a, name_b) pairs with configurable size.
    Tracks the current pair being displayed and only regenerates when queue runs low.

    Attributes:
        queue: Deque of (name_a, name_b) pairs waiting to be displayed
        current_pair: The pair currently being displayed, or None if not set
        queue_size: Target size for the queue when refilling
    """

    def __init__(
        self,
        names: list[str],
        features_matrix: np.ndarray | None = None,
        queue_size: int = 10,
    ) -> None:
        """Initialize the pair queue.

        Args:
            names: List of available names to generate pairs from
            features_matrix: Optional feature matrix for smart pair selection
            queue_size: Target number of pairs to maintain in queue (default 10)
        """
        self.queue: deque[tuple[str, str]] = deque()
        self.current_pair: tuple[str, str] | None = None
        self.queue_size: int = max(queue_size, 1)

        # Initial population of the queue
        self._populate(names, features_matrix)

        # Set the first pair as current if available
        if self.queue:
            self.current_pair = self.queue.popleft()
            logger.debug(f"PairQueue initialized with {len(self.queue) + 1} pairs")
        else:
            logger.warning("PairQueue initialized with no available pairs")

    def _populate(
        self,
        names: list[str],
        features_matrix: np.ndarray | None = None,
        batch_size: int | None = None,
    ) -> int:
        """Populate the queue with new pairs.

        Args:
            names: List of available names to generate pairs from
            features_matrix: Optional feature matrix for smart pair selection
            batch_size: Number of pairs to generate (defaults to queue_size)

        Returns:
            Number of pairs added to the queue
        """
        if batch_size is None:
            batch_size = self.queue_size

        if len(names) < 2:
            logger.debug("Cannot populate queue: need at least 2 names")
            return 0

        pairs = select_candidate_batch(names, features_matrix, batch_size)

        for pair in pairs:
            self.queue.append(pair)

        added = len(pairs)
        if added > 0:
            logger.debug(f"Added {added} pairs to queue (total: {len(self.queue)})")

        return added

    def get_current(self) -> tuple[str, str] | None:
        """Get the current pair being displayed.

        Returns:
            The current (name_a, name_b) pair, or None if no pair is set
        """
        return self.current_pair

    def advance(self) -> tuple[str, str] | None:
        """Move to the next pair in the queue.

        Returns the next pair immediately if preloaded. If the queue is empty,
        returns None.

        Returns:
            The next (name_a, name_b) pair, or None if queue is empty
        """
        if not self.queue:
            logger.debug("Cannot advance: queue is empty")
            self.current_pair = None
            return None

        self.current_pair = self.queue.popleft()
        logger.debug(f"Advanced to next pair, {len(self.queue)} remaining in queue")
        return self.current_pair

    def preload_next(
        self,
        names: list[str],
        features_matrix: np.ndarray | None = None,
    ) -> int:
        """Preload more pairs into the queue.

        Appends new pairs to the existing queue without replacing current contents.

        Args:
            names: List of available names to generate pairs from
            features_matrix: Optional feature matrix for smart pair selection

        Returns:
            Number of pairs added to the queue
        """
        return self._populate(names, features_matrix)

    def has_more(self) -> bool:
        """Check if there are more pairs available in the queue.

        Returns:
            True if queue has pairs waiting, False otherwise
        """
        return len(self.queue) > 0

    def is_ready(self) -> bool:
        """Check if the next pair is already preloaded and ready.

        Returns:
            True if queue has at least one pair ready, False otherwise
        """
        return self.has_more()

    def refill_if_needed(
        self,
        names: list[str],
        features_matrix: np.ndarray | None = None,
        threshold: int = 3,
    ) -> int:
        """Refill the queue if it falls below the threshold.

        When queue size drops below threshold, generates a new batch of pairs
        and appends them to the existing queue.

        Args:
            names: List of available names to generate pairs from
            features_matrix: Optional feature matrix for smart pair selection
            threshold: Minimum queue size before triggering refill (default 3)

        Returns:
            Number of pairs added (0 if no refill was needed)
        """
        if len(self.queue) >= threshold:
            logger.debug(f"Queue size {len(self.queue)} >= threshold {threshold}, no refill needed")
            return 0

        needed = self.queue_size - len(self.queue)
        logger.info(f"Queue low ({len(self.queue)} < {threshold}), refilling with {needed} pairs")

        return self._populate(names, features_matrix, needed)

    def clear(self) -> None:
        """Clear the queue and reset the current pair.

        Use this when the model changes significantly and preloaded pairs
        may no longer be relevant.
        """
        self.queue.clear()
        self.current_pair = None
        logger.info("PairQueue cleared")

    def needs_refill(self, threshold: int = 3) -> bool:
        """Check if the queue needs to be refilled.

        Args:
            threshold: Minimum queue size before refill is needed (default 3)

        Returns:
            True if queue size is below threshold, False otherwise
        """
        return len(self.queue) < threshold

    def __len__(self) -> int:
        """Return the number of pairs waiting in the queue (not including current)."""
        return len(self.queue)

    def __repr__(self) -> str:
        """Return a string representation of the queue state."""
        return (
            f"PairQueue(queue_size={self.queue_size}, "
            f"waiting={len(self.queue)}, "
            f"current={'set' if self.current_pair else 'None'})"
        )


def get_pair_queue_from_session(
    names: list[str],
    features_matrix: np.ndarray | None = None,
    queue_size: int = 10,
) -> PairQueue:
    """Get or create a PairQueue from Streamlit session state.

    If a queue already exists in session state, returns it. Otherwise,
    creates a new queue and stores it in session state.

    Args:
        names: List of available names to generate pairs from
        features_matrix: Optional feature matrix for smart pair selection
        queue_size: Target number of pairs to maintain in queue (default 10)

    Returns:
        The PairQueue instance from session state (existing or newly created)
    """
    if PAIR_QUEUE_KEY in st.session_state:
        queue = st.session_state[PAIR_QUEUE_KEY]
        logger.debug("Retrieved existing PairQueue from session state")
        return queue

    queue = PairQueue(names, features_matrix, queue_size)
    st.session_state[PAIR_QUEUE_KEY] = queue
    logger.info(f"Created new PairQueue with size {queue_size} and stored in session state")
    return queue


def save_pair_queue_to_session(queue: PairQueue) -> None:
    """Save a PairQueue to Streamlit session state.

    Args:
        queue: The PairQueue instance to save
    """
    st.session_state[PAIR_QUEUE_KEY] = queue
    logger.debug("PairQueue saved to session state")


def clear_pair_queue_session() -> None:
    """Clear the PairQueue from Streamlit session state.

    Removes the queue from session state if it exists. Safe to call
    even if no queue is stored.
    """
    if PAIR_QUEUE_KEY in st.session_state:
        del st.session_state[PAIR_QUEUE_KEY]
        logger.info("PairQueue cleared from session state")
    else:
        logger.debug("No PairQueue found in session state to clear")
