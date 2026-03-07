"""FIFO queue for preloading names with pre-computed display properties."""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import streamlit as st

logger = logging.getLogger(__name__)


def compute_name_data(name: str, index: int, inclusions: dict[str, bool]) -> dict[str, Any]:
    """Pre-compute display properties for a name.

    Args:
        name: The name string
        index: The index of the name in the list
        inclusions: Dictionary mapping names to inclusion status
                   (True=included, False=excluded, None/Not present=not decided)

    Returns:
        Dictionary containing name data with display properties
    """
    status = inclusions.get(name)

    if status is None:
        border_color = "#757575"  # Gray
        status_text = "Not decided"
        bg_color = "#FAFAFA"
    elif status is True:
        border_color = "#4CAF50"  # Green
        status_text = "Included"
        bg_color = "#E8F5E9"
    else:
        border_color = "#F44336"  # Red
        status_text = "Excluded"
        bg_color = "#FFEBEE"

    return {
        "name": name,
        "index": index,
        "status": status,
        "status_text": status_text,
        "border_color": border_color,
        "bg_color": bg_color,
    }


class NameQueue:
    """FIFO queue for preloading names with pre-computed display properties.

    This class maintains a queue of preloaded names for smooth navigation
    through a list of names, particularly useful for Streamlit applications
    where reruns can cause UI flickering.
    """

    def __init__(
        self,
        names: list[str],
        current_index: int,
        inclusions: dict[str, bool],
        queue_size: int = 5,
    ) -> None:
        """Initialize the NameQueue.

        Args:
            names: List of all names
            current_index: Current position in the names list
            inclusions: Dictionary mapping names to inclusion status
            queue_size: Number of names to preload ahead (default 5)
        """
        self._names = names
        self._current_index = current_index
        self._inclusions = inclusions
        self._queue_size = queue_size
        self._queue: deque[dict[str, Any]] = deque()
        self._total_names = len(names)

        logger.debug(
            "NameQueue initialized with %d names, current_index=%d, queue_size=%d",
            self._total_names,
            current_index,
            queue_size,
        )

        # Preload initial batch
        self.preload_next()

    @property
    def current_index(self) -> int:
        """Get the current index position."""
        return self._current_index

    @current_index.setter
    def current_index(self, value: int) -> None:
        """Set the current index position."""
        self._current_index = max(0, min(value, self._total_names - 1))

    def get_current(self) -> dict[str, Any] | None:
        """Get current name data without advancing.

        Returns:
            Name data dict or None if no names available
        """
        if not self._names or self._current_index >= self._total_names:
            return None

        current_name = self._names[self._current_index]
        return compute_name_data(current_name, self._current_index, self._inclusions)

    def advance(self) -> dict[str, Any] | None:
        """Move to next name and return it immediately if preloaded.

        If the next name is already in the queue, returns it immediately.
        Otherwise, computes it on the fly.

        Returns:
            Name data dict for the new current name, or None if at end
        """
        if not self.has_more():
            logger.debug("Cannot advance: already at end of names list")
            return None

        self._current_index += 1

        # Check if next name is already in queue
        if self._queue:
            next_data = self._queue.popleft()
            logger.debug("Advanced to index %d (preloaded)", self._current_index)
        else:
            # Compute on the fly if not preloaded
            current_name = self._names[self._current_index]
            next_data = compute_name_data(current_name, self._current_index, self._inclusions)
            logger.debug("Advanced to index %d (computed on fly)", self._current_index)

        # Trigger background preload
        self.preload_next()

        return next_data

    def preload_next(self) -> None:
        """Preload upcoming names in background (non-blocking).

        Fills the queue up to queue_size with pre-computed name data.
        """
        target_size = self._queue_size

        while len(self._queue) < target_size:
            next_index = self._current_index + len(self._queue) + 1

            if next_index >= self._total_names:
                break

            next_name = self._names[next_index]
            name_data = compute_name_data(next_name, next_index, self._inclusions)
            self._queue.append(name_data)

        logger.debug(
            "Preloaded %d names (queue size: %d)",
            len(self._queue),
            self._queue_size,
        )

    def has_more(self) -> bool:
        """Check if there are more names to load.

        Returns:
            True if there are more names after current position
        """
        return self._current_index < self._total_names - 1

    def is_ready(self) -> bool:
        """Check if next name is already preloaded.

        Returns:
            True if the next name is in the preload queue
        """
        return len(self._queue) > 0

    def peek_next(self, n: int = 1) -> list[dict[str, Any]]:
        """Preview next n names without advancing.

        Args:
            n: Number of names to preview (default 1)

        Returns:
            List of name data dicts for the next n names
        """
        result = []
        start_index = self._current_index + 1

        for i in range(n):
            peek_index = start_index + i
            if peek_index >= self._total_names:
                break

            # First check if in queue
            queue_index = i
            if queue_index < len(self._queue):
                result.append(self._queue[queue_index])
            else:
                # Compute on the fly
                name = self._names[peek_index]
                result.append(compute_name_data(name, peek_index, self._inclusions))

        return result

    def go_back(self) -> dict[str, Any] | None:
        """Move to previous name.

        Returns:
            Name data dict for the new current name, or None if at start
        """
        if self._current_index <= 0:
            logger.debug("Cannot go back: already at start of names list")
            return None

        self._current_index -= 1
        current_name = self._names[self._current_index]

        logger.debug("Went back to index %d", self._current_index)

        # Rebuild queue from new position
        self._queue.clear()
        self.preload_next()

        return compute_name_data(current_name, self._current_index, self._inclusions)

    def go_to(self, index: int) -> dict[str, Any] | None:
        """Jump to a specific index.

        Args:
            index: Target index to jump to

        Returns:
            Name data dict for the target name, or None if invalid index
        """
        if index < 0 or index >= self._total_names:
            logger.debug("Cannot go to index %d: out of bounds", index)
            return None

        self._current_index = index
        current_name = self._names[self._current_index]

        logger.debug("Jumped to index %d", self._current_index)

        # Rebuild queue from new position
        self._queue.clear()
        self.preload_next()

        return compute_name_data(current_name, self._current_index, self._inclusions)

    def refresh_current(self) -> dict[str, Any] | None:
        """Recompute current name data (useful if inclusions changed).

        Returns:
            Updated name data dict for current name
        """
        if not self._names or self._current_index >= self._total_names:
            return None

        current_name = self._names[self._current_index]
        return compute_name_data(current_name, self._current_index, self._inclusions)

    def update_inclusions(self, inclusions: dict[str, bool]) -> None:
        """Update the inclusions dictionary and refresh queue.

        Args:
            inclusions: New inclusions dictionary
        """
        self._inclusions = inclusions
        # Refresh queue with new inclusion data
        self._queue.clear()
        self.preload_next()
        logger.debug("Updated inclusions and refreshed queue")

    def to_dict(self) -> dict[str, Any]:
        """Serialize queue state to dictionary.

        Returns:
            Dictionary containing queue state for session storage
        """
        return {
            "names": self._names,
            "current_index": self._current_index,
            "inclusions": self._inclusions,
            "queue_size": self._queue_size,
            "queue": list(self._queue),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NameQueue:
        """Create NameQueue from serialized dictionary.

        Args:
            data: Dictionary containing queue state

        Returns:
            Reconstructed NameQueue instance
        """
        queue = cls(
            names=data["names"],
            current_index=data["current_index"],
            inclusions=data["inclusions"],
            queue_size=data.get("queue_size", 5),
        )
        # Restore queue contents if present
        if data.get("queue"):
            queue._queue = deque(data["queue"])
        return queue


QUEUE_SESSION_KEY = "name_queue"


def get_queue_from_session(
    names: list[str],
    current_index: int,
    inclusions: dict[str, bool],
    queue_size: int = 5,
) -> NameQueue:
    """Get or create NameQueue from session state.

    This function manages NameQueue persistence between Streamlit reruns.
    It will restore an existing queue if the names match, otherwise creates
    a new queue.

    Args:
        names: List of all names
        current_index: Current position in the names list
        inclusions: Dictionary mapping names to inclusion status
        queue_size: Number of names to preload ahead (default 5)

    Returns:
        NameQueue instance from session or newly created
    """
    # Create a hash of names for comparison
    names_hash = str(hash(tuple(names)))

    if QUEUE_SESSION_KEY in st.session_state:
        stored_queue = st.session_state[QUEUE_SESSION_KEY]
        # Check if stored queue matches current names
        stored_names = getattr(stored_queue, "_names", None)
        if stored_names == names:
            # Update index if it changed (e.g., from navigation)
            if stored_queue.current_index != current_index:
                stored_queue.current_index = current_index
                stored_queue._queue.clear()
                stored_queue.preload_next()
                logger.debug("Restored queue with updated index: %d", current_index)
            # Update inclusions if they changed
            if stored_queue._inclusions != inclusions:
                stored_queue.update_inclusions(inclusions)
            return stored_queue

    # Create new queue
    queue = NameQueue(names, current_index, inclusions, queue_size)
    st.session_state[QUEUE_SESSION_KEY] = queue
    st.session_state[f"{QUEUE_SESSION_KEY}_hash"] = names_hash
    logger.debug("Created new queue with %d names", len(names))

    return queue


def save_queue_to_session(queue: NameQueue) -> None:
    """Serialize queue state to session.

    Args:
        queue: NameQueue instance to save
    """
    st.session_state[QUEUE_SESSION_KEY] = queue
    names_hash = str(hash(tuple(queue._names)))
    st.session_state[f"{QUEUE_SESSION_KEY}_hash"] = names_hash
    logger.debug("Saved queue to session state")


def clear_queue_session() -> None:
    """Clear the queue from session state."""
    if QUEUE_SESSION_KEY in st.session_state:
        del st.session_state[QUEUE_SESSION_KEY]
    hash_key = f"{QUEUE_SESSION_KEY}_hash"
    if hash_key in st.session_state:
        del st.session_state[hash_key]
    logger.debug("Cleared queue from session state")
