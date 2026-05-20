"""FIFO queue for preloading names with pre-computed display properties."""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

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
