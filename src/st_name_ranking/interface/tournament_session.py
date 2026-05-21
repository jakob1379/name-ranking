"""Streamlit session ownership for tournament queue state."""

from __future__ import annotations

import logging
import os
from typing import Final

import streamlit as st

from st_name_ranking.active_learning.queue import QueueManager

logger = logging.getLogger(__name__)

QUEUE_MANAGER_KEY: Final[str] = "st_name_ranking_queue_manager"
DEFAULT_QUEUE_SIZE: Final[int] = 15


def get_or_start_tournament_queue(names: list[str], sample_size: int) -> QueueManager:
    """Get the session queue manager using tournament UI settings."""
    target_size = int(os.environ.get("TOURNAMENT_QUEUE_SIZE", str(DEFAULT_QUEUE_SIZE)))
    return get_or_start_queue_manager(names, target_size=target_size, sample_size=sample_size)


def get_or_start_queue_manager(
    names: list[str],
    target_size: int = 15,
    refill_threshold: int = 5,
    sample_size: int = 50,
) -> QueueManager:
    """Get or create the session queue manager and ensure it is running."""
    names_key = tuple(names)
    if QUEUE_MANAGER_KEY in st.session_state:
        existing_manager: QueueManager = st.session_state[QUEUE_MANAGER_KEY]

        names_match = existing_manager.names_key == names_key
        size_match = existing_manager.target_size == target_size
        sample_size_match = existing_manager.sample_size == sample_size

        if names_match and size_match and sample_size_match:
            logger.debug("Reusing existing QueueManager from session state")
            return existing_manager

        logger.info(
            "Settings changed (names=%s, size=%s, sample_size=%s), stopping old QueueManager",
            not names_match,
            not size_match,
            not sample_size_match,
        )
        existing_manager.stop()
        del st.session_state[QUEUE_MANAGER_KEY]

    manager = QueueManager(
        names,
        target_size=target_size,
        refill_threshold=refill_threshold,
        sample_size=sample_size,
    )
    manager.start()
    st.session_state[QUEUE_MANAGER_KEY] = manager
    logger.info("Created and started new QueueManager (target_size=%d)", target_size)

    return manager


def get_queue_manager_stats() -> dict[str, int | float | bool | str] | None:
    """Return stats for the current session queue manager."""
    manager = get_session_queue_manager()
    if manager is None:
        return None

    return build_queue_manager_stats(manager)


def build_queue_manager_stats(manager: QueueManager) -> dict[str, int | float | bool | str]:
    """Return UI-ready stats for a queue manager."""
    stats: dict[str, int | float | bool | str] = manager.get_stats()
    stats["num_names"] = len(manager.names)
    return stats


def get_session_queue_manager() -> QueueManager | None:
    """Return the session queue manager when one has been created."""
    manager = st.session_state.get(QUEUE_MANAGER_KEY)
    return manager if isinstance(manager, QueueManager) else None


def get_current_pair(names: list[str]) -> tuple[str, str] | None:
    """Return the valid current tournament pair stored in session state."""
    names_set = set(names)
    candidate_a = st.session_state.get("candidate_a")
    candidate_b = st.session_state.get("candidate_b")

    if (
        isinstance(candidate_a, str)
        and isinstance(candidate_b, str)
        and candidate_a
        and candidate_b
        and candidate_a in names_set
        and candidate_b in names_set
        and candidate_a != candidate_b
    ):
        return candidate_a, candidate_b

    return None


def set_current_pair(pair: tuple[str, str]) -> None:
    """Persist the current tournament pair in session state."""
    st.session_state.candidate_a, st.session_state.candidate_b = pair
