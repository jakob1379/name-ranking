"""Tournament orchestration for queue ownership and vote advancement."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Literal

import streamlit as st

from st_name_ranking.active_learning.lazy_updates import ModelUpdateStatus, record_comparison_instant
from st_name_ranking.active_learning.queue import QueueManager, get_or_start_queue_manager, get_queue_manager_stats
from st_name_ranking.active_learning.selection import select_random_pair

logger = logging.getLogger(__name__)

MIN_NAMES_FOR_TOURNAMENT = 2
DEFAULT_QUEUE_SIZE = 15

PairSource = Literal["queue", "random"]


@dataclass(frozen=True)
class TournamentRound:
    """Current tournament pair plus queue metadata needed by the UI."""

    manager: QueueManager
    candidate_a: str
    candidate_b: str
    queue_stats: dict[str, int | float | str] | None


@dataclass(frozen=True)
class VoteResult:
    """Result of recording a tournament vote and selecting the next pair."""

    previous_pair: tuple[str, str]
    next_pair: tuple[str, str]
    pair_source: PairSource
    update_status: ModelUpdateStatus


def prepare_tournament_round(names: list[str], sample_size: int) -> TournamentRound:
    """Ensure the tournament queue and current pair are ready for rendering."""
    if len(names) < MIN_NAMES_FOR_TOURNAMENT:
        msg = f"Need at least {MIN_NAMES_FOR_TOURNAMENT} names"
        raise ValueError(msg)

    manager = _get_manager(names, sample_size)
    candidate_a, candidate_b = _ensure_current_pair(names, manager)
    return TournamentRound(
        manager=manager,
        candidate_a=candidate_a,
        candidate_b=candidate_b,
        queue_stats=get_queue_manager_stats(),
    )


def record_tournament_vote(
    names: list[str],
    manager: QueueManager,
    candidate_a: str,
    candidate_b: str,
    preference: int,
) -> VoteResult:
    """Record a vote and advance session state to the next tournament pair."""
    update_status = record_comparison_instant(candidate_a, candidate_b, preference)

    next_pair, source = _select_next_pair(names, manager)
    st.session_state.candidate_a, st.session_state.candidate_b = next_pair
    logger.debug(
        "Tournament transition: (%s, %s) -> (%s, %s) via %s",
        candidate_a,
        candidate_b,
        next_pair[0],
        next_pair[1],
        source,
    )

    return VoteResult(
        previous_pair=(candidate_a, candidate_b),
        next_pair=next_pair,
        pair_source=source,
        update_status=update_status,
    )


def _get_manager(names: list[str], sample_size: int) -> QueueManager:
    target_size = int(os.environ.get("TOURNAMENT_QUEUE_SIZE", str(DEFAULT_QUEUE_SIZE)))
    return get_or_start_queue_manager(names, target_size=target_size, sample_size=sample_size)


def _ensure_current_pair(names: list[str], manager: QueueManager) -> tuple[str, str]:
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

    pair, _source = _select_next_pair(names, manager)
    st.session_state.candidate_a, st.session_state.candidate_b = pair
    return pair


def _select_next_pair(names: list[str], manager: QueueManager) -> tuple[tuple[str, str], PairSource]:
    pair = manager.get_pair()
    if pair:
        return pair, "queue"
    return select_random_pair(names), "random"
