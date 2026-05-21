"""Tournament orchestration for queue ownership and vote advancement."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from st_name_ranking.active_learning.lazy_updates import ModelUpdateStatus, record_comparison_instant
from st_name_ranking.active_learning.selection import select_random_pair

if TYPE_CHECKING:
    from st_name_ranking.active_learning.queue import QueueManager

logger = logging.getLogger(__name__)

MIN_NAMES_FOR_TOURNAMENT = 2

PairSource = Literal["queue", "random", "unchanged"]


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


def prepare_tournament_round(
    names: list[str],
    manager: QueueManager,
    current_pair: tuple[str, str] | None,
    queue_stats: dict[str, int | float | str] | None,
) -> TournamentRound:
    """Build a tournament round from interface-owned state."""
    if len(names) < MIN_NAMES_FOR_TOURNAMENT:
        msg = f"Need at least {MIN_NAMES_FOR_TOURNAMENT} names"
        raise ValueError(msg)

    candidate_a, candidate_b = _ensure_current_pair(names, manager, current_pair)
    return TournamentRound(
        manager=manager,
        candidate_a=candidate_a,
        candidate_b=candidate_b,
        queue_stats=queue_stats,
    )


def record_tournament_vote(
    names: list[str],
    manager: QueueManager,
    candidate_a: str,
    candidate_b: str,
    preference: int,
) -> VoteResult:
    """Record a vote and return the next tournament pair."""
    update_status = record_comparison_instant(candidate_a, candidate_b, preference)
    previous_pair = (candidate_a, candidate_b)
    if not update_status.recorded:
        logger.warning(
            "Tournament vote for (%s, %s) was not recorded; keeping current pair: %s",
            candidate_a,
            candidate_b,
            update_status.error,
        )
        return VoteResult(
            previous_pair=previous_pair,
            next_pair=previous_pair,
            pair_source="unchanged",
            update_status=update_status,
        )

    next_pair, source = _select_next_pair(names, manager)
    logger.debug(
        "Tournament transition: (%s, %s) -> (%s, %s) via %s",
        candidate_a,
        candidate_b,
        next_pair[0],
        next_pair[1],
        source,
    )

    return VoteResult(
        previous_pair=previous_pair,
        next_pair=next_pair,
        pair_source=source,
        update_status=update_status,
    )


def _ensure_current_pair(
    names: list[str],
    manager: QueueManager,
    current_pair: tuple[str, str] | None,
) -> tuple[str, str]:
    if current_pair is not None:
        return current_pair
    pair, _source = _select_next_pair(names, manager)
    return pair


def _select_next_pair(names: list[str], manager: QueueManager) -> tuple[tuple[str, str], PairSource]:
    pair = manager.try_pop_pair()
    if pair:
        return pair, "queue"
    return select_random_pair(names), "random"
