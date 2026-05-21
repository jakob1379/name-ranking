"""Tournament screen rendering."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import streamlit as st

from st_name_ranking.interface.tournament_orchestration import prepare_tournament_round, record_tournament_vote
from st_name_ranking.interface.tournament_session import (
    get_current_pair,
    get_or_start_tournament_queue,
    get_queue_manager_stats,
    set_current_pair,
)
from st_name_ranking.interface.ui_support import (
    FAST_REFILL_THRESHOLD_MS,
    MIN_NAMES_FOR_COMPARISON,
    MODERATE_REFILL_THRESHOLD_MS,
    RenderTimer,
)
from st_name_ranking.persistence.database import INITIAL_SCORE

if TYPE_CHECKING:
    from collections.abc import Mapping

    from st_name_ranking.active_learning.queue import QueueManager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TournamentVoteClicks:
    prefer_a: bool
    prefer_b: bool
    draw: bool
    dislike_both: bool


def display_name_with_rating(
    name: str,
    rating: float,
    delta: float | str | None = None,
) -> None:
    """Display name much larger than rating using st.metric with custom styling.
    CSS is injected in render_tournament to make the value (name) larger
    and label (rating) smaller.
    delta: difference in preference score compared to opponent (positive if higher,
    negative if lower).
    """
    delta_str = (f"{delta:+.0f}" if isinstance(delta, (int, float)) else str(delta)) if delta is not None else None

    # Use st.metric with swapped label/value to get desired visual hierarchy
    st.metric(value=name, label=f"{rating:.0f}", delta=delta_str, border=True)


def _record_vote_and_get_next_pair(names: list[str], manager: QueueManager, preference: int) -> tuple[str, str]:
    candidate_a = st.session_state.candidate_a
    candidate_b = st.session_state.candidate_b

    if preference == -1:
        logger.info("🎮 Vote: '%s' preferred over '%s'", candidate_a, candidate_b)
    elif preference == 1:
        logger.info("🎮 Vote: '%s' preferred over '%s'", candidate_b, candidate_a)
    elif preference == 0:
        logger.info("🎮 Vote: Draw between '%s' and '%s'", candidate_a, candidate_b)
        st.toast("🤝 you chose a draw!", duration="long")
    else:
        logger.info("🎮 Vote: Both '%s' and '%s' disliked", candidate_a, candidate_b)
        st.toast("👎 you disliked both!", duration="long")

    result = record_tournament_vote(names, manager, candidate_a, candidate_b, preference)
    if not result.update_status.recorded:
        current_pair = result.previous_pair
        error = result.update_status.error or "unknown error"
        st.toast(f"Vote was not saved: {error}", icon="❌", duration="long")
        logger.warning("Tournament vote was not saved; keeping current pair: %s", error)
        return current_pair

    next_pair = result.next_pair
    set_current_pair(next_pair)
    logger.debug("Next tournament pair (%s): %s vs %s", result.pair_source, next_pair[0], next_pair[1])
    return next_pair


def _render_tournament_pair_display(
    pair_display_placeholder: Any,
    candidate_a: str,
    candidate_b: str,
    ratings: Mapping[str, float],
) -> None:
    """Render the active tournament pair inside the stable placeholder."""
    rating_a = ratings.get(candidate_a, INITIAL_SCORE)
    rating_b = ratings.get(candidate_b, INITIAL_SCORE)
    delta_a = rating_a - rating_b
    delta_b = rating_b - rating_a

    with pair_display_placeholder.container():
        _, col_left, _, col_right, _ = st.columns([0.8, 1, 0.4, 1, 0.8])

        with col_left:
            display_name_with_rating(candidate_a, rating_a, delta=delta_a)

        with col_right:
            display_name_with_rating(candidate_b, rating_b, delta=delta_b)


def _render_tournament_queue_caption(queue_stats: Mapping[str, object] | None, manager: QueueManager) -> None:
    if not queue_stats or int(queue_stats.get("refill_count", 0)) <= 0:
        st.caption("Queue warming up... | Choose the name you prefer")
        return

    last_refill_ms = float(queue_stats.get("last_refill_ms", 0.0))
    avg_refill_ms = float(queue_stats.get("avg_refill_ms", 0.0))
    refill_added = int(queue_stats.get("last_refill_added", 0))
    current_queue_size = int(queue_stats.get("queue_size", 0))
    target_queue_size = int(queue_stats.get("target_size", manager.target_size))

    if last_refill_ms <= FAST_REFILL_THRESHOLD_MS:
        latency_indicator = "🟢"
    elif last_refill_ms <= MODERATE_REFILL_THRESHOLD_MS:
        latency_indicator = "🟡"
    else:
        latency_indicator = "🔴"

    st.caption(
        f"{latency_indicator} Queue {current_queue_size}/{target_queue_size} | "
        f"Last refill {last_refill_ms:.0f} ms (avg {avg_refill_ms:.0f} ms, +{refill_added} pairs) | "
        "Choose the name you prefer",
    )


def _handle_tournament_vote(
    names: list[str],
    manager: QueueManager,
    *,
    preference: int,
    pair_display_placeholder: Any,
) -> None:
    next_pair = _record_vote_and_get_next_pair(names, manager, preference)
    _render_tournament_pair_display(pair_display_placeholder, next_pair[0], next_pair[1], st.session_state.ratings)


def _inject_tournament_styles() -> None:
    st.markdown(
        """
    <style>
    /* Style for st.metric display - ensure full name visibility */
    div[data-testid="stMetricValue"] {
        overflow: visible !important;
        white-space: normal !important;
        word-wrap: break-word !important;
        word-break: break-word !important;
        max-width: 100% !important;
    }
    div[data-testid="stMetricValue"] p {
        font-size: clamp(24px, 5vw, 48px) !important;
        font-weight: bold !important;
        text-align: center !important;
        margin-bottom: 5px !important;
        overflow: visible !important;
        white-space: normal !important;
        word-wrap: break-word !important;
        word-break: break-word !important;
        line-height: 1.2 !important;
    }
    div[data-testid="stMetricLabel"] p {
        font-size: 24px !important;
        color: #666 !important;
        text-align: center !important;
        margin-top: 0 !important;
    }
    div[data-testid="stMetric"] {
        text-align: center !important;
        overflow: visible !important;
        min-width: 200px !important;
    }
    div[data-testid="stMetricDelta"] svg {
        width: 20px !important;
        height: 20px !important;
    }

    /* Ensure columns have equal height */
    div[data-testid="column"] {
        display: flex !important;
        flex-direction: column !important;
        min-height: 300px !important;
    }
    div[data-testid="column"] > div[data-testid="stVerticalBlock"] {
        flex-grow: 1 !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


def _render_tournament_vote_buttons() -> TournamentVoteClicks:
    _, col_left, _, col_right, _ = st.columns([0.8, 1, 0.4, 1, 0.8])

    with col_left:
        st.markdown("<div style='text-align: center'>", unsafe_allow_html=True)
        prefer_a = st.button(
            f"Prefer {st.session_state.candidate_a}",
            key="vote_a",
            width="stretch",
            type="primary",
            shortcut="Left",
            help=f"Select {st.session_state.candidate_a} as preferred (Left arrow key)",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("<div style='text-align: center'>", unsafe_allow_html=True)
        prefer_b = st.button(
            f"Prefer {st.session_state.candidate_b}",
            key="vote_b",
            width="stretch",
            type="primary",
            shortcut="Right",
            help=f"Select {st.session_state.candidate_b} as preferred (Right arrow key)",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    _, col_mid, _ = st.columns([1, 0.4, 1])
    with col_mid:
        draw = st.button(
            "🤝 Draw / Equal Preference",
            key="vote_draw",
            width="stretch",
            type="secondary",
            shortcut="Up",
            help="Mark both names as equally preferred (Up arrow key)",
        )
        dislike_both = st.button(
            "👎 Down / Dislike Both",
            key="vote_both_disliked",
            width="stretch",
            type="secondary",
            shortcut="Down",
            help="Mark both names as disliked (Down arrow key)",
        )

    return TournamentVoteClicks(prefer_a=prefer_a, prefer_b=prefer_b, draw=draw, dislike_both=dislike_both)


def _selected_tournament_vote_action(clicks: TournamentVoteClicks) -> tuple[int, str] | None:
    if clicks.prefer_a:
        return -1, "Vote A"
    if clicks.prefer_b:
        return 1, "Vote B"
    if clicks.draw:
        return 0, "Draw"
    if clicks.dislike_both:
        return 2, "Both disliked"
    return None


@st.fragment
def render_tournament(names: list[str]) -> None:
    """Render tournament interface for comparing names.

    Shows two names side by side with rating scores.
    User clicks on which name they prefer.
    Uses lazy model updates for instant UI feedback.
    """
    timer = RenderTimer.start("Tournament")
    logger.debug("Tournament render started with %d names", len(names))

    if len(names) < MIN_NAMES_FOR_COMPARISON:
        if len(names) == 0:
            st.info("No names to compare. Please select at least two names.")
        else:
            st.info(f"Only one name ('{names[0]}') selected. Please select at least two names to compare.")
        return

    selected_sample_size = int(st.session_state.get("tournament_sample_size", len(names)))

    timer.log("Before queue manager")

    try:
        manager = get_or_start_tournament_queue(names, selected_sample_size)
        round_state = prepare_tournament_round(
            names,
            manager,
            get_current_pair(names),
            get_queue_manager_stats(),
        )
    except ValueError:
        st.info("No names to compare. Please select at least two names.")
        return

    manager = round_state.manager
    set_current_pair((round_state.candidate_a, round_state.candidate_b))
    timer.log("After queue manager")
    _render_tournament_queue_caption(round_state.queue_stats, manager)

    pair_display_placeholder = st.empty()
    st.empty()

    _inject_tournament_styles()
    vote_action = _selected_tournament_vote_action(_render_tournament_vote_buttons())

    timer.log("After button creation")

    if vote_action is not None:
        preference, timing_label = vote_action
        timer.log(f"{timing_label} clicked - handling")
        _handle_tournament_vote(
            names,
            manager,
            preference=preference,
            pair_display_placeholder=pair_display_placeholder,
        )

    # Render display with current candidates (will reflect any updates above)
    _render_tournament_pair_display(
        pair_display_placeholder,
        st.session_state.candidate_a,
        st.session_state.candidate_b,
        st.session_state.ratings,
    )
    timer.log("After display render")

    # NOTE: Statistics expander removed due to 20+ second render time
    # The dataframes and tabs were being processed even when collapsed

    timer.log("At end")
