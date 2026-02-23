"""UI rendering functions for the name ranking application."""

import json
import logging
import time
from typing import Literal

import pandas as pd
import streamlit as st

from st_name_ranking.database import (
    INITIAL_SCORE,
    get_preference_stats_by_gender,
    get_preference_stats_by_origin,
    get_preference_stats_by_phonetic,
    load_user_setting,
    save_user_setting,
)
from st_name_ranking.similarity import (
    get_string_similarity_scores,
    get_vector_similarity_scores,
    load_embedding_model,
)
from st_name_ranking.types import PreferenceStats
from st_name_ranking.name_queue import (
    NameQueue,
    clear_queue_session,
    get_queue_from_session,
    save_queue_to_session,
)
from st_name_ranking.async_model import select_random_pair
from st_name_ranking.background_queue import get_queue_manager, stop_queue_manager
from st_name_ranking.utils import (
    get_names_features,
    record_comparison_instant,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
    # Format delta for display
    if delta is not None:
        # Convert numeric delta to string with sign
        if isinstance(delta, (int, float)):
            delta_str = f"{delta:+.0f}"
        else:
            delta_str = str(delta)
    else:
        delta_str = None

    # Use st.metric with swapped label/value to get desired visual hierarchy
    # Value will be large (name), label will be smaller (rating)
    st.metric(value=name, label=f"{rating:.0f}", delta=delta_str, border=True)


def render_preferences_panel() -> None:
    """Render panel showing overall preferences across different groups."""
    st.subheader("Overall Preferences")

    # Get statistics
    gender_stats = get_preference_stats_by_gender()
    origin_stats = get_preference_stats_by_origin()
    phonetic_stats = get_preference_stats_by_phonetic()

    def create_percentage_dataframe(stats: dict[str, PreferenceStats]) -> pd.DataFrame:
        """Convert stats dict to DataFrame with percentage columns for visualization."""
        rows = []
        for group, data in stats.items():
            wins = data.wins
            losses = data.losses
            draws = data.draws
            total = data.total

            # Calculate percentages
            win_pct = wins / total * 100 if total > 0 else 0.0
            loss_pct = losses / total * 100 if total > 0 else 0.0
            draw_pct = draws / total * 100 if total > 0 else 0.0
            win_rate = wins / (wins + losses) * 100 if wins + losses > 0 else 0.0

            rows.append(
                {
                    "Group": group,
                    "Wins": wins,
                    "Losses": losses,
                    "Draws": draws,
                    "Total": total,
                    "win_pct": win_pct,
                    "loss_pct": loss_pct,
                    "draw_pct": draw_pct,
                    "win_rate_pct": win_rate,
                },
            )
        return pd.DataFrame(rows)

    def create_stacked_bar_chart(df: pd.DataFrame, title: str) -> None:
        """Create stacked bar chart showing win/loss/draw percentages."""
        if df.empty:
            return

        # Create percentage columns for stacked bar chart
        # Select relevant columns before setting index to avoid type issues
        chart_df = df[["Group", "win_pct", "loss_pct", "draw_pct"]].set_index("Group")

        # Sort by win percentage (descending)
        chart_df = chart_df.sort_values("win_pct", ascending=False)

        # Create stacked bar chart
        st.subheader(title, divider="gray")

        # Display chart with custom colors
        st.bar_chart(
            chart_df,
            height=400,
            width="stretch",
            color=["#2E7D32", "#C62828", "#FF9800"],  # Green for wins, red for losses, orange for draws
        )

        # Add data table below chart
        display_df = df.copy()
        display_df = display_df.sort_values("win_pct", ascending=False)

        # Format percentage columns for display (add % symbol)
        for col in ["win_pct", "loss_pct", "draw_pct", "win_rate_pct"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}%")

        # Show detailed table
        with st.expander(f"Detailed {title} Statistics", expanded=False):
            st.dataframe(
                display_df,
                hide_index=True,
                width="stretch",
                column_config={
                    "Group": st.column_config.TextColumn("Group", width="medium"),
                    "Wins": st.column_config.NumberColumn("Wins", width="small"),
                    "Losses": st.column_config.NumberColumn("Losses", width="small"),
                    "Draws": st.column_config.NumberColumn("Draws", width="small"),
                    "Total": st.column_config.NumberColumn("Total", width="small"),
                    "win_pct": st.column_config.TextColumn("Win %", width="small"),
                    "loss_pct": st.column_config.TextColumn("Loss %", width="small"),
                    "draw_pct": st.column_config.TextColumn("Draw %", width="small"),
                    "win_rate_pct": st.column_config.TextColumn(
                        "Win Rate %",
                        help="Wins / (Wins + Losses)",
                        width="small",
                    ),
                },
            )

        # Add legend
        st.caption("🎯 **Legend**: 🟢 Wins | 🔴 Losses | 🟠 Draws")

        # Add key insight based on data
        if not df.empty:
            best_group = df.loc[df["win_pct"].idxmax()]
            worst_group = df.loc[df["win_pct"].idxmin()]
            st.info(
                f"**Insight**: {best_group['Group']} has the highest win rate ({best_group['win_pct']:.1f}%), "
                f"while {worst_group['Group']} has the lowest ({worst_group['win_pct']:.1f}%).",
            )

    # Gender preferences
    if gender_stats:
        df_gender = create_percentage_dataframe(gender_stats)
        create_stacked_bar_chart(df_gender, "Gender Preferences")
    else:
        st.info("No gender preference data available.")

    # Origin preferences
    if origin_stats:
        df_origin = create_percentage_dataframe(origin_stats)
        create_stacked_bar_chart(df_origin, "Origin Preferences")
    else:
        st.info("No origin preference data available.")

    # Phonetic preferences
    if phonetic_stats:
        df_phonetic = create_percentage_dataframe(phonetic_stats)
        create_stacked_bar_chart(df_phonetic, "Phonetic Preferences")
    else:
        st.info("No phonetic preference data available.")


@st.fragment
def render_tournament(names: list[str]) -> None:
    """Render tournament interface for comparing names.

    Shows two names side by side with rating scores.
    User clicks on which name they prefer.
    Uses lazy model updates for instant UI feedback.
    """

    logger.debug("Rendering tournament with %d names", len(names))
    st.header("Name Ranking Tournament")
    st.write(f"Comparing {len(names)} names")
    st.caption("Tournament mode: click which name you prefer")

    # Handle empty names list gracefully
    if len(names) < 2:
        if len(names) == 0:
            st.info("No names to compare. Please select at least two names.")
        else:
            st.info(f"Only one name ('{names[0]}') selected. Please select at least two names to compare.")
        return

    # Precompute features and names set
    features_matrix = get_names_features(names)
    names_set = set(names)

    # Create placeholders for dynamic content
    pair_display_placeholder = st.empty()
    buttons_placeholder = st.empty()

    # Get or create background queue manager for instant transitions
    queue_size = st.session_state.get("tournament_queue_size", 15)
    manager = get_queue_manager(names, queue_size)

    # Initialize candidates if not set or empty
    if (
        "candidate_a" not in st.session_state
        or "candidate_b" not in st.session_state
        or not st.session_state.candidate_a
        or not st.session_state.candidate_b
    ):
        current_pair = manager.get_pair()
        if not current_pair:
            # Fallback if queue empty
            current_pair = (names[0], names[1]) if len(names) > 1 else (names[0], names[0])
        st.session_state.candidate_a, st.session_state.candidate_b = current_pair

    st.markdown(
        """
    <style>
    /* Style for st.metric display */
    div[data-testid="stMetricValue"] p {
        font-size: 48px !important;
        font-weight: bold !important;
        text-align: center !important;
        margin-bottom: 5px !important;
    }
    div[data-testid="stMetricLabel"] p {
        font-size: 24px !important;
        color: #666 !important;
        text-align: center !important;
        margin-top: 0 !important;
    }
    div[data-testid="stMetric"] {
        text-align: center !important;
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

    def update_display(candidate_a: str, candidate_b: str, ratings: dict) -> None:
        """Update the display with new pair and ratings.

        Called instantly after vote without requiring a full rerun.
        """
        # Get both ratings
        rating_a = ratings.get(candidate_a, INITIAL_SCORE)
        rating_b = ratings.get(candidate_b, INITIAL_SCORE)

        # Calculate rating differences for delta display
        delta_a = rating_a - rating_b  # Positive if A is higher rated
        delta_b = rating_b - rating_a  # Positive if B is higher rated

        # Update pair display
        with pair_display_placeholder.container():
            _, col_left, _, col_right, _ = st.columns([0.8, 1, 0.4, 1, 0.8])

            with col_left:
                display_name_with_rating(candidate_a, rating_a, delta=delta_a)

            with col_right:
                display_name_with_rating(candidate_b, rating_b, delta=delta_b)

    # Button handlers with background QueueManager for instant transitions
    _, col_left, _, col_right, _ = st.columns([0.8, 1, 0.4, 1, 0.8])

    with col_left:
        st.markdown("<div style='text-align: center'>", unsafe_allow_html=True)
        vote_a_clicked = st.button(
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
        vote_b_clicked = st.button(
            f"Prefer {st.session_state.candidate_b}",
            key="vote_b",
            width="stretch",
            type="primary",
            shortcut="Right",
            help=f"Select {st.session_state.candidate_b} as preferred (Right arrow key)",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Draw and Down buttons centered below both names
    _, col_mid, _ = st.columns([1, 0.4, 1])
    with col_mid:
        draw_clicked = st.button(
            "🤝 Draw / Equal Preference",
            key="vote_draw",
            width="stretch",
            type="secondary",
            shortcut="Up",
            help="Mark both names as equally preferred (Up arrow key)",
        )
        down_clicked = st.button(
            "👎 Down / Dislike Both",
            key="vote_down",
            width="stretch",
            type="secondary",
            shortcut="Down",
            help="Mark both names as disliked (Down arrow key)",
        )

    # Handle votes with instant async model updates for UI feedback
    vote_handled = False
    next_pair = None

    if vote_a_clicked:
        vote_handled = True
        # 1. Record comparison instantly (async model update in background)
        record_comparison_instant(st.session_state.candidate_a, st.session_state.candidate_b, -1)
        # 2. Get next pair instantly from background queue
        next_pair = manager.get_pair()
        if not next_pair:
            next_pair = select_random_pair(names)
    elif vote_b_clicked:
        vote_handled = True
        # 1. Record comparison instantly (async model update in background)
        record_comparison_instant(st.session_state.candidate_a, st.session_state.candidate_b, 1)
        # 2. Get next pair instantly from background queue
        next_pair = manager.get_pair()
        if not next_pair:
            next_pair = select_random_pair(names)
    elif draw_clicked:
        vote_handled = True
        st.toast("🤝 you chose a draw!", duration="long")
        # 1. Record comparison instantly (async model update in background)
        record_comparison_instant(st.session_state.candidate_a, st.session_state.candidate_b, 0)
        # 2. Get next pair instantly from background queue
        next_pair = manager.get_pair()
        if not next_pair:
            next_pair = select_random_pair(names)
    elif down_clicked:
        vote_handled = True
        st.toast("👎 you disliked both!", duration="long")
        # 1. Record comparison instantly (async model update in background)
        record_comparison_instant(st.session_state.candidate_a, st.session_state.candidate_b, 2)
        # 2. Get next pair instantly from background queue
        next_pair = manager.get_pair()
        if not next_pair:
            next_pair = select_random_pair(names)

    # Update session state and UI instantly after vote (no rerun needed)
    if vote_handled and next_pair:
        st.session_state.candidate_a, st.session_state.candidate_b = next_pair
        # Update UI instantly with st.fragment placeholder updates
        update_display(next_pair[0], next_pair[1], st.session_state.ratings)

    # Render display with current candidates (will reflect any updates above)
    update_display(st.session_state.candidate_a, st.session_state.candidate_b, st.session_state.ratings)

    with st.expander(label="statistics"):
        st.divider()
        st.subheader("Top Rankings")

        col_df, col_prefs = st.columns(2)

        with col_df:
            # Get gender-specific name lists if available
            male_names = []
            female_names = []
            if "all_names_data" in st.session_state:
                gender_data = st.session_state.all_names_data
                male_names = gender_data.get("Male", [])
                female_names = gender_data.get("Female", [])

            # Overall ratings (filtered by current selection)
            filtered_ratings = {name: rating for name, rating in st.session_state.ratings.items() if name in names_set}

            tab_overall, tab_male, tab_female = st.tabs(["Overall", "Male", "Female"])

            with tab_overall:
                sorted_ratings = sorted(
                    filtered_ratings.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
                df = pd.DataFrame(sorted_ratings[:10], columns=["Name", "Rating"])  # type: ignore[call-overload]
                st.dataframe(
                    df,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Rating": st.column_config.NumberColumn(
                            "Rating",
                            help="Higher is better",
                            format="%d",
                            pinned=True,
                            width="small",
                        ),
                    },
                )

            with tab_male:
                if male_names:
                    male_ratings = {
                        name: rating
                        for name, rating in st.session_state.ratings.items()
                        if name in male_names and name in names_set
                    }
                    sorted_male = sorted(
                        male_ratings.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                    df_male = pd.DataFrame(sorted_male[:10], columns=["Name", "Rating"])  # type: ignore[call-overload]
                    st.dataframe(
                        df_male,
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "Rating": st.column_config.NumberColumn(
                                "Rating",
                                help="Higher is better",
                                format="%d",
                                pinned=True,
                                width="small",
                            ),
                        },
                    )
                else:
                    st.info("No male names in current filter")

            with tab_female:
                if female_names:
                    female_ratings = {
                        name: rating
                        for name, rating in st.session_state.ratings.items()
                        if name in female_names and name in names_set
                    }
                    sorted_female = sorted(
                        female_ratings.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                    df_female = pd.DataFrame(sorted_female[:10], columns=["Name", "Rating"])  # type: ignore[call-overload]
                    st.dataframe(
                        df_female,
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "Rating": st.column_config.NumberColumn(
                                "Rating",
                                help="Higher is better",
                                format="%d",
                                pinned=True,
                                width="small",
                            ),
                        },
                    )
                else:
                    st.info("No female names in current filter")

        # with col_prefs:
        #     render_preferences_panel()


@st.fragment
def render_binary_filter(names: list[str]) -> None:
    """Render binary filter interface for including/excluding names.

    Users review names one by one, marking them as included or excluded.
    """
    logger.debug("Rendering binary filter with %d names", len(names))
    start_time = time.perf_counter()

    # Clear last button press time (used for performance monitoring)
    if "last_button_press_time" in st.session_state:
        del st.session_state.last_button_press_time

    # Helper function to update counts incrementally
    def update_counts(name: str, *, old_status: bool | None, new_status: bool) -> None:
        """Update filter counts when a name's inclusion status changes."""
        # old_status: None (not decided), True (included), False (excluded)
        # new_status: True (included) or False (excluded)

        # Decrement old category
        if old_status is None:
            st.session_state.filter_counts_not_decided -= 1
        elif old_status is True:
            st.session_state.filter_counts_included -= 1
        else:  # old_status is False
            st.session_state.filter_counts_excluded -= 1

        # Increment new category
        if new_status is True:
            st.session_state.filter_counts_included += 1
        else:  # new_status is False
            st.session_state.filter_counts_excluded += 1

    st.header("Name Filter")
    st.write(
        "Review names one by one. Include names you want to compare in the tournament, "
        "exclude names you don't care about.",
    )
    st.caption(
        "💡 **Keyboard shortcuts**: Left arrow (←) to exclude, Right arrow (→) to include, Space to include",
    )

    # Create placeholders for instant updates
    progress_placeholder = st.empty()
    stats_placeholder = st.empty()
    name_display_placeholder = st.empty()

    # Load inclusion decisions from session state or database
    if "name_inclusions" not in st.session_state:
        try:
            inclusions_json = load_user_setting("name_inclusions", "{}")
            st.session_state.name_inclusions = json.loads(inclusions_json)
        except json.JSONDecodeError:
            st.session_state.name_inclusions = {}

    inclusions = st.session_state.name_inclusions
    time_after_inclusions = time.perf_counter()

    # Detect if names list has changed (e.g., gender/origin filter changed)
    # Store a hash of the current names list (names are already sorted from database)
    names_hash = str(hash(tuple(names)))
    if "filter_names_hash" not in st.session_state:
        st.session_state.filter_names_hash = names_hash
        st.session_state.filter_index = 0
    elif st.session_state.filter_names_hash != names_hash:
        # Names list changed, reset index
        st.session_state.filter_names_hash = names_hash
        st.session_state.filter_index = 0

    # Initialize current index if not exists (should be handled above)
    if "filter_index" not in st.session_state:
        st.session_state.filter_index = 0

    # Ensure index is within bounds
    if st.session_state.filter_index >= len(names):
        st.session_state.filter_index = 0

    # Initialize or update filter counts if names list changed
    if (
        "filter_counts_not_decided" not in st.session_state
        or "filter_counts_included" not in st.session_state
        or "filter_counts_excluded" not in st.session_state
        or st.session_state.get("filter_counts_names_hash") != names_hash
    ):
        # Recompute counts from scratch
        not_decided = explicitly_included = explicitly_excluded = 0
        for name in names:
            status = inclusions.get(name)
            if status is None:
                not_decided += 1
            elif status is True:
                explicitly_included += 1
            else:  # status is False
                explicitly_excluded += 1

        st.session_state.filter_counts_not_decided = not_decided
        st.session_state.filter_counts_included = explicitly_included
        st.session_state.filter_counts_excluded = explicitly_excluded
        st.session_state.filter_counts_names_hash = names_hash

    time_after_counts = time.perf_counter()

    # Get current name
    current_idx = st.session_state.filter_index
    current_name = names[current_idx]

    def update_display(current_name: str, current_idx: int) -> None:
        """Update display instantly without rerun."""
        # Progress bar
        progress = current_idx / len(names)
        progress_placeholder.progress(progress, text=f"Progress: {current_idx + 1} of {len(names)}")

        # Stats
        not_decided = st.session_state.filter_counts_not_decided
        explicitly_included = st.session_state.filter_counts_included
        explicitly_excluded = st.session_state.filter_counts_excluded
        stats_placeholder.caption(
            f"Not decided: {not_decided} | Included: {explicitly_included} | Excluded: {explicitly_excluded}",
        )

        # Name display with colors
        if current_name not in inclusions:
            border_color = "#757575"
            status_text = "Not decided"
            bg_color = "#FAFAFA"
        elif inclusions[current_name]:
            border_color = "#4CAF50"
            status_text = "Included"
            bg_color = "#E8F5E9"
        else:
            border_color = "#F44336"
            status_text = "Excluded"
            bg_color = "#FFEBEE"

        name_display_placeholder.markdown(
            f"<div style='border: 4px solid {border_color}; background-color: {bg_color}; "
            f"border-radius: 12px; padding: 20px; text-align: center;'>"
            f"<h1 style='font-size: 72px; margin: 0; color: #212121;'>{current_name}</h1>"
            f"<p style='font-size: 16px; margin: 10px 0 0 0; color: {border_color}; "
            f"font-weight: bold;'>{status_text}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Initial display
    update_display(current_name, current_idx)
    time_after_name_display = time.perf_counter()

    # Decision buttons - simplified to two clear options
    col_exclude, col_include = st.columns(2)
    with col_exclude:
        if st.button(
            "Exclude",
            key="exclude_btn",
            use_container_width=True,
            type="secondary",
            shortcut="Left",
        ):
            old_status = inclusions.get(current_name)
            inclusions[current_name] = False
            update_counts(current_name, old_status=old_status, new_status=False)
            st.session_state.filter_index += 1
            st.toast(f"Excluded: {current_name}", icon="👎")
            st.session_state.last_button_press_time = time.perf_counter()
            # INSTANT UPDATE - no rerun!
            next_idx = st.session_state.filter_index
            if next_idx < len(names):
                update_display(names[next_idx], next_idx)
    with col_include:
        if st.button(
            "Include",
            key="include_btn",
            use_container_width=True,
            type="primary",
            shortcut="Right",
        ):
            # Include (explicitly mark as included)
            old_status = inclusions.get(current_name)
            inclusions[current_name] = True
            update_counts(current_name, old_status=old_status, new_status=True)
            st.session_state.filter_index += 1
            st.toast(f"Included: {current_name}", icon="👍")
            st.session_state.last_button_press_time = time.perf_counter()
            # INSTANT UPDATE - no rerun!
            next_idx = st.session_state.filter_index
            if next_idx < len(names):
                update_display(names[next_idx], next_idx)

    # Save decisions periodically (every 50 actions to reduce DB writes)
    if current_idx % 50 == 0:
        # Time JSON serialization and database save
        json_start = time.perf_counter()
        inclusions_json = json.dumps(inclusions)
        json_time = (time.perf_counter() - json_start) * 1000

        save_start = time.perf_counter()
        save_user_setting("name_inclusions", inclusions_json)
        save_time = (time.perf_counter() - save_start) * 1000

        logger.debug("Periodic save: JSON=%.1fms, DB=%.1fms, entries=%d", json_time, save_time, len(inclusions))

    # Batch operations
    col_batch1, col_batch2 = st.columns(2)
    with col_batch1:
        if st.button("Include All Remaining", type="secondary", help="Include all remaining names"):
            count = len(names) - current_idx
            for name in names[current_idx:]:
                inclusions[name] = True
            st.session_state.filter_index = len(names)  # Move to end
            # Invalidate counts cache to force recomputation
            if "filter_counts_names_hash" in st.session_state:
                del st.session_state.filter_counts_names_hash
            st.toast(f"Included {count} remaining names", icon="✅")
            st.session_state.last_button_press_time = time.perf_counter()
            # Fragment-level refresh for batch operations
            st.rerun(scope="fragment")
    with col_batch2:
        if st.button("Exclude All Remaining", type="secondary", help="Exclude all remaining names"):
            count = len(names) - current_idx
            for name in names[current_idx:]:
                inclusions[name] = False
            st.session_state.filter_index = len(names)
            # Invalidate counts cache to force recomputation
            if "filter_counts_names_hash" in st.session_state:
                del st.session_state.filter_counts_names_hash
            st.toast(f"Excluded {count} remaining names", icon="✅")
            st.session_state.last_button_press_time = time.perf_counter()
            # Fragment-level refresh for batch operations
            st.rerun(scope="fragment")

    # Navigation buttons
    st.divider()
    col_nav1, col_nav2, col_nav3 = st.columns(3)
    with col_nav1:
        if st.button("Previous", disabled=current_idx == 0):
            st.session_state.filter_index -= 1
            st.session_state.last_button_press_time = time.perf_counter()
            # INSTANT UPDATE - no full rerun!
            prev_idx = st.session_state.filter_index
            update_display(names[prev_idx], prev_idx)
    with col_nav2:
        if st.button("Reset All Decisions", type="secondary"):
            st.session_state.name_inclusions = {}
            st.session_state.filter_index = 0
            # Reset counts
            st.session_state.filter_counts_not_decided = len(names)
            st.session_state.filter_counts_included = 0
            st.session_state.filter_counts_excluded = 0
            st.session_state.filter_counts_names_hash = names_hash
            save_user_setting("name_inclusions", "{}")
            st.session_state.last_button_press_time = time.perf_counter()
            # Fragment-level refresh for reset
            st.rerun(scope="fragment")
    with col_nav3:
        if st.button("Save & Continue", type="primary"):
            save_user_setting("name_inclusions", json.dumps(inclusions))
            st.toast("Decisions saved!", icon="✅")
            # Switch to tournament tab with included names
            st.session_state.active_tab = "Tournament"
            # This needs full rerun to switch tabs
            st.rerun()

    # Show list of excluded names (collapsible)
    with st.expander("Show excluded names"):
        excluded_names = [name for name in names if inclusions.get(name) is False]
        if excluded_names:
            # Compute sorted list once
            sorted_excluded = sorted(excluded_names)
            # Multiselect widget: selected names remain excluded
            selected = st.multiselect(
                f"Excluded names ({len(excluded_names)})",
                sorted_excluded,
                default=sorted_excluded,
                help="Uncheck names to include them in the tournament.",
            )
            # Find names that were deselected (removed from excluded)
            selected_set = set(selected)
            for name in excluded_names:
                if name not in selected_set:
                    # Move from excluded to included
                    old_status = inclusions.get(name)
                    inclusions[name] = True
                    update_counts(name, old_status=old_status, new_status=True)
            # Note: No need to rerun explicitly - widget triggers rerun
        else:
            st.write("No names excluded yet.")

    # Performance logging
    end_time = time.perf_counter()
    elapsed_ms = (end_time - start_time) * 1000

    # Calculate section timings if variables exist
    sections = []
    if "time_after_inclusions" in locals():
        inclusions_ms = (time_after_inclusions - start_time) * 1000
        sections.append(f"inclusions: {inclusions_ms:.1f}ms")
    if "time_after_counts" in locals():
        counts_ms = (
            time_after_counts - (time_after_inclusions if "time_after_inclusions" in locals() else start_time)
        ) * 1000
        sections.append(f"counts: {counts_ms:.1f}ms")
    if "time_after_name_display" in locals():
        name_display_ms = (
            time_after_name_display - (time_after_counts if "time_after_counts" in locals() else start_time)
        ) * 1000
        sections.append(f"name_display: {name_display_ms:.1f}ms")

    section_str = ", ".join(sections) if sections else "no section timing"
    logger.debug("render_binary_filter completed in %.1fms (%s)", elapsed_ms, section_str)


def render_similarity(names: list[str]) -> None:
    """Render similarity search interface.

    Allows searching for names similar to a given name.
    """
    logger.debug("Rendering similarity search with %d names", len(names))
    st.header("Similarity Search")

    search_type: Literal["String (Levenshtein)", "Vector (LLM Embedding)"] = st.radio(
        "Search Method",
        ["String (Levenshtein)", "Vector (LLM Embedding)"],
    )

    query = st.text_input("Reference Name", value="Alma")

    if st.button("Find Similar") and query:
        if search_type == "String (Levenshtein)":
            results = get_string_similarity_scores(query, names, limit=10)
            st.dataframe(
                pd.DataFrame(results, columns=["Name", "Similarity Score"]),
                width="stretch",
                hide_index=True,
            )
        else:
            with st.spinner("Loading Embedding Model (first run is slow)..."):
                model = load_embedding_model()

            with st.spinner("Computing Similarities..."):
                results = get_vector_similarity_scores(
                    model,
                    query,
                    names,
                    limit=10,
                )

            st.dataframe(
                pd.DataFrame(results, columns=["Name", "Cosine Similarity"]),
                width="stretch",
                hide_index=True,
            )
