"""UI rendering functions for the name ranking application."""

import logging
from typing import Literal

import pandas as pd
import streamlit as st

from st_name_ranking.database import (
    INITIAL_SCORE,
    get_preference_stats_by_gender,
    get_preference_stats_by_origin,
    get_preference_stats_by_phonetic,
)
from st_name_ranking.similarity import (
    get_string_similarity_scores,
    get_vector_similarity_scores,
    load_embedding_model,
)
from st_name_ranking.types import PreferenceStats
from st_name_ranking.utils import (
    get_names_features,
    select_candidate_batch,
    select_candidates,
    update_preference_and_save,
    update_preference_down_and_save,
    update_preference_draw_and_save,
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


def render_tournament(names: list[str]) -> None:
    """Render tournament interface for comparing names.

    Shows two names side by side with rating scores.
    User clicks on which name they prefer.
    """

    logger.debug("Rendering tournament with %d names", len(names))
    st.header("Name Ranking Tournament")
    st.write(f"Comparing {len(names)} names")
    st.caption("Tournament mode: click which name you prefer")

    # Precompute features and names set
    features_matrix = get_names_features(names)
    names_set = set(names)

    # Initialize candidates if not set or empty
    if (
        "candidate_a" not in st.session_state
        or "candidate_b" not in st.session_state
        or not st.session_state.candidate_a
        or not st.session_state.candidate_b
    ):
        c_a, c_b = select_candidates(names, features_matrix)
        st.session_state.candidate_a = c_a
        st.session_state.candidate_b = c_b
        # Pre-fill queue with batch for faster navigation
        batch = select_candidate_batch(names, features_matrix, batch_size=3)
        valid_batch = [(a, b) for a, b in batch if a in names_set and b in names_set]
        st.session_state.candidate_queue = valid_batch

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

    _, col_left, _, col_right, _ = st.columns([0.8, 1, 0.4, 1, 0.8])

    # Get both ratings before displaying
    rating_a = st.session_state.ratings.get(
        st.session_state.candidate_a,
        INITIAL_SCORE,
    )
    rating_b = st.session_state.ratings.get(
        st.session_state.candidate_b,
        INITIAL_SCORE,
    )

    # Calculate rating differences for delta display
    delta_a = rating_a - rating_b  # Positive if A is higher rated
    delta_b = rating_b - rating_a  # Positive if B is higher rated

    with col_left:
        display_name_with_rating(
            st.session_state.candidate_a,
            rating_a,
            delta=delta_a,
        )

        # Centered button container
        button_container = st.container()
        with button_container:
            st.markdown(
                "<div style='text-align: center'>",
                unsafe_allow_html=True,
            )
            button_clicked = st.button(
                f"Prefer {st.session_state.candidate_a}",
                key="vote_a",
                width="stretch",
                type="primary",
                shortcut="Left",
                help=(f"Select {st.session_state.candidate_a} as preferred (Left arrow key)"),
            )
            st.markdown("</div>", unsafe_allow_html=True)

        if button_clicked:
            update_preference_and_save(
                st.session_state.ratings,
                st.session_state.candidate_a,
                st.session_state.candidate_b,
            )
            # Clear queue because model has changed
            st.session_state.candidate_queue = []
            # Generate new batch of pairs
            batch = select_candidate_batch(names, features_matrix, batch_size=3)
            # Filter valid pairs (should be all)
            valid_batch = [(a, b) for a, b in batch if a in names_set and b in names_set]
            if valid_batch:
                # Pop first pair for immediate display
                c_a, c_b = valid_batch.pop(0)
                st.session_state.candidate_a = c_a
                st.session_state.candidate_b = c_b
                # Store remaining pairs in queue
                st.session_state.candidate_queue = valid_batch
            else:
                # Fallback to single pair selection
                c_a, c_b = select_candidates(names, features_matrix)
                st.session_state.candidate_a = c_a
                st.session_state.candidate_b = c_b
            st.rerun()

    with col_right:
        display_name_with_rating(
            st.session_state.candidate_b,
            rating_b,
            delta=delta_b,
        )

        # Centered button container (same as left side)
        button_container = st.container()
        with button_container:
            st.markdown(
                "<div style='text-align: center'>",
                unsafe_allow_html=True,
            )
            button_clicked = st.button(
                f"Prefer {st.session_state.candidate_b}",
                key="vote_b",
                width="stretch",
                type="primary",
                shortcut="Right",
                help=(f"Select {st.session_state.candidate_b} as preferred (Right arrow key)"),
            )
            st.markdown("</div>", unsafe_allow_html=True)

        if button_clicked:
            update_preference_and_save(
                st.session_state.ratings,
                st.session_state.candidate_b,
                st.session_state.candidate_a,
            )
            # Clear queue because model has changed
            st.session_state.candidate_queue = []
            # Generate new batch of pairs
            batch = select_candidate_batch(names, features_matrix, batch_size=3)
            # Filter valid pairs (should be all)
            valid_batch = [(a, b) for a, b in batch if a in names_set and b in names_set]
            if valid_batch:
                # Pop first pair for immediate display
                c_a, c_b = valid_batch.pop(0)
                st.session_state.candidate_a = c_a
                st.session_state.candidate_b = c_b
                # Store remaining pairs in queue
                st.session_state.candidate_queue = valid_batch
            else:
                # Fallback to single pair selection
                c_a, c_b = select_candidates(names, features_matrix)
                st.session_state.candidate_a = c_a
                st.session_state.candidate_b = c_b
            st.rerun()

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

    if draw_clicked:
        st.toast("🤝 you chose a draw!", duration="long")
        update_preference_draw_and_save(
            st.session_state.ratings,
            st.session_state.candidate_a,
            st.session_state.candidate_b,
        )
        # Clear queue because model has changed
        st.session_state.candidate_queue = []
        # Generate new batch of pairs
        batch = select_candidate_batch(names, features_matrix, batch_size=3)
        # Filter valid pairs (should be all)
        valid_batch = [(a, b) for a, b in batch if a in names_set and b in names_set]
        if valid_batch:
            # Pop first pair for immediate display
            c_a, c_b = valid_batch.pop(0)
            st.session_state.candidate_a = c_a
            st.session_state.candidate_b = c_b
            # Store remaining pairs in queue
            st.session_state.candidate_queue = valid_batch
        else:
            # Fallback to single pair selection
            c_a, c_b = select_candidates(names, features_matrix)
            st.session_state.candidate_a = c_a
            st.session_state.candidate_b = c_b
        st.rerun()
    elif down_clicked:
        st.toast("👎 you disliked both!", duration="long")
        update_preference_down_and_save(
            st.session_state.ratings,
            st.session_state.candidate_a,
            st.session_state.candidate_b,
        )
        # Clear queue because model has changed
        st.session_state.candidate_queue = []
        # Generate new batch of pairs
        batch = select_candidate_batch(names, features_matrix, batch_size=3)
        # Filter valid pairs (should be all)
        valid_batch = [(a, b) for a, b in batch if a in names_set and b in names_set]
        if valid_batch:
            # Pop first pair for immediate display
            c_a, c_b = valid_batch.pop(0)
            st.session_state.candidate_a = c_a
            st.session_state.candidate_b = c_b
            # Store remaining pairs in queue
            st.session_state.candidate_queue = valid_batch
        else:
            # Fallback to single pair selection
            c_a, c_b = select_candidates(names, features_matrix)
            st.session_state.candidate_a = c_a
            st.session_state.candidate_b = c_b
        st.rerun()

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


def render_binary_filter(names: list[str]) -> None:
    """Render binary filter interface for including/excluding names.

    Users review names one by one, marking them as included/neutral or excluded/dislike.
    """
    logger.debug("Rendering binary filter with %d names", len(names))
    import time

    start_time = time.perf_counter()

    # Clear last button press time (used for performance monitoring)
    if "last_button_press_time" in st.session_state:
        del st.session_state.last_button_press_time

    # Helper function to update counts incrementally
    def update_counts(name: str, old_status: bool | None, new_status: bool) -> None:
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
        "💡 **Keyboard shortcuts**: Left arrow (←) to exclude/dislike, "
        "Right arrow (→) to include/neutral, Space to include/neutral",
    )

    # Load inclusion decisions from session state or database
    if "name_inclusions" not in st.session_state:
        import json

        from st_name_ranking.database import load_user_setting

        try:
            inclusions_json = load_user_setting("name_inclusions", "{}")
            st.session_state.name_inclusions = json.loads(inclusions_json)
        except Exception:
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

    # Display progress
    progress = current_idx / len(names)
    st.progress(progress, text=f"Progress: {current_idx + 1} of {len(names)}")

    # Display stats from session state
    not_decided = st.session_state.filter_counts_not_decided
    explicitly_included = st.session_state.filter_counts_included
    explicitly_excluded = st.session_state.filter_counts_excluded

    st.caption(
        f"Not decided: {not_decided} | "
        f"Included/Neutral: {explicitly_included} | "
        f"Excluded/Dislike: {explicitly_excluded}",
    )

    # Display current name prominently with visual decision indicator
    # Three states: not decided (gray), included/neutral (green), excluded/dislike (red)
    if current_name not in inclusions:
        # Not decided yet
        border_color = "#757575"  # Gray
        status_text = "Not decided"
        bg_color = "#FAFAFA"
    elif inclusions[current_name]:
        # Explicitly included/neutral
        border_color = "#4CAF50"  # Green
        status_text = "Included/Neutral"
        bg_color = "#E8F5E9"
    else:
        # Explicitly excluded/dislike
        border_color = "#F44336"  # Red
        status_text = "Excluded/Dislike"
        bg_color = "#FFEBEE"

    st.markdown(
        f"<div style='border: 4px solid {border_color}; background-color: {bg_color}; "
        f"border-radius: 12px; padding: 20px; text-align: center;'>"
        f"<h1 style='font-size: 72px; margin: 0; color: #212121;'>{current_name}</h1>"
        f"<p style='font-size: 16px; margin: 10px 0 0 0; color: {border_color}; "
        f"font-weight: bold;'>{status_text}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )
    time_after_name_display = time.perf_counter()

    # Decision buttons - simplified to two clear options
    col_exclude, col_include = st.columns(2)
    with col_exclude:
        if st.button(
            "Exclude/Dislike",
            key="exclude_btn",
            use_container_width=True,
            type="secondary",
            shortcut="Left",
        ):
            old_status = inclusions.get(current_name)
            inclusions[current_name] = False
            update_counts(current_name, old_status, False)
            st.session_state.filter_index += 1
            st.toast(f"Excluded: {current_name}", icon="👎")
            st.session_state.last_button_press_time = time.perf_counter()
            st.rerun()
    with col_include:
        if st.button(
            "Include/Neutral",
            key="include_btn",
            use_container_width=True,
            type="primary",
            shortcut="Right",
        ):
            # Include (explicitly mark as included/neutral)
            old_status = inclusions.get(current_name)
            inclusions[current_name] = True
            update_counts(current_name, old_status, True)
            st.session_state.filter_index += 1
            st.toast(f"Included: {current_name}", icon="👍")
            st.session_state.last_button_press_time = time.perf_counter()
            st.rerun()

    # Save decisions periodically (every 50 actions to reduce DB writes)
    if current_idx % 50 == 0:
        import json

        from st_name_ranking.database import save_user_setting

        # Time JSON serialization and database save
        json_start = time.perf_counter()
        inclusions_json = json.dumps(inclusions)
        json_time = (time.perf_counter() - json_start) * 1000

        save_start = time.perf_counter()
        save_user_setting("name_inclusions", inclusions_json)
        save_time = (time.perf_counter() - save_start) * 1000

        logger.debug(f"Periodic save: JSON={json_time:.1f}ms, DB={save_time:.1f}ms, entries={len(inclusions)}")

    # Batch operations
    col_batch1, col_batch2 = st.columns(2)
    with col_batch1:
        if st.button("Include All Remaining", type="secondary", help="Include all remaining names as neutral"):
            count = len(names) - current_idx
            for name in names[current_idx:]:
                inclusions[name] = True
            st.session_state.filter_index = len(names)  # Move to end
            # Invalidate counts cache to force recomputation
            if "filter_counts_names_hash" in st.session_state:
                del st.session_state.filter_counts_names_hash
            st.toast(f"Included {count} remaining names", icon="✅")
            st.session_state.last_button_press_time = time.perf_counter()
            st.rerun()
    with col_batch2:
        if st.button("Exclude All Remaining", type="secondary", help="Exclude/dislike all remaining names"):
            count = len(names) - current_idx
            for name in names[current_idx:]:
                inclusions[name] = False
            st.session_state.filter_index = len(names)
            # Invalidate counts cache to force recomputation
            if "filter_counts_names_hash" in st.session_state:
                del st.session_state.filter_counts_names_hash
            st.toast(f"Excluded {count} remaining names", icon="✅")
            st.session_state.last_button_press_time = time.perf_counter()
            st.rerun()

    # Navigation buttons
    st.divider()
    col_nav1, col_nav2, col_nav3 = st.columns(3)
    with col_nav1:
        if st.button("Previous", disabled=current_idx == 0):
            st.session_state.filter_index -= 1
            st.session_state.last_button_press_time = time.perf_counter()
            st.rerun()
    with col_nav2:
        if st.button("Reset All Decisions", type="secondary"):
            st.session_state.name_inclusions = {}
            st.session_state.filter_index = 0
            # Reset counts
            st.session_state.filter_counts_not_decided = len(names)
            st.session_state.filter_counts_included = 0
            st.session_state.filter_counts_excluded = 0
            st.session_state.filter_counts_names_hash = names_hash
            from st_name_ranking.database import save_user_setting

            save_user_setting("name_inclusions", "{}")
            st.session_state.last_button_press_time = time.perf_counter()
            st.rerun()
    with col_nav3:
        if st.button("Save & Continue", type="primary"):
            import json

            from st_name_ranking.database import save_user_setting

            save_user_setting("name_inclusions", json.dumps(inclusions))
            st.toast("Decisions saved!", icon="✅")
            # Switch to tournament tab with included names
            st.session_state.active_tab = "Tournament"
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
                    update_counts(name, old_status, True)
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
    logger.debug(f"render_binary_filter completed in {elapsed_ms:.1f}ms ({section_str})")


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
