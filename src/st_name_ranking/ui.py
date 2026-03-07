"""UI rendering functions for the name ranking application."""

import json
import logging
import os
import time
from typing import Literal

import numpy as np
import polars as pl
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from st_name_ranking.async_model import select_random_pair
from st_name_ranking.background_queue import get_queue_manager, get_queue_manager_stats
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
from st_name_ranking.utils import (
    get_active_learning_model,
    get_names_features,
    record_comparison_instant,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
MIN_NAMES_FOR_COMPARISON = 2
MIN_NAMES_FOR_LANDSCAPE = 25
MIN_NON_NOISE_CLUSTERS = 2
FAST_REFILL_THRESHOLD_MS = 120
MODERATE_REFILL_THRESHOLD_MS = 300
SLOW_RENDER_THRESHOLD_MS = 100

try:
    import altair as alt
except ImportError:
    alt = None

try:
    import pacmap
except ImportError:
    pacmap = None

try:
    import hdbscan
except ImportError:
    hdbscan = None

try:
    from sklearn.cluster import HDBSCAN as SKHDBSCAN
except ImportError:
    SKHDBSCAN = None


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
    delta_str = (f"{delta:+.0f}" if isinstance(delta, (int, float)) else str(delta)) if delta is not None else None

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

    def create_percentage_dataframe(stats: dict[str, PreferenceStats]) -> pl.DataFrame:
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
        return pl.DataFrame(rows)

    def create_stacked_bar_chart(df: pl.DataFrame, title: str) -> None:
        """Create stacked bar chart showing win/loss/draw percentages."""
        if df.is_empty():
            return

        # Create percentage columns for stacked bar chart
        chart_df = df.select(["Group", "win_pct", "loss_pct", "draw_pct"])

        # Sort by win percentage (descending)
        chart_df = chart_df.sort("win_pct", descending=True)

        # Create stacked bar chart
        st.subheader(title, divider="gray")

        # Display chart with custom colors
        st.bar_chart(
            chart_df,
            x="Group",
            y=["win_pct", "loss_pct", "draw_pct"],
            height=400,
            width="stretch",
            color=["#2E7D32", "#C62828", "#FF9800"],  # Green for wins, red for losses, orange for draws
        )

        # Add data table below chart
        display_df = df.sort("win_pct", descending=True)

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
                    "win_pct": st.column_config.NumberColumn("Win %", format="%.1f%%", width="small"),
                    "loss_pct": st.column_config.NumberColumn("Loss %", format="%.1f%%", width="small"),
                    "draw_pct": st.column_config.NumberColumn("Draw %", format="%.1f%%", width="small"),
                    "win_rate_pct": st.column_config.NumberColumn(
                        "Win Rate %",
                        help="Wins / (Wins + Losses)",
                        format="%.1f%%",
                        width="small",
                    ),
                },
            )

        # Add legend
        st.caption("🎯 **Legend**: 🟢 Wins | 🔴 Losses | 🟠 Draws")

        # Add key insight based on data
        if not df.is_empty():
            sorted_by_win_rate = df.sort("win_pct", descending=True)
            best_group = sorted_by_win_rate.row(0, named=True)
            worst_group = sorted_by_win_rate.row(sorted_by_win_rate.height - 1, named=True)
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
    start_time = time.perf_counter()

    def log_timing(step: str) -> None:
        logger.info("⏱️ Tournament [%s]: %.2fms", step, (time.perf_counter() - start_time) * 1000)

    logger.info("🎮 Tournament started with %d names", len(names))

    total_filtered = st.session_state.get("tournament_filtered_count", len(names))
    selected_sample_size = int(st.session_state.get("tournament_sample_size", 50))
    if len(names) < total_filtered:
        st.write(f"Comparing {len(names)} sampled names")
    else:
        st.write(f"Comparing {len(names)} names")
    st.caption(f"Selection sample size: {selected_sample_size}")

    log_timing("Before queue manager")

    # Get or create background queue manager for instant transitions
    # Queue size from environment variable (default 15)
    queue_size = int(os.environ.get("TOURNAMENT_QUEUE_SIZE", "15"))
    manager = get_queue_manager(names, queue_size, sample_size=selected_sample_size)
    log_timing("After queue manager")

    queue_stats = get_queue_manager_stats()
    if queue_stats and int(queue_stats.get("refill_count", 0)) > 0:
        last_refill_ms = float(queue_stats.get("last_refill_ms", 0.0))
        avg_refill_ms = float(queue_stats.get("avg_refill_ms", 0.0))
        refill_added = int(queue_stats.get("last_refill_added", 0))
        current_queue_size = int(queue_stats.get("queue_size", 0))
        target_queue_size = int(queue_stats.get("target_size", queue_size))

        if last_refill_ms <= FAST_REFILL_THRESHOLD_MS:
            latency_indicator = "🟢"
        elif last_refill_ms <= MODERATE_REFILL_THRESHOLD_MS:
            latency_indicator = "🟡"
        else:
            latency_indicator = "🔴"

        st.caption(
            f"{latency_indicator} Queue {current_queue_size}/{target_queue_size} | "
            f"Last refill {last_refill_ms:.0f} ms (avg {avg_refill_ms:.0f} ms, +{refill_added} pairs)",
        )
    else:
        st.caption("Queue warming up...")

    st.caption("Tournament mode: click which name you prefer")

    # Handle empty names list gracefully
    if len(names) < MIN_NAMES_FOR_COMPARISON:
        if len(names) == 0:
            st.info("No names to compare. Please select at least two names.")
        else:
            st.info(f"Only one name ('{names[0]}') selected. Please select at least two names to compare.")
        return

    # Create names set for filtering
    names_set = set(names)

    # Create placeholders for dynamic content
    pair_display_placeholder = st.empty()
    st.empty()

    # Initialize candidates if not set, empty, or no longer in current filtered names
    if (
        "candidate_a" not in st.session_state
        or "candidate_b" not in st.session_state
        or not st.session_state.candidate_a
        or not st.session_state.candidate_b
        or st.session_state.candidate_a not in names_set
        or st.session_state.candidate_b not in names_set
        or st.session_state.candidate_a == st.session_state.candidate_b
    ):
        current_pair = manager.get_pair()
        if not current_pair:
            # Fallback if queue empty (fresh pair on reload/new session)
            current_pair = select_random_pair(names)
        st.session_state.candidate_a, st.session_state.candidate_b = current_pair

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

    log_timing("After button creation")

    # Handle votes with instant async model updates for UI feedback
    vote_handled = False
    next_pair = None

    if vote_a_clicked:
        log_timing("Vote A clicked - handling")
        vote_handled = True
        logger.info("🎮 Vote: '%s' preferred over '%s'", st.session_state.candidate_a, st.session_state.candidate_b)
        # 1. Record comparison instantly (async model update in background)
        record_comparison_instant(st.session_state.candidate_a, st.session_state.candidate_b, -1)
        # 2. Get next pair instantly from background queue
        next_pair = manager.get_pair()
        if not next_pair:
            next_pair = select_random_pair(names)
            logger.info("📦 Next pair (random): %s vs %s", next_pair[0], next_pair[1])
        else:
            logger.info("📦 Next pair (queue): %s vs %s", next_pair[0], next_pair[1])
    elif vote_b_clicked:
        vote_handled = True
        logger.info("🎮 Vote: '%s' preferred over '%s'", st.session_state.candidate_b, st.session_state.candidate_a)
        # 1. Record comparison instantly (async model update in background)
        record_comparison_instant(st.session_state.candidate_a, st.session_state.candidate_b, 1)
        # 2. Get next pair instantly from background queue
        next_pair = manager.get_pair()
        if not next_pair:
            next_pair = select_random_pair(names)
            logger.info("📦 Next pair (random): %s vs %s", next_pair[0], next_pair[1])
        else:
            logger.info("📦 Next pair (queue): %s vs %s", next_pair[0], next_pair[1])
    elif draw_clicked:
        vote_handled = True
        logger.info("🎮 Vote: Draw between '%s' and '%s'", st.session_state.candidate_a, st.session_state.candidate_b)
        st.toast("🤝 you chose a draw!", duration="long")
        # 1. Record comparison instantly (async model update in background)
        record_comparison_instant(st.session_state.candidate_a, st.session_state.candidate_b, 0)
        # 2. Get next pair instantly from background queue
        next_pair = manager.get_pair()
        if not next_pair:
            next_pair = select_random_pair(names)
            logger.info("📦 Next pair (random): %s vs %s", next_pair[0], next_pair[1])
        else:
            logger.info("📦 Next pair (queue): %s vs %s", next_pair[0], next_pair[1])
    elif down_clicked:
        vote_handled = True
        logger.info("🎮 Vote: Both '%s' and '%s' disliked", st.session_state.candidate_a, st.session_state.candidate_b)
        st.toast("👎 you disliked both!", duration="long")
        # 1. Record comparison instantly (async model update in background)
        record_comparison_instant(st.session_state.candidate_a, st.session_state.candidate_b, 2)
        # 2. Get next pair instantly from background queue
        next_pair = manager.get_pair()
        if not next_pair:
            next_pair = select_random_pair(names)
            logger.info("📦 Next pair (random): %s vs %s", next_pair[0], next_pair[1])
        else:
            logger.info("📦 Next pair (queue): %s vs %s", next_pair[0], next_pair[1])

    # Update session state and UI instantly after vote (no rerun needed)
    if vote_handled and next_pair:
        old_a, old_b = st.session_state.candidate_a, st.session_state.candidate_b
        st.session_state.candidate_a, st.session_state.candidate_b = next_pair
        logger.info("🔄 Transition: (%s, %s) → (%s, %s)", old_a, old_b, next_pair[0], next_pair[1])
        # Update UI instantly with st.fragment placeholder updates
        update_display(next_pair[0], next_pair[1], st.session_state.ratings)

    # Render display with current candidates (will reflect any updates above)
    update_display(st.session_state.candidate_a, st.session_state.candidate_b, st.session_state.ratings)
    log_timing("After display render")

    # NOTE: Statistics expander removed due to 20+ second render time
    # The dataframes and tabs were being processed even when collapsed

    log_timing("At end")


def render_rankings(names: list[str]) -> None:
    """Render rankings view showing top rated names.

    This is a separate tab to avoid slowing down the tournament UI.
    Only renders when the Rankings tab is active.
    """
    logger.info("🏅 Rendering rankings for %d names", len(names))
    st.header("Name Rankings")
    st.write(f"Showing ratings for {len(names)} names")

    if len(names) == 0:
        st.info("No names to rank. Please include some names in the Name Filter tab first.")
        return

    # Create names set for filtering
    names_set = set(names)

    # Get gender-specific name lists if available
    male_names = []
    female_names = []
    if "all_names_data" in st.session_state:
        gender_data = st.session_state.all_names_data
        male_names = gender_data.get("Male", [])
        female_names = gender_data.get("Female", [])

    # Filter ratings to only show current selection
    filtered_ratings = {name: rating for name, rating in st.session_state.ratings.items() if name in names_set}

    @st.cache_data(show_spinner=False)
    def build_rankings_dataframe(
        ratings_pairs: tuple[tuple[str, float], ...],
        *,
        include_gender_male: bool,
    ) -> tuple[pl.DataFrame, list[str]]:
        sorted_pairs = sorted(ratings_pairs, key=lambda item: item[1], reverse=True)
        ordered_names = [name for name, _ in sorted_pairs]
        ordered_ratings = [rating for _, rating in sorted_pairs]
        base_df = pl.DataFrame({"Name": ordered_names, "Rating": ordered_ratings})

        try:
            model = get_active_learning_model()
            feature_names = list(model.feature_names)
            feature_matrix = get_names_features(ordered_names)
            top_feature_count = min(6, len(feature_names))
            top_feature_idx = np.argsort(np.abs(model.state.weight_mean))[::-1][:top_feature_count]

            feature_columns = {feature_names[idx]: feature_matrix[:, idx].astype(float) for idx in top_feature_idx}
            enriched_df = base_df.with_columns(
                [pl.Series(feature_name, values) for feature_name, values in feature_columns.items()],
            )
            selected_features = [feature_names[idx] for idx in top_feature_idx]
            if not include_gender_male and "gender_male" in enriched_df.columns:
                enriched_df = enriched_df.drop("gender_male")
                selected_features = [name for name in selected_features if name != "gender_male"]
        except (RuntimeError, ValueError, AttributeError):
            logger.exception("Failed to enrich rankings table with feature columns")
            return base_df, []
        else:
            return enriched_df, selected_features

    @st.cache_data(show_spinner=False)
    def compute_landscape(
        sorted_names: tuple[str, ...],
        ratings_pairs: tuple[tuple[str, float], ...],
        random_state: int,
    ) -> tuple[pl.DataFrame, np.ndarray, list[str], str]:
        ratings_dict = dict(ratings_pairs)
        model = get_active_learning_model()
        feature_names = list(model.feature_names)
        feature_matrix = get_names_features(list(sorted_names))
        scaled_features = StandardScaler().fit_transform(feature_matrix)

        weight_mean = model.state.weight_mean
        weight_cov = model.state.weight_cov

        utility = feature_matrix @ weight_mean
        variance = np.einsum("ij,jk,ik->i", feature_matrix, weight_cov, feature_matrix)
        uncertainty = np.sqrt(np.clip(variance, a_min=0.0, a_max=None))
        ratings = np.array([ratings_dict.get(name, INITIAL_SCORE) for name in sorted_names], dtype=np.float64)
        confidence = 1.0 / (1.0 + uncertainty)

        status_note = "PaCMAP projection + HDBSCAN clustering"
        if pacmap is not None:
            projection = pacmap.PaCMAP(
                n_components=2,
                n_neighbors=min(15, max(3, len(sorted_names) - 1)),
                MN_ratio=0.5,
                FP_ratio=2.0,
                random_state=random_state,
            ).fit_transform(scaled_features)
        else:
            projection = PCA(n_components=2, random_state=random_state).fit_transform(scaled_features)
            status_note = "PaCMAP unavailable, using PCA projection + HDBSCAN clustering"

        labels: np.ndarray
        hdbscan_valid = False
        min_cluster_size = max(20, int(0.01 * len(sorted_names)))
        min_samples = max(5, min_cluster_size // 2)

        if hdbscan is not None:
            try:
                labels = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_method="eom",
                    metric="euclidean",
                    prediction_data=True,
                ).fit_predict(projection)
                non_noise = labels[labels >= 0]
                hdbscan_valid = len(non_noise) > 0 and len(np.unique(non_noise)) >= MIN_NON_NOISE_CLUSTERS
            except (ValueError, RuntimeError):
                hdbscan_valid = False

        if not hdbscan_valid and SKHDBSCAN is not None:
            try:
                labels = SKHDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_method="eom",
                    copy=False,
                ).fit_predict(
                    projection,
                )
                non_noise = labels[labels >= 0]
                hdbscan_valid = len(non_noise) > 0 and len(np.unique(non_noise)) >= MIN_NON_NOISE_CLUSTERS
            except (ValueError, RuntimeError):
                hdbscan_valid = False

        if not hdbscan_valid:
            cluster_count = min(8, max(2, len(sorted_names) // 12))
            labels = KMeans(n_clusters=cluster_count, random_state=random_state, n_init=10).fit_predict(projection)
            status_note = f"{status_note} (HDBSCAN fallback: KMeans)"

        landscape = pl.DataFrame(
            {
                "Name": list(sorted_names),
                "Projection X": projection[:, 0],
                "Projection Y": projection[:, 1],
                "Cluster": labels,
                "Rating": ratings,
                "Utility": utility,
                "Uncertainty": uncertainty,
                "Confidence": confidence,
            },
        )
        return landscape, feature_matrix, feature_names, status_note

    if not filtered_ratings:
        st.info("No ratings yet. Start comparing names in the Tournament tab to generate rankings.")
        return

    # Create tabs for different views
    tab_overall, tab_male, tab_female = st.tabs(["Overall", "Male", "Female"])

    with tab_overall:
        overall_pairs = tuple(filtered_ratings.items())
        df, feature_columns = build_rankings_dataframe(overall_pairs, include_gender_male=True)
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
        if feature_columns:
            st.caption(f"Feature columns shown: {', '.join(feature_columns)}")

        st.divider()
        st.subheader("Preference Landscape")

        if len(filtered_ratings) < MIN_NAMES_FOR_LANDSCAPE:
            st.info(f"Preference landscape appears after at least {MIN_NAMES_FOR_LANDSCAPE} rated names.")
        else:
            try:
                random_state = st.slider(
                    "Projection seed",
                    min_value=0,
                    max_value=99,
                    value=42,
                    key="rankings_projection_seed",
                )

                sorted_names = tuple(sorted(filtered_ratings))
                name_to_index = {name: idx for idx, name in enumerate(sorted_names)}
                ratings_pairs = tuple(sorted(filtered_ratings.items()))
                with st.status("Building preference landscape...", expanded=False) as status:
                    status.write("Projecting names with PaCMAP")
                    status.write("Clustering projection with HDBSCAN")
                    landscape_df, feature_matrix, feature_names, status_note = compute_landscape(
                        sorted_names,
                        ratings_pairs,
                        random_state,
                    )
                    status.update(label="Preference landscape ready", state="complete")

                st.caption(status_note)

                plot_df = landscape_df.with_columns(
                    pl.col("Cluster").cast(pl.String).alias("Cluster Label"),
                    (20 + pl.col("Confidence") * 280).alias("Point Size"),
                ).to_pandas()
                if alt is not None:
                    chart = (
                        alt.Chart(plot_df)
                        .mark_circle(opacity=0.85)
                        .encode(
                            x=alt.X("Projection X:Q", title="Component 1"),
                            y=alt.Y("Projection Y:Q", title="Component 2"),
                            color=alt.Color("Cluster Label:N", title="Cluster"),
                            size=alt.Size("Point Size:Q", legend=None),
                            tooltip=[
                                alt.Tooltip("Name:N"),
                                alt.Tooltip("Rating:Q", format=".1f"),
                                alt.Tooltip("Cluster Label:N", title="Cluster"),
                                alt.Tooltip("Utility:Q", format=".4f"),
                                alt.Tooltip("Uncertainty:Q", format=".4f"),
                            ],
                        )
                        .properties(height=480)
                    )
                    st.altair_chart(chart, width="stretch")
                else:
                    st.scatter_chart(
                        landscape_df,
                        x="Projection X",
                        y="Projection Y",
                        color="Cluster",
                        size="Confidence",
                        width="stretch",
                    )

                model = get_active_learning_model()
                feature_weights = model.state.weight_mean
                feature_rank = np.argsort(np.abs(feature_weights))[::-1]
                top_k = min(8, len(feature_names))
                global_rows = [
                    {
                        "Feature": feature_names[idx],
                        "Weight": float(feature_weights[idx]),
                        "Direction": "Positive" if feature_weights[idx] >= 0 else "Negative",
                        "Strength": float(abs(feature_weights[idx])),
                    }
                    for idx in feature_rank[:top_k]
                ]

                st.markdown("**Global predictors**")
                st.dataframe(
                    pl.DataFrame(global_rows),
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "Weight": st.column_config.NumberColumn(format="%.4f"),
                        "Strength": st.column_config.NumberColumn(format="%.4f"),
                    },
                )

                summary_df = (
                    landscape_df.group_by("Cluster")
                    .agg(
                        pl.len().alias("Size"),
                        pl.col("Rating").mean().alias("Avg Rating"),
                        pl.col("Utility").mean().alias("Avg Utility"),
                        pl.col("Uncertainty").mean().alias("Avg Uncertainty"),
                    )
                    .sort("Size", descending=True)
                )

                st.markdown("**Cluster summary**")
                st.dataframe(
                    summary_df,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "Avg Rating": st.column_config.NumberColumn(format="%.1f"),
                        "Avg Utility": st.column_config.NumberColumn(format="%.4f"),
                        "Avg Uncertainty": st.column_config.NumberColumn(format="%.4f"),
                    },
                )

                cluster_profiles = []
                for cluster_id in summary_df["Cluster"].to_list():
                    cluster_names = landscape_df.filter(pl.col("Cluster") == cluster_id)["Name"].to_list()
                    cluster_idx = [name_to_index[name] for name in cluster_names]
                    cluster_features = feature_matrix[cluster_idx]

                    contribution = cluster_features.mean(axis=0) * feature_weights
                    rank_idx = np.argsort(np.abs(contribution))[::-1][:3]
                    label_tokens = [
                        f"{feature_names[idx]} ({'+' if contribution[idx] >= 0 else '-'}{abs(contribution[idx]):.3f})"
                        for idx in rank_idx
                    ]
                    cluster_profiles.append(
                        {
                            "Cluster": cluster_id,
                            "Profile": " | ".join(label_tokens),
                        },
                    )

                st.markdown("**Cluster profiles**")
                st.dataframe(
                    pl.DataFrame(cluster_profiles).sort("Cluster"),
                    hide_index=True,
                    width="stretch",
                )
            except (RuntimeError, ValueError, ImportError) as err:
                logger.exception("Failed to render preference landscape")
                st.info(f"Preference landscape is temporarily unavailable: {err}")

    with tab_male:
        if male_names:
            male_ratings = {
                name: rating
                for name, rating in st.session_state.ratings.items()
                if name in male_names and name in names_set
            }
            if male_ratings:
                df_male, _ = build_rankings_dataframe(
                    tuple(male_ratings.items()),
                    include_gender_male=False,
                )
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
                st.info("No male names rated yet.")
        else:
            st.info("No gender data available for male names.")

    with tab_female:
        if female_names:
            female_ratings = {
                name: rating
                for name, rating in st.session_state.ratings.items()
                if name in female_names and name in names_set
            }
            if female_ratings:
                df_female, _ = build_rankings_dataframe(
                    tuple(female_ratings.items()),
                    include_gender_male=False,
                )
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
                st.info("No female names rated yet.")
        else:
            st.info("No gender data available for female names.")


@st.fragment
def render_binary_filter(names: list[str]) -> None:
    """Render binary filter interface for including/excluding names.

    Users review names one by one, marking them as included or excluded.
    """
    logger.info("🎛️  Filter rendering %d names", len(names))
    start_time = time.perf_counter()

    def log_timing(step_name: str) -> None:
        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info("⏱️  %s: %.1fms", step_name, elapsed)

    # Clear last button press time (used for performance monitoring)
    if "last_button_press_time" in st.session_state:
        del st.session_state.last_button_press_time

    # Helper function to update counts incrementally
    def update_counts(_name: str, *, old_status: bool | None, new_status: bool | None) -> None:
        """Update filter counts when a name's inclusion status changes."""
        # old_status: None (not decided), True (included), False (excluded)

        # Decrement old category
        if old_status is None:
            st.session_state.filter_counts_not_decided -= 1
        elif old_status is True:
            st.session_state.filter_counts_included -= 1
        else:  # old_status is False
            st.session_state.filter_counts_excluded -= 1

        # Increment new category
        if new_status is None:
            st.session_state.filter_counts_not_decided += 1
        elif new_status is True:
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

    # Load inclusion decisions from session state or database (cached)
    cache_key = "name_inclusions_loaded"
    if cache_key not in st.session_state:
        try:
            inclusions_json = load_user_setting("name_inclusions", "{}")
            st.session_state.name_inclusions = json.loads(inclusions_json)
            st.session_state[cache_key] = True
            logger.debug("Loaded %d inclusions from database", len(st.session_state.name_inclusions))
        except json.JSONDecodeError:
            st.session_state.name_inclusions = {}
            st.session_state[cache_key] = True

    inclusions = st.session_state.name_inclusions
    log_timing("After inclusions loaded")

    # Detect if names list has changed (e.g., gender/origin filter changed)
    # Use a faster hash: just hash the first, last, and length
    # This is O(1) instead of O(n) for hashing the full tuple
    fast_hash = hash((names[0], names[-1], len(names))) if len(names) > 0 else hash(0)
    names_hash = str(fast_hash)

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
    needs_recount = (
        "filter_counts_not_decided" not in st.session_state
        or "filter_counts_included" not in st.session_state
        or "filter_counts_excluded" not in st.session_state
        or st.session_state.get("filter_counts_names_hash") != names_hash
    )

    if needs_recount:
        # Recompute counts from scratch - O(n) operation
        logger.info("📊 Computing counts for %d names...", len(names))
        count_loop_start = time.perf_counter()
        not_decided = explicitly_included = explicitly_excluded = 0
        for name in names:
            status = inclusions.get(name)
            if status is None:
                not_decided += 1
            elif status is True:
                explicitly_included += 1
            else:  # status is False
                explicitly_excluded += 1
        count_loop_time = time.perf_counter() - count_loop_start

        st.session_state.filter_counts_not_decided = not_decided
        st.session_state.filter_counts_included = explicitly_included
        st.session_state.filter_counts_excluded = explicitly_excluded
        st.session_state.filter_counts_names_hash = names_hash

        logger.info(
            "✅ Counts computed: %d not decided, %d included, %d excluded (%.1fms)",
            not_decided,
            explicitly_included,
            explicitly_excluded,
            count_loop_time * 1000,
        )

    log_timing("After counts")

    # Filter to only undecided names for the progress tracking
    # This allows users to resume filtering without going through all already-decided names
    undecided_names = [name for name in names if name not in inclusions]

    # If no undecided names, show completion message
    if not undecided_names:
        st.success("✅ All names have been processed! Switch to the Tournament tab to compare your selected names.")
        return

    # Get current name from undecided list
    current_idx = st.session_state.filter_index
    if current_idx >= len(undecided_names):
        current_idx = 0
        st.session_state.filter_index = 0
    current_name = undecided_names[current_idx]

    def update_display(current_name: str, current_idx: int) -> None:
        """Update display instantly without rerun."""
        # Progress bar - show progress through UNDECIDED names only
        progress = current_idx / len(undecided_names)
        progress_placeholder.progress(progress, text=f"Progress: {current_idx + 1} of {len(undecided_names)} remaining")

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
    log_timing("After display update")

    # Decision buttons - simplified to two clear options
    col_exclude, col_include = st.columns(2)
    with col_exclude:
        if st.button(
            "Exclude",
            key="exclude_btn",
            width="stretch",
            type="secondary",
            shortcut="Left",
        ):
            logger.info("👎 Excluding: %s", current_name)
            button_click_start = time.perf_counter()
            old_status = inclusions.get(current_name)
            inclusions[current_name] = False
            update_counts(current_name, old_status=old_status, new_status=False)
            st.session_state.filter_index += 1
            st.toast(f"Excluded: {current_name}", icon="👎")
            st.session_state.last_button_press_time = time.perf_counter()
            # Save to database immediately to prevent data loss on reload
            save_user_setting("name_inclusions", json.dumps(inclusions))
            # INSTANT UPDATE - no rerun!
            # Remove current name from undecided list for instant feedback
            if current_name in undecided_names:
                undecided_names.remove(current_name)
            next_idx = st.session_state.filter_index
            if next_idx < len(undecided_names):
                update_display(undecided_names[next_idx], next_idx)
            else:
                # All done!
                st.success("✅ All names processed! Switch to Tournament tab.")
            logger.info("⚡ Exclude handled in %.1fms", (time.perf_counter() - button_click_start) * 1000)
    with col_include:
        if st.button(
            "Include",
            key="include_btn",
            width="stretch",
            type="primary",
            shortcut="Right",
        ):
            logger.info("👍 Including: %s", current_name)
            button_click_start = time.perf_counter()
            # Include (explicitly mark as included)
            old_status = inclusions.get(current_name)
            inclusions[current_name] = True
            update_counts(current_name, old_status=old_status, new_status=True)
            st.session_state.filter_index += 1
            st.toast(f"Included: {current_name}", icon="👍")
            st.session_state.last_button_press_time = time.perf_counter()
            # Save to database immediately to prevent data loss on reload
            save_user_setting("name_inclusions", json.dumps(inclusions))
            # INSTANT UPDATE - no rerun!
            # Remove current name from undecided list for instant feedback
            if current_name in undecided_names:
                undecided_names.remove(current_name)
            next_idx = st.session_state.filter_index
            if next_idx < len(undecided_names):
                update_display(undecided_names[next_idx], next_idx)
            else:
                # All done!
                st.success("✅ All names processed! Switch to Tournament tab.")
            logger.info("⚡ Include handled in %.1fms", (time.perf_counter() - button_click_start) * 1000)

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
        if st.button("Include All Remaining", type="secondary", help="Include all remaining undecided names"):
            count = len(undecided_names) - current_idx
            for name in undecided_names[current_idx:]:
                inclusions[name] = True
            st.session_state.filter_index = 0  # Reset since undecided list will be empty
            # Invalidate counts cache to force recomputation
            if "filter_counts_names_hash" in st.session_state:
                del st.session_state.filter_counts_names_hash
            st.toast(f"Included {count} remaining names", icon="✅")
            st.session_state.last_button_press_time = time.perf_counter()
            # Save to database immediately to prevent data loss on reload
            save_user_setting("name_inclusions", json.dumps(inclusions))
            # Fragment-level refresh for batch operations
            st.rerun(scope="fragment")
    with col_batch2:
        if st.button("Exclude All Remaining", type="secondary", help="Exclude all remaining undecided names"):
            count = len(undecided_names) - current_idx
            for name in undecided_names[current_idx:]:
                inclusions[name] = False
            st.session_state.filter_index = 0  # Reset since undecided list will be empty
            # Invalidate counts cache to force recomputation
            if "filter_counts_names_hash" in st.session_state:
                del st.session_state.filter_counts_names_hash
            st.toast(f"Excluded {count} remaining names", icon="✅")
            st.session_state.last_button_press_time = time.perf_counter()
            # Save to database immediately to prevent data loss on reload
            save_user_setting("name_inclusions", json.dumps(inclusions))
            # Fragment-level refresh for batch operations
            st.rerun(scope="fragment")

    log_timing("After buttons")

    # Show included names in a multiselect (usually small number, performant)
    # Get list of included names for the multiselect
    included_names = [name for name in names if inclusions.get(name) is True]
    excluded_count = st.session_state.filter_counts_excluded

    st.divider()
    st.subheader("📋 Your Selections")

    # Only show multiselect if there are included names
    if included_names:
        # Sort for consistent display
        sorted_included = sorted(included_names)

        selected_included = st.multiselect(
            f"✅ Included names for tournament ({len(sorted_included)})",
            options=sorted_included,
            default=sorted_included,
            help="Uncheck names to move them back to 'not decided'",
        )

        # Check if any were deselected
        selected_set = set(selected_included)
        for name in sorted_included:
            if name not in selected_set:
                # User unchecked this name - move to not decided
                old_status = inclusions.get(name)
                del inclusions[name]  # Remove from dict = not decided
                update_counts(name, old_status=old_status, new_status=None)
                logger.info("🔄 %s moved from included to not decided", name)
                st.toast(f"{name} moved to not decided", icon="🔄")
                # Save to database immediately to prevent data loss on reload
                save_user_setting("name_inclusions", json.dumps(inclusions))
                st.rerun(scope="fragment")
    else:
        st.info("No names included yet. Use 'Include' button above to add names.")

    # Show excluded names in an expander (lazy loaded to avoid performance issues)
    if excluded_count > 0:
        with st.expander(f"❌ Show Excluded Names ({excluded_count})"):
            # Get list of excluded names
            excluded_names = [name for name in names if inclusions.get(name) is False]
            if excluded_names:
                # Sort and limit for performance
                sorted_excluded = sorted(excluded_names)[:100]
                remaining = len(excluded_names) - 100

                if remaining > 0:
                    st.caption(f"Showing first 100 of {len(excluded_names)} excluded names")

                # Allow user to un-exclude (move back to undecided)
                selected_excluded = st.multiselect(
                    "Uncheck names to include them again",
                    options=sorted_excluded,
                    default=sorted_excluded,
                    help="Uncheck names to move them back to 'not decided'",
                )

                # Check if any were deselected
                selected_set = set(selected_excluded)
                for name in sorted_excluded:
                    if name not in selected_set:
                        # User unchecked this name - move to not decided
                        old_status = inclusions.get(name)
                        del inclusions[name]  # Remove from dict = not decided
                        update_counts(name, old_status=old_status, new_status=None)
                        logger.info("🔄 %s moved from excluded to not decided", name)
                        st.toast(f"{name} moved to not decided", icon="🔄")
                        # Save to database immediately to prevent data loss on reload
                        save_user_setting("name_inclusions", json.dumps(inclusions))
                        st.rerun(scope="fragment")

    log_timing("At end")

    # Performance logging
    end_time = time.perf_counter()
    elapsed_ms = (end_time - start_time) * 1000

    # Log timing at INFO level if slow, DEBUG otherwise
    if elapsed_ms > SLOW_RENDER_THRESHOLD_MS:
        logger.info("🐌 Filter render slow: %.1fms", elapsed_ms)
    else:
        logger.info("✅ Filter render fast: %.1fms", elapsed_ms)


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
                pl.DataFrame(results, schema=["Name", "Similarity Score"], orient="row"),
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
                pl.DataFrame(results, schema=["Name", "Cosine Similarity"], orient="row"),
                width="stretch",
                hide_index=True,
            )
