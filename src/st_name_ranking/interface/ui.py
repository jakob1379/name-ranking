"""UI rendering functions for the name ranking application."""

import json
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import polars as pl
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from st_name_ranking.active_learning.selection import (
    get_names_features,
    get_or_initialize_active_learning_model,
)
from st_name_ranking.interface.filter_state import (
    FilterCounts,
    apply_filter_count_transition,
    count_filter_statuses,
    get_excluded_names,
    get_included_names,
    get_undecided_names,
    load_name_inclusions_json,
    set_many_filter_statuses,
)
from st_name_ranking.interface.rankings_data import (
    ClusterProfileInputs,
    build_cluster_profiles,
    build_cluster_summary,
    build_global_predictor_rows,
    build_preference_percentage_dataframe,
    filter_ratings_for_names,
)
from st_name_ranking.interface.tournament_session import (
    get_current_pair,
    get_or_start_tournament_queue,
    get_queue_manager_stats,
    set_current_pair,
)
from st_name_ranking.persistence.database import (
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
from st_name_ranking.tournament_orchestration import prepare_tournament_round, record_tournament_vote

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
MIN_NAMES_FOR_COMPARISON = 2
MIN_NAMES_FOR_LANDSCAPE = 25
MIN_NON_NOISE_CLUSTERS = 2
FAST_REFILL_THRESHOLD_MS = 120
MODERATE_REFILL_THRESHOLD_MS = 300
SLOW_RENDER_THRESHOLD_MS = 100
MS_PER_SECOND = 1000
FILTER_SAVE_INTERVAL = 50
MAX_EXCLUDED_NAMES_DISPLAY = 100


@dataclass(frozen=True)
class RenderTimer:
    """Small timing helper for Streamlit fragments."""

    label: str
    start_time: float

    @classmethod
    def start(cls, label: str) -> "RenderTimer":
        return cls(label=label, start_time=time.perf_counter())

    def log(self, step: str) -> None:
        logger.debug("%s [%s]: %.2fms", self.label, step, (time.perf_counter() - self.start_time) * MS_PER_SECOND)


@dataclass(frozen=True)
class FilterRenderContext:
    names: list[str]
    inclusions: dict[str, bool]
    undecided_names: list[str]
    progress_placeholder: Any
    stats_placeholder: Any
    name_display_placeholder: Any


@dataclass(frozen=True)
class TournamentVoteClicks:
    prefer_a: bool
    prefer_b: bool
    draw: bool
    dislike_both: bool


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
        df_gender = build_preference_percentage_dataframe(gender_stats)
        create_stacked_bar_chart(df_gender, "Gender Preferences")
    else:
        st.info("No gender preference data available.")

    # Origin preferences
    if origin_stats:
        df_origin = build_preference_percentage_dataframe(origin_stats)
        create_stacked_bar_chart(df_origin, "Origin Preferences")
    else:
        st.info("No origin preference data available.")

    # Phonetic preferences
    if phonetic_stats:
        df_phonetic = build_preference_percentage_dataframe(phonetic_stats)
        create_stacked_bar_chart(df_phonetic, "Phonetic Preferences")
    else:
        st.info("No phonetic preference data available.")


def _record_vote_and_get_next_pair(names: list[str], manager: Any, preference: int) -> tuple[str, str]:
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


def _render_tournament_queue_caption(queue_stats: Mapping[str, object] | None, manager: Any) -> None:
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
    manager: Any,
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

    # Create placeholders for dynamic content
    pair_display_placeholder = st.empty()
    st.empty()

    _inject_tournament_styles()
    vote_action = _selected_tournament_vote_action(_render_tournament_vote_buttons())

    timer.log("After button creation")

    # Update session state and UI instantly after vote (no rerun needed)
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


@st.cache_data(show_spinner=False)
def _build_rankings_dataframe(
    ratings_pairs: tuple[tuple[str, float], ...],
    *,
    include_gender_male: bool,
) -> tuple[pl.DataFrame, list[str]]:
    sorted_pairs = sorted(ratings_pairs, key=lambda item: item[1], reverse=True)
    ordered_names = [name for name, _ in sorted_pairs]
    ordered_ratings = [rating for _, rating in sorted_pairs]
    base_df = pl.DataFrame({"Name": ordered_names, "Rating": ordered_ratings})

    try:
        model = get_or_initialize_active_learning_model()
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
def _compute_landscape(
    sorted_names: tuple[str, ...],
    ratings_pairs: tuple[tuple[str, float], ...],
    random_state: int,
) -> tuple[pl.DataFrame, np.ndarray, list[str], str]:
    ratings_dict = dict(ratings_pairs)
    model = get_or_initialize_active_learning_model()
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


def _rating_column_config() -> dict[str, Any]:
    return {
        "Rating": st.column_config.NumberColumn(
            "Rating",
            help="Higher is better",
            format="%d",
            pinned=True,
            width="small",
        ),
    }


def _render_rankings_table(df: pl.DataFrame) -> None:
    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        column_config=_rating_column_config(),
    )


def _render_overall_rankings(filtered_ratings: dict[str, float]) -> None:
    overall_pairs = tuple(filtered_ratings.items())
    df, feature_columns = _build_rankings_dataframe(overall_pairs, include_gender_male=True)
    _render_rankings_table(df)
    if feature_columns:
        st.caption(f"Feature columns shown: {', '.join(feature_columns)}")


def _render_preference_landscape(filtered_ratings: dict[str, float]) -> None:
    st.divider()
    st.subheader("Preference Landscape")

    if len(filtered_ratings) < MIN_NAMES_FOR_LANDSCAPE:
        st.info(f"Preference landscape appears after at least {MIN_NAMES_FOR_LANDSCAPE} rated names.")
        return

    try:
        random_state = st.slider(
            "Projection seed",
            min_value=0,
            max_value=99,
            value=42,
            key="rankings_projection_seed",
        )

        sorted_names = tuple(sorted(filtered_ratings))
        ratings_pairs = tuple(sorted(filtered_ratings.items()))
        with st.status("Building preference landscape...", expanded=False) as status:
            status.write("Projecting names with PaCMAP")
            status.write("Clustering projection with HDBSCAN")
            landscape_df, feature_matrix, feature_names, status_note = _compute_landscape(
                sorted_names,
                ratings_pairs,
                random_state,
            )
            status.update(label="Preference landscape ready", state="complete")

        st.caption(status_note)
        _render_landscape_chart(landscape_df)
        _render_global_predictors(feature_names)
        summary_df = _render_cluster_summary(landscape_df)
        model = get_or_initialize_active_learning_model()
        _render_cluster_profiles(
            ClusterProfileInputs(
                landscape_df=landscape_df,
                summary_df=summary_df,
                sorted_names=sorted_names,
                feature_matrix=feature_matrix,
                feature_names=feature_names,
                feature_weights=model.state.weight_mean,
            ),
        )
    except (RuntimeError, ValueError, ImportError) as err:
        logger.exception("Failed to render preference landscape")
        st.info(f"Preference landscape is temporarily unavailable: {err}")


def _render_landscape_chart(landscape_df: pl.DataFrame) -> None:
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


def _render_global_predictors(feature_names: list[str]) -> None:
    model = get_or_initialize_active_learning_model()
    global_rows = build_global_predictor_rows(feature_names, model.state.weight_mean)

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


def _render_cluster_summary(landscape_df: pl.DataFrame) -> pl.DataFrame:
    summary_df = build_cluster_summary(landscape_df)

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
    return summary_df


def _render_cluster_profiles(inputs: ClusterProfileInputs) -> None:
    cluster_profiles = build_cluster_profiles(inputs)
    st.markdown("**Cluster profiles**")
    st.dataframe(
        pl.DataFrame(cluster_profiles).sort("Cluster"),
        hide_index=True,
        width="stretch",
    )


def _render_gender_rankings(
    label: str,
    gender_names: list[str],
    names: list[str],
) -> None:
    if not gender_names:
        st.info(f"No gender data available for {label.lower()} names.")
        return

    gender_ratings = filter_ratings_for_names(st.session_state.ratings, names, allowed_names=gender_names)
    if not gender_ratings:
        st.info(f"No {label.lower()} names rated yet.")
        return

    df, _feature_columns = _build_rankings_dataframe(
        tuple(gender_ratings.items()),
        include_gender_male=False,
    )
    _render_rankings_table(df)


def render_rankings(names: list[str]) -> None:
    """Render rankings view showing top rated names.

    This is a separate tab to avoid slowing down the tournament UI.
    Only renders when the Rankings tab is active.
    """
    logger.debug("Rendering rankings for %d names", len(names))
    st.header("Name Rankings")
    st.write(f"Showing ratings for {len(names)} names")

    if len(names) == 0:
        st.info("No names to rank. Please include some names in the Name Filter tab first.")
        return

    # Get gender-specific name lists if available
    male_names = []
    female_names = []
    if "all_names_data" in st.session_state:
        gender_data = st.session_state.all_names_data
        male_names = gender_data.get("Male", [])
        female_names = gender_data.get("Female", [])

    # Filter ratings to only show current selection
    filtered_ratings = filter_ratings_for_names(st.session_state.ratings, names)

    if not filtered_ratings:
        st.info("No ratings yet. Start comparing names in the Tournament tab to generate rankings.")
        return

    # Create tabs for different views
    tab_overall, tab_male, tab_female = st.tabs(["Overall", "Male", "Female"])

    with tab_overall:
        _render_overall_rankings(filtered_ratings)
        _render_preference_landscape(filtered_ratings)

    with tab_male:
        _render_gender_rankings("Male", male_names, names)

    with tab_female:
        _render_gender_rankings("Female", female_names, names)


def _load_cached_name_inclusions() -> dict[str, bool]:
    cache_key = "name_inclusions_loaded"
    if cache_key in st.session_state:
        return st.session_state.name_inclusions

    try:
        inclusions_json = load_user_setting("name_inclusions", "{}")
        st.session_state.name_inclusions = load_name_inclusions_json(inclusions_json)
        logger.debug("Loaded %d inclusions from database", len(st.session_state.name_inclusions))
    except TypeError:
        st.session_state.name_inclusions = {}
    st.session_state[cache_key] = True
    return st.session_state.name_inclusions


def _persist_name_inclusions(inclusions: dict[str, bool]) -> None:
    _persist_name_inclusions_json(json.dumps(inclusions))


def _persist_name_inclusions_json(inclusions_json: str) -> None:
    save_user_setting("name_inclusions", inclusions_json)


def _current_filter_counts() -> FilterCounts:
    return FilterCounts(
        not_decided=st.session_state.filter_counts_not_decided,
        included=st.session_state.filter_counts_included,
        excluded=st.session_state.filter_counts_excluded,
    )


def _store_filter_counts(counts: FilterCounts, names_hash: str | None = None) -> None:
    st.session_state.filter_counts_not_decided = counts.not_decided
    st.session_state.filter_counts_included = counts.included
    st.session_state.filter_counts_excluded = counts.excluded
    if names_hash is not None:
        st.session_state.filter_counts_names_hash = names_hash


def _update_filter_counts(*, old_status: bool | None, new_status: bool | None) -> None:
    _store_filter_counts(
        apply_filter_count_transition(
            _current_filter_counts(),
            old_status=old_status,
            new_status=new_status,
        ),
    )


def _clear_filter_count_cache() -> None:
    if "filter_counts_names_hash" in st.session_state:
        del st.session_state.filter_counts_names_hash


def _names_filter_hash(names: list[str]) -> str:
    fast_hash = hash((names[0], names[-1], len(names))) if names else hash(0)
    return str(fast_hash)


def _ensure_filter_counts(names: list[str], inclusions: dict[str, bool], names_hash: str) -> None:
    needs_recount = (
        "filter_counts_not_decided" not in st.session_state
        or "filter_counts_included" not in st.session_state
        or "filter_counts_excluded" not in st.session_state
        or st.session_state.get("filter_counts_names_hash") != names_hash
    )
    if not needs_recount:
        return

    logger.debug("Computing filter counts for %d names", len(names))
    count_loop_start = time.perf_counter()
    counts = count_filter_statuses(names, inclusions)
    _store_filter_counts(counts, names_hash)

    logger.debug(
        "Filter counts computed: %d not decided, %d included, %d excluded (%.1fms)",
        counts.not_decided,
        counts.included,
        counts.excluded,
        (time.perf_counter() - count_loop_start) * MS_PER_SECOND,
    )


def _sync_filter_session(names_hash: str) -> None:
    if "filter_names_hash" not in st.session_state or st.session_state.filter_names_hash != names_hash:
        st.session_state.filter_names_hash = names_hash
        st.session_state.filter_index = 0

    if "filter_index" not in st.session_state:
        st.session_state.filter_index = 0


def _current_filter_selection(undecided_names: list[str]) -> tuple[str, int]:
    current_idx = st.session_state.filter_index
    if current_idx >= len(undecided_names):
        current_idx = 0
        st.session_state.filter_index = 0
    return undecided_names[current_idx], current_idx


def _clamp_filter_index(names: list[str]) -> None:
    if st.session_state.filter_index >= len(names):
        st.session_state.filter_index = 0


def _render_filter_name_display(context: FilterRenderContext, current_name: str, current_idx: int) -> None:
    progress = current_idx / len(context.undecided_names)
    context.progress_placeholder.progress(
        progress,
        text=f"Progress: {current_idx + 1} of {len(context.undecided_names)} remaining",
    )

    not_decided = st.session_state.filter_counts_not_decided
    explicitly_included = st.session_state.filter_counts_included
    explicitly_excluded = st.session_state.filter_counts_excluded
    context.stats_placeholder.caption(
        f"Not decided: {not_decided} | Included: {explicitly_included} | Excluded: {explicitly_excluded}",
    )

    if current_name not in context.inclusions:
        border_color = "#757575"
        status_text = "Not decided"
        bg_color = "#FAFAFA"
    elif context.inclusions[current_name]:
        border_color = "#4CAF50"
        status_text = "Included"
        bg_color = "#E8F5E9"
    else:
        border_color = "#F44336"
        status_text = "Excluded"
        bg_color = "#FFEBEE"

    context.name_display_placeholder.markdown(
        f"<div style='border: 4px solid {border_color}; background-color: {bg_color}; "
        f"border-radius: 12px; padding: 20px; text-align: center;'>"
        f"<h1 style='font-size: 72px; margin: 0; color: #212121;'>{current_name}</h1>"
        f"<p style='font-size: 16px; margin: 10px 0 0 0; color: {border_color}; "
        f"font-weight: bold;'>{status_text}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _display_next_undecided_name(context: FilterRenderContext, current_name: str) -> None:
    if current_name in context.undecided_names:
        context.undecided_names.remove(current_name)

    if st.session_state.filter_index >= len(context.undecided_names):
        st.session_state.filter_index = 0

    next_idx = st.session_state.filter_index
    if context.undecided_names:
        _render_filter_name_display(context, context.undecided_names[next_idx], next_idx)
    else:
        st.success("✅ All names processed! Switch to Tournament tab.")


def _apply_filter_decision(
    context: FilterRenderContext,
    current_name: str,
    *,
    status: bool,
    label: str,
    icon: str,
) -> None:
    logger.info("%s %s: %s", icon, label, current_name)
    button_click_start = time.perf_counter()

    old_status = context.inclusions.get(current_name)
    context.inclusions[current_name] = status
    _update_filter_counts(old_status=old_status, new_status=status)
    st.toast(f"{label}: {current_name}", icon=icon)
    st.session_state.last_button_press_time = time.perf_counter()
    _persist_name_inclusions(context.inclusions)
    _display_next_undecided_name(context, current_name)

    logger.debug("%s handled in %.1fms", label, (time.perf_counter() - button_click_start) * MS_PER_SECOND)


def _move_filter_name_to_undecided(context: FilterRenderContext, name: str, source_label: str) -> None:
    old_status = context.inclusions.get(name)
    if old_status is None:
        return

    del context.inclusions[name]
    _update_filter_counts(old_status=old_status, new_status=None)
    logger.info("🔄 %s moved from %s to not decided", name, source_label)
    st.toast(f"{name} moved to not decided", icon="🔄")
    _persist_name_inclusions(context.inclusions)
    st.rerun(scope="fragment")


def _render_filter_decision_buttons(context: FilterRenderContext, current_name: str) -> None:
    col_exclude, col_include = st.columns(2)
    with col_exclude:
        if st.button(
            "Exclude",
            key="exclude_btn",
            width="stretch",
            type="secondary",
            shortcut="Left",
        ):
            _apply_filter_decision(context, current_name, status=False, label="Excluded", icon="👎")
    with col_include:
        if st.button(
            "Include",
            key="include_btn",
            width="stretch",
            type="primary",
            shortcut="Right",
        ):
            _apply_filter_decision(context, current_name, status=True, label="Included", icon="👍")


def _save_filter_periodically(context: FilterRenderContext, current_idx: int) -> None:
    if current_idx % FILTER_SAVE_INTERVAL != 0:
        return

    json_start = time.perf_counter()
    inclusions_json = json.dumps(context.inclusions)
    json_time = (time.perf_counter() - json_start) * MS_PER_SECOND

    save_start = time.perf_counter()
    _persist_name_inclusions_json(inclusions_json)
    save_time = (time.perf_counter() - save_start) * MS_PER_SECOND

    logger.debug("Periodic save: JSON=%.1fms, DB=%.1fms, entries=%d", json_time, save_time, len(context.inclusions))


def _render_filter_batch_buttons(context: FilterRenderContext, current_idx: int) -> None:
    col_batch1, col_batch2 = st.columns(2)
    with col_batch1:
        if st.button("Include All Remaining", type="secondary", help="Include all remaining undecided names"):
            count = set_many_filter_statuses(context.inclusions, context.undecided_names[current_idx:], status=True)
            st.session_state.filter_index = 0  # Reset since undecided list will be empty
            _clear_filter_count_cache()
            st.toast(f"Included {count} remaining names", icon="✅")
            st.session_state.last_button_press_time = time.perf_counter()
            _persist_name_inclusions(context.inclusions)
            st.rerun(scope="fragment")
    with col_batch2:
        if st.button("Exclude All Remaining", type="secondary", help="Exclude all remaining undecided names"):
            count = set_many_filter_statuses(context.inclusions, context.undecided_names[current_idx:], status=False)
            st.session_state.filter_index = 0  # Reset since undecided list will be empty
            _clear_filter_count_cache()
            st.toast(f"Excluded {count} remaining names", icon="✅")
            st.session_state.last_button_press_time = time.perf_counter()
            _persist_name_inclusions(context.inclusions)
            st.rerun(scope="fragment")


def _render_filter_selection_editor(context: FilterRenderContext) -> None:
    included_names = get_included_names(context.names, context.inclusions)
    excluded_count = st.session_state.filter_counts_excluded

    st.divider()
    st.subheader("📋 Your Selections")

    if included_names:
        sorted_included = sorted(included_names)
        selected_included = st.multiselect(
            f"✅ Included names for tournament ({len(sorted_included)})",
            options=sorted_included,
            default=sorted_included,
            help="Uncheck names to move them back to 'not decided'",
        )

        selected_set = set(selected_included)
        for name in sorted_included:
            if name not in selected_set:
                _move_filter_name_to_undecided(context, name, "included")
    else:
        st.info("No names included yet. Use 'Include' button above to add names.")

    if excluded_count > 0:
        with st.expander(f"❌ Show Excluded Names ({excluded_count})"):
            excluded_names = get_excluded_names(context.names, context.inclusions)
            if excluded_names:
                sorted_excluded = sorted(excluded_names)[:MAX_EXCLUDED_NAMES_DISPLAY]
                remaining = len(excluded_names) - MAX_EXCLUDED_NAMES_DISPLAY

                if remaining > 0:
                    st.caption(f"Showing first {MAX_EXCLUDED_NAMES_DISPLAY} of {len(excluded_names)} excluded names")

                selected_excluded = st.multiselect(
                    "Uncheck names to include them again",
                    options=sorted_excluded,
                    default=sorted_excluded,
                    help="Uncheck names to move them back to 'not decided'",
                )

                selected_set = set(selected_excluded)
                for name in sorted_excluded:
                    if name not in selected_set:
                        _move_filter_name_to_undecided(context, name, "excluded")


@st.fragment
def render_binary_filter(names: list[str]) -> None:
    """Render binary filter interface for including/excluding names.

    Users review names one by one, marking them as included or excluded.
    """
    logger.debug("Filter render started with %d names", len(names))
    timer = RenderTimer.start("Filter")

    if "last_button_press_time" in st.session_state:
        del st.session_state.last_button_press_time

    st.header("Name Filter")
    st.write(
        "Review names one by one. Include names you want to compare in the tournament, "
        "exclude names you don't care about.",
    )
    st.caption(
        "💡 **Keyboard shortcuts**: Left arrow (←) to exclude, Right arrow (→) to include, Space to include",
    )

    progress_placeholder = st.empty()
    stats_placeholder = st.empty()
    name_display_placeholder = st.empty()

    inclusions = _load_cached_name_inclusions()
    timer.log("After inclusions loaded")

    names_hash = _names_filter_hash(names)
    _sync_filter_session(names_hash)
    _clamp_filter_index(names)
    _ensure_filter_counts(names, inclusions, names_hash)
    timer.log("After counts")

    undecided_names = get_undecided_names(names, inclusions)
    if not undecided_names:
        st.success("✅ All names have been processed! Switch to the Tournament tab to compare your selected names.")
        return

    context = FilterRenderContext(
        names=names,
        inclusions=inclusions,
        undecided_names=undecided_names,
        progress_placeholder=progress_placeholder,
        stats_placeholder=stats_placeholder,
        name_display_placeholder=name_display_placeholder,
    )
    current_name, current_idx = _current_filter_selection(undecided_names)

    _render_filter_name_display(context, current_name, current_idx)
    timer.log("After display update")

    _render_filter_decision_buttons(context, current_name)
    _save_filter_periodically(context, current_idx)
    _render_filter_batch_buttons(context, current_idx)
    timer.log("After buttons")

    _render_filter_selection_editor(context)

    timer.log("At end")

    end_time = time.perf_counter()
    elapsed_ms = (end_time - timer.start_time) * MS_PER_SECOND

    if elapsed_ms > SLOW_RENDER_THRESHOLD_MS:
        logger.info("Filter render slow: %.1fms", elapsed_ms)
    else:
        logger.debug("Filter render fast: %.1fms", elapsed_ms)


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
                try:
                    model = load_embedding_model()
                except RuntimeError as e:
                    st.warning(str(e))
                    return

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
