"""Ranking and preference analytics rendering."""

import logging
from typing import Any

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
from st_name_ranking.interface.rankings_data import (
    ClusterProfileInputs,
    build_cluster_profiles,
    build_cluster_summary,
    build_global_predictor_rows,
    build_preference_percentage_dataframe,
    filter_ratings_for_names,
)
from st_name_ranking.interface.ui_support import MIN_NAMES_FOR_LANDSCAPE, MIN_NON_NOISE_CLUSTERS
from st_name_ranking.persistence.database import (
    INITIAL_SCORE,
    get_preference_stats_by_gender,
    get_preference_stats_by_origin,
    get_preference_stats_by_phonetic,
)

logger = logging.getLogger(__name__)

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


def render_preferences_panel() -> None:
    """Render panel showing overall preferences across different groups."""
    st.subheader("Overall Preferences")

    gender_stats = get_preference_stats_by_gender()
    origin_stats = get_preference_stats_by_origin()
    phonetic_stats = get_preference_stats_by_phonetic()

    def create_stacked_bar_chart(df: pl.DataFrame, title: str) -> None:
        """Create stacked bar chart showing win/loss/draw percentages."""
        if df.is_empty():
            return

        chart_df = df.select(["Group", "win_pct", "loss_pct", "draw_pct"])
        chart_df = chart_df.sort("win_pct", descending=True)
        st.subheader(title, divider="gray")

        st.bar_chart(
            chart_df,
            x="Group",
            y=["win_pct", "loss_pct", "draw_pct"],
            height=400,
            width="stretch",
            color=["#2E7D32", "#C62828", "#FF9800"],  # Green for wins, red for losses, orange for draws
        )

        display_df = df.sort("win_pct", descending=True)

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

        st.caption("🎯 **Legend**: 🟢 Wins | 🔴 Losses | 🟠 Draws")

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

    male_names = []
    female_names = []
    if "all_names_data" in st.session_state:
        gender_data = st.session_state.all_names_data
        male_names = gender_data.get("Male", [])
        female_names = gender_data.get("Female", [])

    filtered_ratings = filter_ratings_for_names(st.session_state.ratings, names)

    if not filtered_ratings:
        st.info("No ratings yet. Start comparing names in the Tournament tab to generate rankings.")
        return

    tab_overall, tab_male, tab_female = st.tabs(["Overall", "Male", "Female"])

    with tab_overall:
        _render_overall_rankings(filtered_ratings)
        _render_preference_landscape(filtered_ratings)

    with tab_male:
        _render_gender_rankings("Male", male_names, names)

    with tab_female:
        _render_gender_rankings("Female", female_names, names)
