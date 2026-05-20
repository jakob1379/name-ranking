"""Pure data builders for rankings and preference-landscape views."""

from dataclasses import dataclass

import numpy as np
import polars as pl

from st_name_ranking.types import PreferenceStats


@dataclass(frozen=True)
class ClusterProfileInputs:
    """Inputs needed to explain preference-landscape clusters."""

    landscape_df: pl.DataFrame
    summary_df: pl.DataFrame
    sorted_names: tuple[str, ...]
    feature_matrix: np.ndarray
    feature_names: list[str]
    feature_weights: np.ndarray


def build_preference_percentage_dataframe(stats: dict[str, PreferenceStats]) -> pl.DataFrame:
    """Convert preference stats into percentage rows for visualization."""
    rows = []
    for group, data in stats.items():
        wins = data.wins
        losses = data.losses
        draws = data.draws
        total = data.total

        rows.append(
            {
                "Group": group,
                "Wins": wins,
                "Losses": losses,
                "Draws": draws,
                "Total": total,
                "win_pct": wins / total * 100 if total > 0 else 0.0,
                "loss_pct": losses / total * 100 if total > 0 else 0.0,
                "draw_pct": draws / total * 100 if total > 0 else 0.0,
                "win_rate_pct": wins / (wins + losses) * 100 if wins + losses > 0 else 0.0,
            },
        )
    return pl.DataFrame(rows)


def filter_ratings_for_names(
    ratings: dict[str, float],
    names: list[str],
    *,
    allowed_names: list[str] | None = None,
) -> dict[str, float]:
    """Filter ratings to the active name set and optional gender/category set."""
    names_set = set(names)
    allowed_set = set(allowed_names) if allowed_names is not None else None
    return {
        name: rating
        for name, rating in ratings.items()
        if name in names_set and (allowed_set is None or name in allowed_set)
    }


def build_global_predictor_rows(
    feature_names: list[str],
    feature_weights: np.ndarray,
    *,
    limit: int = 8,
) -> list[dict[str, str | float]]:
    """Build display rows for the strongest global model predictors."""
    feature_rank = np.argsort(np.abs(feature_weights))[::-1]
    top_k = min(limit, len(feature_names))
    return [
        {
            "Feature": feature_names[idx],
            "Weight": float(feature_weights[idx]),
            "Direction": "Positive" if feature_weights[idx] >= 0 else "Negative",
            "Strength": float(abs(feature_weights[idx])),
        }
        for idx in feature_rank[:top_k]
    ]


def build_cluster_summary(landscape_df: pl.DataFrame) -> pl.DataFrame:
    """Summarize preference landscape clusters for display."""
    return (
        landscape_df.group_by("Cluster")
        .agg(
            pl.len().alias("Size"),
            pl.col("Rating").mean().alias("Avg Rating"),
            pl.col("Utility").mean().alias("Avg Utility"),
            pl.col("Uncertainty").mean().alias("Avg Uncertainty"),
        )
        .sort("Size", descending=True)
    )


def build_cluster_profiles(inputs: ClusterProfileInputs) -> list[dict[str, int | str]]:
    """Build short feature-contribution profiles for each landscape cluster."""
    name_to_index = {name: idx for idx, name in enumerate(inputs.sorted_names)}
    cluster_profiles = []
    for cluster_id in inputs.summary_df["Cluster"].to_list():
        cluster_names = inputs.landscape_df.filter(pl.col("Cluster") == cluster_id)["Name"].to_list()
        cluster_idx = [name_to_index[name] for name in cluster_names]
        cluster_features = inputs.feature_matrix[cluster_idx]

        contribution = cluster_features.mean(axis=0) * inputs.feature_weights
        rank_idx = np.argsort(np.abs(contribution))[::-1][:3]
        label_tokens = [
            f"{inputs.feature_names[idx]} ({'+' if contribution[idx] >= 0 else '-'}{abs(contribution[idx]):.3f})"
            for idx in rank_idx
        ]
        cluster_profiles.append(
            {
                "Cluster": cluster_id,
                "Profile": " | ".join(label_tokens),
            },
        )

    return cluster_profiles
