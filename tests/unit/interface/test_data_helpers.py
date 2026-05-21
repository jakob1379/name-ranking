"""Tests for pure interface data/state helpers."""

import numpy as np
import polars as pl

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
from st_name_ranking.types import PreferenceStats


def test_filter_state_counts_and_transitions() -> None:
    names = ["Anna", "Peter", "Maria", "Jens"]
    inclusions = {"Anna": True, "Peter": False}

    assert count_filter_statuses(names, inclusions) == FilterCounts(
        not_decided=2,
        included=1,
        excluded=1,
    )
    assert get_included_names(names, inclusions) == ["Anna"]
    assert get_excluded_names(names, inclusions) == ["Peter"]
    assert get_undecided_names(names, inclusions) == ["Maria", "Jens"]

    counts = apply_filter_count_transition(
        FilterCounts(not_decided=2, included=1, excluded=1),
        old_status=None,
        new_status=True,
    )
    assert counts == FilterCounts(not_decided=1, included=2, excluded=1)

    changed = set_many_filter_statuses(inclusions, ["Maria", "Jens"], status=True)
    assert changed == 2
    assert inclusions == {"Anna": True, "Peter": False, "Maria": True, "Jens": True}


def test_load_name_inclusions_json_accepts_only_str_bool_maps() -> None:
    assert load_name_inclusions_json('{"Anna": true, "Peter": false}') == {
        "Anna": True,
        "Peter": False,
    }

    assert load_name_inclusions_json('["Anna", true]') == {}
    assert load_name_inclusions_json('{"Anna": "true"}') == {}
    assert load_name_inclusions_json('{"Anna": true, "Peter": null}') == {}
    assert load_name_inclusions_json("{invalid json") == {}


def test_preference_percentage_dataframe() -> None:
    df = build_preference_percentage_dataframe(
        {
            "Female": PreferenceStats(wins=3, losses=1, draws=1, total=5),
            "Male": PreferenceStats(wins=0, losses=0, draws=0, total=0),
        },
    ).sort("Group")

    female = df.row(0, named=True)
    assert female["Group"] == "Female"
    assert female["win_pct"] == 60.0
    assert female["loss_pct"] == 20.0
    assert female["draw_pct"] == 20.0
    assert female["win_rate_pct"] == 75.0

    male = df.row(1, named=True)
    assert male["win_pct"] == 0.0
    assert male["win_rate_pct"] == 0.0


def test_rankings_filter_and_predictor_rows() -> None:
    ratings = {"Anna": 1600.0, "Peter": 1400.0, "Maria": 1550.0}

    assert filter_ratings_for_names(ratings, ["Anna", "Maria"]) == {
        "Anna": 1600.0,
        "Maria": 1550.0,
    }
    assert filter_ratings_for_names(ratings, ["Anna", "Maria"], allowed_names=["Anna"]) == {
        "Anna": 1600.0,
    }

    rows = build_global_predictor_rows(
        ["length", "gender_female", "origin_nordic"],
        np.array([0.2, -0.8, 0.5]),
        limit=2,
    )
    assert [row["Feature"] for row in rows] == ["gender_female", "origin_nordic"]
    assert rows[0]["Direction"] == "Negative"


def test_cluster_summary_and_profiles() -> None:
    landscape_df = pl.DataFrame(
        {
            "Name": ["Anna", "Peter", "Maria"],
            "Cluster": [0, 0, 1],
            "Rating": [1600.0, 1400.0, 1550.0],
            "Utility": [0.3, -0.1, 0.2],
            "Uncertainty": [0.2, 0.4, 0.3],
        },
    )
    summary = build_cluster_summary(landscape_df)

    assert summary["Size"].to_list() == [2, 1]

    profiles = build_cluster_profiles(
        ClusterProfileInputs(
            landscape_df=landscape_df,
            summary_df=summary,
            sorted_names=("Anna", "Peter", "Maria"),
            feature_matrix=np.array([[1.0, 0.0], [0.5, 1.0], [0.0, 1.0]]),
            feature_names=["soft", "sharp"],
            feature_weights=np.array([0.6, -0.4]),
        ),
    )

    assert profiles[0]["Cluster"] == 0
    assert "soft" in profiles[0]["Profile"]
