"""Deprecated compatibility exports for historical utility imports."""

from st_name_ranking.active_learning.lazy_updates import record_comparison_instant
from st_name_ranking.active_learning.selection import (
    PairSelectionOptions,
    get_active_learning_model,
    get_feature_extractor,
    get_name_features,
    get_names_features,
    select_candidate_batch,
    select_candidates,
    select_random_batch,
    select_random_pair,
    try_select_candidates,
)
from st_name_ranking.interface.app_actions import (
    pull_submodule_updates,
    setup_session_state,
    sync_names_from_submodule,
)

__all__ = [
    "PairSelectionOptions",
    "get_active_learning_model",
    "get_feature_extractor",
    "get_name_features",
    "get_names_features",
    "pull_submodule_updates",
    "record_comparison_instant",
    "select_candidate_batch",
    "select_candidates",
    "select_random_batch",
    "select_random_pair",
    "setup_session_state",
    "sync_names_from_submodule",
    "try_select_candidates",
]
