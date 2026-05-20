"""Deprecated compatibility exports for historical utility imports."""

import warnings

from st_name_ranking.active_learning.lazy_updates import record_comparison_instant
from st_name_ranking.active_learning.selection import (
    PairSelectionOptions,
    select_candidate_batch,
    select_candidates,
)
from st_name_ranking.interface.app_actions import (
    pull_submodule_updates,
    setup_session_state,
    sync_names_from_submodule,
)

warnings.warn(
    "st_name_ranking.utils is deprecated; import active-learning and interface helpers "
    "from their canonical subpackages instead. This shim is planned for removal in 0.3.0.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "PairSelectionOptions",
    "pull_submodule_updates",
    "record_comparison_instant",
    "select_candidate_batch",
    "select_candidates",
    "setup_session_state",
    "sync_names_from_submodule",
]
