"""Compatibility exports for Streamlit UI renderers.

New code should import focused renderers from ``tournament_ui``, ``rankings_ui``,
``filter_ui``, or ``similarity_ui``.
"""

from st_name_ranking.interface.filter_ui import render_binary_filter
from st_name_ranking.interface.rankings_ui import render_preferences_panel, render_rankings
from st_name_ranking.interface.similarity_ui import render_similarity
from st_name_ranking.interface.tournament_ui import display_name_with_rating, render_tournament
from st_name_ranking.interface.ui_support import MS_PER_SECOND

__all__ = [
    "MS_PER_SECOND",
    "display_name_with_rating",
    "render_binary_filter",
    "render_preferences_panel",
    "render_rankings",
    "render_similarity",
    "render_tournament",
]
