"""Name Ranker - Main entry point.
Refactored version with modular imports.
"""

import json
import logging
import secrets
import sqlite3
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import streamlit as st

from st_name_ranking.interface.app_actions import (
    setup_session_state,
    sync_names_from_submodule,
)
from st_name_ranking.interface.filter_state import load_name_inclusions_json
from st_name_ranking.interface.ui import render_binary_filter, render_rankings, render_similarity, render_tournament
from st_name_ranking.persistence import database
from st_name_ranking.persistence.data_loader import DataLoaderError, load_names_by_gender
from st_name_ranking.persistence.database import initialize_ratings

logger = logging.getLogger(__name__)
DEFAULT_TOURNAMENT_SAMPLE_SIZE: int | None = None
logger.setLevel(logging.INFO)
TAB_NAME_FILTER = "Name Filter"
TAB_TOURNAMENT = "Tournament"
TAB_RANKINGS = "Rankings"
TAB_SIMILARITY = "Similarity Search"

# Configure logging - suppress debug noise
logging.getLogger("watchdog").setLevel(logging.WARNING)
logging.getLogger("sqlite3").setLevel(logging.WARNING)

# Only configure logging if no handlers are configured yet
# This prevents duplicate logs when Streamlit reloads the script
root_logger = logging.getLogger()
if not root_logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


@dataclass(frozen=True)
class ActiveTabRender:
    renderer: str
    names: list[str]


@st.dialog("⚠️ Confirm Reset Ratings", width="small")
def show_reset_ratings_dialog() -> None:
    """Dialog for confirming ratings reset."""
    st.write("This will reset **all** ratings to their initial values.")
    st.write("**This action cannot be undone.**")

    col_confirm, col_cancel = st.columns(2)
    with col_confirm:
        if st.button("Yes, Reset Ratings", type="primary", width="stretch"):
            st.session_state.ratings = initialize_ratings(st.session_state.names)
            st.toast("✅ Ratings reset to initial values", icon="✅")
            st.rerun()
    with col_cancel:
        if st.button("Cancel", type="secondary", width="stretch"):
            st.rerun()


@st.dialog("⚠️ Confirm Reset Excluded", width="small")
def show_reset_excluded_dialog() -> None:
    """Dialog for confirming excluded names reset."""
    st.write("This will move all **excluded** names back to 'not decided'.")
    st.write("**This action cannot be undone.**")

    col_confirm, col_cancel = st.columns(2)
    with col_confirm:
        if st.button("Yes, Reset Excluded", type="primary", width="stretch"):
            # Only remove excluded entries (False values), keep included
            inclusions = st.session_state.get("name_inclusions", {})
            st.session_state.name_inclusions = {k: v for k, v in inclusions.items() if v is True}
            st.session_state.filter_index = 0
            # Reset counts
            for key in [
                "filter_counts_not_decided",
                "filter_counts_included",
                "filter_counts_excluded",
                "filter_counts_names_hash",
            ]:
                if key in st.session_state:
                    del st.session_state[key]
            database.save_user_setting("name_inclusions", json.dumps(st.session_state.name_inclusions))
            st.toast("Excluded names reset", icon="🔄")
            st.rerun()
    with col_cancel:
        if st.button("Cancel", type="secondary", width="stretch"):
            st.rerun()


@st.dialog("⚠️ Confirm Reset Included", width="small")
def show_reset_included_dialog() -> None:
    """Dialog for confirming included names reset."""
    st.write("This will move all **included** names back to 'not decided'.")
    st.write("**This action cannot be undone.**")

    col_confirm, col_cancel = st.columns(2)
    with col_confirm:
        if st.button("Yes, Reset Included", type="primary", width="stretch"):
            # Only remove included entries (True values), keep excluded
            inclusions = st.session_state.get("name_inclusions", {})
            st.session_state.name_inclusions = {k: v for k, v in inclusions.items() if v is False}
            st.session_state.filter_index = 0
            # Reset counts
            for key in [
                "filter_counts_not_decided",
                "filter_counts_included",
                "filter_counts_excluded",
                "filter_counts_names_hash",
            ]:
                if key in st.session_state:
                    del st.session_state[key]
            database.save_user_setting("name_inclusions", json.dumps(st.session_state.name_inclusions))
            st.toast("Included names reset", icon="🔄")
            st.rerun()
    with col_cancel:
        if st.button("Cancel", type="secondary", width="stretch"):
            st.rerun()


def render_sidebar_database_controls() -> None:
    st.subheader("Database Management")

    database.init_database()
    try:
        stats = database.get_stats()
        total_names = _stats_total_names(stats)
        try:
            total_comparisons = int(database.get_total_comparisons())
        except (TypeError, ValueError):
            total_comparisons = 0

        st.metric(
            "Names in Database",
            f"{total_names:,}",
            f"{total_comparisons:,} comparisons",
            border=True,
        )
    except sqlite3.Error:
        st.caption("Database stats unavailable")

    if st.button(
        "Sync Names",
        icon="🔄",
        help="Sync names from submodule to database",
        width="stretch",
    ):
        inserted = sync_names_from_submodule()
        if inserted > 0:
            st.rerun()


def ensure_names_loaded() -> None:
    if "all_names_data" in st.session_state:
        return

    try:
        with st.spinner("Loading names from submodule..."):
            gender_data = load_names_by_gender(sync_with_submodule=False)
    except DataLoaderError as e:
        st.toast(
            f"Failed to load names from database: {e}",
            icon="❌",
            duration="long",
        )
        return

    if gender_data and "All" in gender_data:
        st.session_state.all_names_data = gender_data
        st.session_state.all_names = gender_data["All"]
        setup_session_state(gender_data["All"])

        st.toast(
            f"Loaded {len(gender_data['All'])} total names",
            icon="✅",
        )

        for gender in ["Male", "Female", "Unisex"]:
            if gender in gender_data:
                st.toast(
                    f"{gender}: {len(gender_data[gender])} names",
                    icon="ℹ️",
                )
        return

    st.toast(
        "Failed to load names from submodule",
        icon="❌",
        duration="long",
    )


def render_sidebar_filter_controls() -> None:
    st.subheader("Filtering")

    if "gender_filter" not in st.session_state:
        st.session_state.gender_filter = secrets.choice(["Male", "Female"])

    gender_option = st.pills(
        "Gender:",
        ["All", "Male", "Female"],
        default=st.session_state.gender_filter,
        help=("Select which gender of names to compare. Click or use left/right arrow keys to navigate."),
    )

    if gender_option != st.session_state.gender_filter:
        st.session_state.gender_filter = gender_option
        st.toast(
            f"Filter set to: {gender_option}",
            icon="ℹ️",
        )
        st.rerun()

    database.init_database()
    available_regions = database.get_all_origin_regions()

    if "origin_filter" not in st.session_state:
        saved_origins_json = database.load_user_setting(
            "selected_origins",
            "[]",
        )
        try:
            saved_origins = json.loads(saved_origins_json)
            st.session_state.origin_filter = [origin for origin in saved_origins if origin in available_regions]
        except json.JSONDecodeError:
            st.session_state.origin_filter = []

    selected_origins = st.multiselect(
        "Origin regions:",
        options=available_regions,
        default=st.session_state.origin_filter,
        help="Select origin regions. Empty shows all.",
    )

    if selected_origins != st.session_state.origin_filter:
        st.session_state.origin_filter = selected_origins
        database.save_user_setting(
            "selected_origins",
            json.dumps(selected_origins),
        )
        st.toast(
            f"Filter: {selected_origins or 'All'}",
            icon="ℹ️",
        )
        st.rerun()


def render_sidebar_danger_zone() -> None:
    st.divider()
    st.subheader("⚠️ Danger Zone")

    if "names" in st.session_state and st.session_state.names:
        st.caption(f"Active Dataset: {len(st.session_state.names)} names")

    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "Reset\nExcluded",
            type="secondary",
            help="Move all excluded names back to 'not decided'",
            width="stretch",
            key="reset_excluded_btn",
        ):
            show_reset_excluded_dialog()

    with col2:
        if st.button(
            "Reset\nIncluded",
            type="secondary",
            help="Move all included names back to 'not decided'",
            width="stretch",
            key="reset_included_btn",
        ):
            show_reset_included_dialog()

    if st.button(
        "Reset Ratings",
        type="secondary",
        help="Reset all tournament ratings to initial values",
        width="stretch",
        key="reset_ratings_btn",
    ):
        show_reset_ratings_dialog()


def render_sidebar_export_controls() -> None:
    st.subheader("Export")
    if not st.button("Export Database", width="stretch"):
        return

    try:
        db_bytes = database.export_database()
        st.download_button(
            label="Download Database",
            data=db_bytes,
            file_name=f"name_ranker_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.db",
            mime="application/x-sqlite3",
        )
    except (OSError, RuntimeError) as e:
        st.error(f"Failed to export database: {e}")


def render_sidebar() -> None:
    with st.sidebar:
        render_sidebar_database_controls()
        st.divider()
        ensure_names_loaded()
        render_sidebar_filter_controls()
        render_sidebar_danger_zone()
        render_sidebar_export_controls()


def _stats_total_names(stats: object) -> int:
    if isinstance(stats, dict):
        return int(stats.get("total_names", 0))
    return int(getattr(stats, "total_names", 0))


def _render_missing_names_message() -> None:
    st.error("No names loaded in the database.")
    st.info(
        "Click **Sync Names** in the sidebar to load names from the submodule.",
    )

    database.init_database()
    try:
        stats = database.get_stats()
        total_names = _stats_total_names(stats)
        if total_names == 0:
            st.warning("Database is empty. You need to sync names first.")
        else:
            st.success(
                f"Database has {total_names} names. Try reloading the page.",
            )
    except sqlite3.Error:
        st.warning("Unable to read database statistics.")


def resolve_filtered_names() -> list[str] | None:
    current_gender = st.session_state.get("gender_filter", "All")
    current_origins = st.session_state.get("origin_filter", [])
    origins_to_filter = current_origins or None
    cache_key = f"filtered_names_{current_gender}_{tuple(sorted(current_origins)) if current_origins else 'all'}"

    if (
        "filtered_names_cache" in st.session_state
        and "filtered_cache_key" in st.session_state
        and st.session_state.filtered_cache_key == cache_key
    ):
        # Use cached filtered names
        filtered_names = st.session_state.filtered_names_cache
        logger.debug("Using cached filtered names (cache hit)")
    else:
        # Compute and cache
        start_time = time.perf_counter()
        database.init_database()
        db_time = time.perf_counter()
        filtered_names = database.get_names_by_filters(
            gender=current_gender if current_gender != "All" else None,
            origins=origins_to_filter,
        )
        filter_time = time.perf_counter()
        st.session_state.filtered_names_cache = filtered_names
        st.session_state.filtered_cache_key = cache_key
        total_time = (filter_time - start_time) * 1000
        db_init_time = (db_time - start_time) * 1000
        query_time = (filter_time - db_time) * 1000
        logger.debug(
            "Computed filtered names (cache miss): %d names, total=%.1fms, db_init=%.1fms, query=%.1fms",
            len(filtered_names),
            total_time,
            db_init_time,
            query_time,
        )

    if not filtered_names:
        if current_origins:
            st.toast(
                f"No names found for gender: {current_gender}, origins: {current_origins}",
                icon="⚠️",
            )
        else:
            st.toast(
                f"No names found for gender filter: {current_gender}",
                icon="⚠️",
            )
        return None

    total_names_count = len(st.session_state.all_names)
    if current_origins:
        st.toast(
            f"Showing {len(filtered_names)} names "
            f"(gender: {current_gender.lower()}, "
            f"origins: {', '.join(current_origins)}) "
            f"out of {total_names_count} total",
            icon="ℹ️",
        )
    else:
        st.toast(
            f"Showing {len(filtered_names)} {current_gender.lower()} names out of {total_names_count} total",
            icon="ℹ️",
        )
    return filtered_names


def _load_name_inclusions() -> dict[str, bool]:
    inclusions_json = database.load_user_setting("name_inclusions", "{}")
    return load_name_inclusions_json(inclusions_json)


def _get_included_names(filtered_names: list[str]) -> list[str]:
    inclusions = _load_name_inclusions()
    return [
        name
        for name in filtered_names
        if inclusions.get(name, True)  # True if not in dict or value is True
    ]


def _build_sample_size_options(count: int) -> list[int]:
    if count <= 0:
        return []
    options = [50, 100, 500, 1000]
    options.extend(range(2000, count + 1, 1000))
    valid_options = sorted({value for value in options if value <= count})
    if count not in valid_options:
        valid_options.append(count)
    return sorted(valid_options)


def _resolve_sample_size(count: int, candidate: int | None) -> int:
    options = _build_sample_size_options(count)
    if not options:
        return 0
    if candidate in options:
        return candidate
    return options[-1]


def resolve_tournament_sample_size(filtered_count: int) -> int:
    if "tournament_sample_size" not in st.session_state:
        stored_sample_size = database.load_user_setting(
            "tournament_sample_size",
            str(DEFAULT_TOURNAMENT_SAMPLE_SIZE),
        )
        try:
            parsed_sample_size = int(stored_sample_size)
        except (TypeError, ValueError):
            parsed_sample_size = DEFAULT_TOURNAMENT_SAMPLE_SIZE
        st.session_state.tournament_sample_size = _resolve_sample_size(filtered_count, parsed_sample_size)

    sample_size_options = _build_sample_size_options(filtered_count)
    selected_sample_size = _resolve_sample_size(
        filtered_count,
        st.session_state.get("tournament_sample_size"),
    )
    st.session_state.tournament_sample_size = selected_sample_size

    if sample_size_options:
        selected_sample_size = st.selectbox(
            "Tournament sample size",
            options=sample_size_options,
            index=sample_size_options.index(selected_sample_size),
            help="How many filtered names are sampled for tournament pair generation.",
        )
        if selected_sample_size != st.session_state.tournament_sample_size:
            st.session_state.tournament_sample_size = selected_sample_size
            database.save_user_setting("tournament_sample_size", str(selected_sample_size))
            st.rerun()
    return st.session_state.tournament_sample_size


def render_tab_selector() -> None:
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = TAB_NAME_FILTER

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button(
            "📋 Name Filter",
            width="stretch",
            type="primary" if st.session_state.active_tab == TAB_NAME_FILTER else "secondary",
        ):
            st.session_state.active_tab = TAB_NAME_FILTER
            st.rerun()
    with col2:
        if st.button(
            "🏆 Tournament",
            width="stretch",
            type="primary" if st.session_state.active_tab == TAB_TOURNAMENT else "secondary",
        ):
            st.session_state.active_tab = TAB_TOURNAMENT
            st.rerun()
    with col3:
        if st.button(
            "🏅 Rankings",
            width="stretch",
            type="primary" if st.session_state.active_tab == TAB_RANKINGS else "secondary",
        ):
            st.session_state.active_tab = TAB_RANKINGS
            st.rerun()
    with col4:
        if st.button(
            "🔍 Similarity Search",
            width="stretch",
            type="primary" if st.session_state.active_tab == TAB_SIMILARITY else "secondary",
        ):
            st.session_state.active_tab = TAB_SIMILARITY
            st.rerun()


def resolve_active_tab_render(
    active_tab: str,
    filtered_names: list[str],
    filtered_names_included: list[str],
) -> ActiveTabRender:
    if active_tab == TAB_NAME_FILTER:
        return ActiveTabRender("binary_filter", filtered_names)
    if active_tab == TAB_TOURNAMENT:
        return ActiveTabRender("tournament", filtered_names_included)
    if active_tab == TAB_RANKINGS:
        return ActiveTabRender("rankings", filtered_names_included)
    return ActiveTabRender("similarity", filtered_names_included)


def _render_active_tab(filtered_names: list[str], filtered_names_included: list[str]) -> None:
    render_tab_selector()
    st.divider()

    tab_render = resolve_active_tab_render(
        st.session_state.active_tab,
        filtered_names,
        filtered_names_included,
    )

    if tab_render.renderer == "binary_filter":
        render_binary_filter(tab_render.names)
    elif tab_render.renderer == "tournament":
        render_tournament(tab_render.names)
    elif tab_render.renderer == "rankings":
        render_rankings(tab_render.names)
    else:
        render_similarity(tab_render.names)


def main() -> None:
    start_time = time.perf_counter()
    st.set_page_config(page_title="Name Ranker", layout="wide")
    st.title("Name Preference Ranker")

    render_sidebar()

    if "all_names_data" not in st.session_state:
        _render_missing_names_message()
        return

    filtered_names = resolve_filtered_names()
    if not filtered_names:
        return

    filtered_names_included = _get_included_names(filtered_names)
    filtered_count = len(filtered_names_included)
    resolve_tournament_sample_size(filtered_count)
    st.session_state.tournament_filtered_count = filtered_count
    _render_active_tab(filtered_names, filtered_names_included)

    end_time = time.perf_counter()
    elapsed_ms = (end_time - start_time) * 1000
    logger.debug("main() execution time: %.1fms (active tab: %s)", elapsed_ms, st.session_state.active_tab)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError) as e:
        import traceback

        logger.exception("Fatal error in main")
        traceback.print_exc()
        # Try to show error in Streamlit if possible
        import sys

        if "streamlit" in sys.modules:
            import streamlit as st

            st.error(f"Fatal error: {e}")
            st.code(traceback.format_exc())
