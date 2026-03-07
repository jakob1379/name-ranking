"""Name Ranker - Main entry point.
Refactored version with modular imports.
"""

import json
import logging
import os
import secrets
import sqlite3
import time
from datetime import UTC, datetime

import streamlit as st

from st_name_ranking import database
from st_name_ranking.background_queue import get_queue_manager
from st_name_ranking.data_loader import load_names_by_gender
from st_name_ranking.database import initialize_ratings
from st_name_ranking.ui import render_binary_filter, render_rankings, render_similarity, render_tournament
from st_name_ranking.utils import (
    get_active_learning_model,
    setup_session_state,
    sync_names_from_submodule,
)

logger = logging.getLogger(__name__)
MIN_NAMES_FOR_TOURNAMENT = 2
DEFAULT_TOURNAMENT_SAMPLE_SIZE: int | None = None
logger.setLevel(logging.INFO)

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


def main() -> None:
    start_time = time.perf_counter()
    st.set_page_config(page_title="Name Ranker", layout="wide")
    st.title("Name Preference Ranker")

    # Data Loading - Only from submodule
    with st.sidebar:
        # Database Management - Always show even if names not loaded
        st.subheader("Database Management")

        # Database stats
        database.init_database()
        try:
            stats = database.get_stats()
            total_names = stats.total_names
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

        st.divider()

        # Auto-load from submodule on first run
        if "all_names_data" not in st.session_state:
            with st.spinner("Loading names from submodule..."):
                gender_data = load_names_by_gender(sync_with_submodule=False)

            if gender_data and "All" in gender_data:
                # Store the full dataset
                st.session_state.all_names_data = gender_data
                st.session_state.all_names = gender_data["All"]

                # Initialize with all names for ratings
                setup_session_state(gender_data["All"])

                # Pre-load the active learning model in background
                # so Tournament tab is ready instantly when clicked
                get_active_learning_model()

                st.toast(
                    f"Loaded {len(gender_data['All'])} total names",
                    icon="✅",
                )

                # Show breakdown
                for gender in ["Male", "Female", "Unisex"]:
                    if gender in gender_data:
                        st.toast(
                            f"{gender}: {len(gender_data[gender])} names",
                            icon="ℹ️",
                        )
            else:
                st.toast(
                    "Failed to load names from submodule",
                    icon="❌",
                    duration="long",
                )
                # Don't return - allow user to sync names from database management section

        # Filtering
        st.subheader("Filtering")

        # Gender selection
        if "gender_filter" not in st.session_state:
            # Random initial gender for demo purposes
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
            # When changing filter, we need to update the displayed names
            # but keep all ratings
            st.rerun()

        # Origin selection
        database.init_database()
        available_regions = database.get_all_origin_regions()

        if "origin_filter" not in st.session_state:
            # Load saved origin filter from database
            saved_origins_json = database.load_user_setting(
                "selected_origins",
                "[]",
            )
            try:
                saved_origins = json.loads(saved_origins_json)
                # Validate that saved origins are still available
                saved_origins = [o for o in saved_origins if o in available_regions]
                st.session_state.origin_filter = saved_origins
            except json.JSONDecodeError:
                # Default: empty list (show all regions)
                st.session_state.origin_filter = []

        selected_origins = st.multiselect(
            "Origin regions:",
            options=available_regions,
            default=st.session_state.origin_filter,
            help="Select origin regions. Empty shows all.",
        )

        # Save to session state and persist to database if changed
        if selected_origins != st.session_state.origin_filter:
            st.session_state.origin_filter = selected_origins
            # Save to database
            database.save_user_setting(
                "selected_origins",
                json.dumps(selected_origins),
            )
            st.toast(
                f"Filter: {selected_origins or 'All'}",
                icon="ℹ️",
            )
            st.rerun()

        # Danger Zone - destructive actions grouped together
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

        # Reset Ratings button (full width, below the two column buttons)
        if st.button(
            "Reset Ratings",
            type="secondary",
            help="Reset all tournament ratings to initial values",
            width="stretch",
            key="reset_ratings_btn",
        ):
            show_reset_ratings_dialog()

        # Export
        st.subheader("Export")
        if st.button("Export Database", width="stretch"):
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

    # Main Content
    if "all_names_data" not in st.session_state:
        st.error("No names loaded in the database.")
        st.info(
            "Click **Sync Names** in the sidebar to load names from the submodule.",
        )

        # Show database status
        database.init_database()
        try:
            stats = database.get_stats()
            total_names = stats.total_names
            if total_names == 0:
                st.warning("Database is empty. You need to sync names first.")
            else:
                st.success(
                    f"Database has {total_names} names. Try reloading the page.",
                )
        except sqlite3.Error:
            st.warning("Unable to read database statistics.")

        return

    # Get current gender filter
    current_gender = st.session_state.get("gender_filter", "All")
    # Get current origin filter
    current_origins = st.session_state.get("origin_filter", [])
    # Empty list means no origin filtering (show all regions)
    origins_to_filter = current_origins or None

    # Get filtered names using database with caching
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
        return

    # Get total names count for reference (all names in database)
    total_names_count = len(st.session_state.all_names)

    # Show filter info
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

    # Load inclusion decisions and apply filter
    inclusions_json = database.load_user_setting("name_inclusions", "{}")
    inclusions = {}
    try:
        loaded = json.loads(inclusions_json)
        if isinstance(loaded, dict):
            inclusions = loaded
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse inclusions JSON: %s", e)

    # Filter names: include names that are not explicitly excluded (default True)
    filtered_names_included = [
        name
        for name in filtered_names
        if inclusions.get(name, True)  # True if not in dict or value is True
    ]

    filtered_count = len(filtered_names_included)

    def build_sample_size_options(count: int) -> list[int]:
        if count <= 0:
            return []
        options = [50, 100, 500, 1000]
        options.extend(range(2000, count + 1, 1000))
        valid_options = sorted({value for value in options if value <= count})
        if count not in valid_options:
            valid_options.append(count)
        return sorted(valid_options)

    def resolve_sample_size(count: int, candidate: int | None) -> int:
        options = build_sample_size_options(count)
        if not options:
            return 0
        if candidate in options:
            return candidate
        return options[-1]

    if "tournament_sample_size" not in st.session_state:
        stored_sample_size = database.load_user_setting(
            "tournament_sample_size",
            str(DEFAULT_TOURNAMENT_SAMPLE_SIZE),
        )
        try:
            parsed_sample_size = int(stored_sample_size)
        except (TypeError, ValueError):
            parsed_sample_size = DEFAULT_TOURNAMENT_SAMPLE_SIZE
        st.session_state.tournament_sample_size = resolve_sample_size(filtered_count, parsed_sample_size)

    sample_size_options = build_sample_size_options(filtered_count)
    selected_sample_size = resolve_sample_size(
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

    tournament_names = filtered_names_included
    st.session_state.tournament_filtered_count = filtered_count

    # Initialize QueueManager ONLY when on Tournament tab
    # This avoids slowing down Name Filter with unnecessary background work
    if st.session_state.get("active_tab") == "Tournament" and len(tournament_names) >= MIN_NAMES_FOR_TOURNAMENT:
        # Queue size from environment variable (default 15)
        queue_size = int(os.environ.get("TOURNAMENT_QUEUE_SIZE", "15"))
        get_queue_manager(tournament_names, queue_size, sample_size=selected_sample_size)
        logger.debug("Initialized QueueManager for Tournament tab")
    else:
        logger.debug(
            "Skipping QueueManager init (tab=%s, names=%d)",
            st.session_state.get("active_tab"),
            len(tournament_names),
        )

    # Tab selection - only render active tab to improve performance
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Name Filter"

    # Display tab selector as radio buttons in columns for better UX
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button(
            "📋 Name Filter",
            width="stretch",
            type="primary" if st.session_state.active_tab == "Name Filter" else "secondary",
        ):
            st.session_state.active_tab = "Name Filter"
            st.rerun()
    with col2:
        if st.button(
            "🏆 Tournament",
            width="stretch",
            type="primary" if st.session_state.active_tab == "Tournament" else "secondary",
        ):
            st.session_state.active_tab = "Tournament"
            st.rerun()
    with col3:
        if st.button(
            "🏅 Rankings",
            width="stretch",
            type="primary" if st.session_state.active_tab == "Rankings" else "secondary",
        ):
            st.session_state.active_tab = "Rankings"
            st.rerun()
    with col4:
        if st.button(
            "🔍 Similarity Search",
            width="stretch",
            type="primary" if st.session_state.active_tab == "Similarity Search" else "secondary",
        ):
            st.session_state.active_tab = "Similarity Search"
            st.rerun()

    st.divider()

    # Render only the active tab
    if st.session_state.active_tab == "Name Filter":
        render_binary_filter(filtered_names)
    elif st.session_state.active_tab == "Tournament":
        render_tournament(tournament_names)
    elif st.session_state.active_tab == "Rankings":
        render_rankings(filtered_names_included)
    else:  # Similarity Search
        render_similarity(filtered_names_included)

    # Log total execution time
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
