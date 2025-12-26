"""
Name Ranker - Main entry point.
Refactored version with modular imports.
"""

import json
import logging
from datetime import datetime

import streamlit as st

from st_name_ranking import database
from st_name_ranking.data_loader import load_names_by_gender, save_ratings
from st_name_ranking.elo import initialize_ratings
from st_name_ranking.ui import render_similarity, render_tournament
from st_name_ranking.utils import setup_session_state

# Configure logging - suppress debug noise
logging.getLogger("watchdog").setLevel(logging.WARNING)
logging.getLogger("sqlite3").setLevel(logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def main() -> None:
    st.set_page_config(page_title="Name Ranker", layout="wide")
    st.title("Name Preference Ranker")

    # Data Loading - Only from submodule
    with st.sidebar:
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
                return


        # Gender Filtering
        st.subheader("Gender Filter")
        if "gender_filter" not in st.session_state:
            st.session_state.gender_filter = "All"

        # Use pills for gender selection - modern, tab-like appearance
        gender_option = st.pills(
            "Filter names by gender:",
            ["All", "Male", "Female", "Unisex"],
            default=st.session_state.gender_filter,
            help=(
                "Select which gender of names to compare. "
                "Click or use left/right arrow keys to navigate."
            ),
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

        # Origin Filtering
        st.subheader("Origin Filter")
        # Get available origin regions from database
        database.init_database()
        available_regions = database.get_all_origin_regions()

        if "origin_filter" not in st.session_state:
            # Load saved origin filter from database
            saved_origins_json = database.load_user_setting(
                "selected_origins", "[]"
            )
            try:
                saved_origins = json.loads(saved_origins_json)
                # Validate that saved origins are still available
                saved_origins = [
                    o for o in saved_origins if o in available_regions
                ]
                st.session_state.origin_filter = saved_origins
            except Exception:
                # Default: empty list (show all regions)
                st.session_state.origin_filter = []

        # Multiselect for origin filter
        selected_origins = st.multiselect(
            "Filter names by origin region:",
            options=available_regions,
            default=st.session_state.origin_filter,
            help="Select origin regions. Empty shows all.",
        )

        # Save to session state and persist to database if changed
        if selected_origins != st.session_state.origin_filter:
            st.session_state.origin_filter = selected_origins
            # Save to database
            database.save_user_setting(
                "selected_origins", json.dumps(selected_origins)
            )
            st.toast(
                f"Filter: {selected_origins if selected_origins else 'All'}",
                icon="ℹ️",
            )
            st.rerun()

        st.divider()

        # Ratings management
        st.subheader("Ratings Management")

        if "names" in st.session_state and st.session_state.names:
            st.caption(f"Active Dataset: {len(st.session_state.names)} names")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Save Ratings"):
                    if save_ratings(st.session_state.ratings):
                        st.toast(
                            "Ratings saved!",
                            icon="✅",
                        )
            with col2:
                if st.button(
                    "Reset Ratings",
                    help=(
                        "Reset all ratings to initial values. "
                        "This action cannot be undone."
                    ),
                    type="secondary",
                ):
                    # Open confirmation dialog
                    with st.dialog("Confirm Reset Ratings"):
                        st.markdown("### ⚠️ Reset All Ratings?")
                        st.write(
                            "This will reset **all** ratings to their "
                            "initial values."
                        )
                        st.write("**This action cannot be undone.**")

                        col_confirm, col_cancel = st.columns(2)
                        with col_confirm:
                            if st.button("Yes, Reset Ratings", type="primary"):
                                st.session_state.ratings = initialize_ratings(
                                    st.session_state.names
                                )
                                st.toast(
                                    "✅ Ratings reset to initial values",
                                    icon="✅",
                                )
                                st.rerun()
                        with col_cancel:
                            if st.button("Cancel", type="secondary"):
                                st.rerun()

            # Export ratings
            st.subheader("Export")
            if st.button("Export Ratings as JSON"):
                import json as json_module

                ratings_json = json_module.dumps(
                    {
                        "ratings": st.session_state.ratings,
                        "export_date": datetime.now().isoformat(),
                        "total_names": len(st.session_state.ratings),
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                st.download_button(
                    label="Download Ratings JSON",
                    data=ratings_json,
                    file_name=f"name_ratings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                )

    # Main Content
    if "all_names_data" not in st.session_state:
        st.toast(
            "Please load names from the sidebar.",
            icon="⚠️",
        )
        return

    # Get current gender filter
    current_gender = st.session_state.get("gender_filter", "All")
    # Get current origin filter
    current_origins = st.session_state.get("origin_filter", [])
    # Empty list means no origin filtering (show all regions)
    origins_to_filter = current_origins if current_origins else None

    # Get filtered names using database with caching
    cache_key = (
        f"filtered_names_{current_gender}_"
        f"{tuple(sorted(current_origins)) if current_origins else 'all'}"
    )
    
    if (
        "filtered_names_cache" in st.session_state and
        "filtered_cache_key" in st.session_state and
        st.session_state.filtered_cache_key == cache_key
    ):
        # Use cached filtered names
        filtered_names = st.session_state.filtered_names_cache
    else:
        # Compute and cache
        database.init_database()
        filtered_names = database.get_names_by_filters(
            gender=current_gender if current_gender != "All" else None,
            origins=origins_to_filter,
        )
        st.session_state.filtered_names_cache = filtered_names
        st.session_state.filtered_cache_key = cache_key

    if not filtered_names:
        if current_origins:
            st.toast(
                f"No names found for gender: {current_gender}, "
                f"origins: {current_origins}",
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
            f"Showing {len(filtered_names)} {current_gender.lower()} names "
            f"out of {total_names_count} total",
            icon="ℹ️",
        )

    tab1, tab2 = st.tabs(["Tournament", "Similarity Search"])

    with tab1:
        render_tournament(filtered_names)

    with tab2:
        render_similarity(filtered_names)


if __name__ == "__main__":
    main()
