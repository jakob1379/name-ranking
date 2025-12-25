"""
Name Ranker - Main entry point.
Refactored version with modular imports.
"""

import json
import logging
import os
from datetime import datetime

import streamlit as st

from . import database
from .data_loader import load_names_by_gender, save_ratings
from .elo import initialize_ratings
from .ui import render_similarity, render_tournament
from .utils import pull_submodule_updates, setup_session_state

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
                # Load gender-categorized data (no sync for faster startup)
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

                    # Auto-save ratings after loading new dataset
                    if save_ratings(st.session_state.ratings):
                        st.toast(
                            "Ratings saved automatically",
                            icon="ℹ️",
                        )
                else:
                    st.toast(
                        "Failed to load names from submodule",
                        icon="❌",
                        duration="long",
                    )
                    return

        # Submodule management
        st.subheader("Submodule Management")

        # Origin classification
        st.write("**Origin Classification**")

        classify_origins = st.checkbox(
            "Auto-classify after update",
            value=False,
            help="Automatically classify origins after git updates",
        )

        col_class1, col_class2 = st.columns(2)
        with col_class1:
            if st.button(
                "Classify 100 Names", help="Classify 100 unclassified names"
            ):
                with st.spinner("Classifying names..."):
                    try:
                        database.init_database()
                        from classify_origins import classify_all_names

                        classified = classify_all_names(limit=100)
                        if classified > 0:
                            st.toast(
                                f"✅ Classified {classified} names",
                                icon="✅",
                            )
                        else:
                            st.toast(
                                "No unclassified names found",
                                icon="ℹ️",
                            )
                        st.rerun()
                    except ImportError:
                        st.toast(
                            "name2nat not installed. Run: pip install name2nat",
                            icon="❌",
                        )
                    except Exception as e:
                        st.toast(
                            f"Classification failed: {e}",
                            icon="❌",
                        )

        with col_class2:
            if st.button(
                "Classify All", help="Classify all unclassified names (slow)"
            ):
                with st.spinner(
                    "Classifying all names (this may take a while)..."
                ):
                    try:
                        database.init_database()
                        from classify_origins import classify_all_names

                        classified = classify_all_names(limit=None)
                        if classified > 0:
                            st.toast(
                                f"✅ Classified {classified} names",
                                icon="✅",
                            )
                        else:
                            st.toast(
                                "No unclassified names found",
                                icon="ℹ️",
                            )
                        st.rerun()
                    except ImportError:
                        st.toast(
                            "name2nat not installed. Run: pip install name2nat",
                            icon="❌",
                        )
                    except Exception as e:
                        st.toast(
                            f"Classification failed: {e}",
                            icon="❌",
                        )

        # Show classification progress
        try:
            database.init_database()
            stats = database.get_stats()
            total = stats['total_names']
            classified = stats['classified_names']
            if total > 0:
                percentage = classified / total * 100
                st.caption(
                    f"Origin classification: {classified}/{total} names "
                    f"({percentage:.1f}%)"
                )
        except Exception:
            pass  # Silently fail if database not ready

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Reload Names", help="Fast reload from database"):
                with st.spinner("Reloading..."):
                    # Reload gender-categorized data (no sync for speed)
                    gender_data = load_names_by_gender(
                        sync_with_submodule=False
                    )
                    if gender_data and "All" in gender_data:
                        # Update the full dataset
                        st.session_state.all_names_data = gender_data
                        st.session_state.all_names = gender_data["All"]

                        # Reinitialize ratings with all names
                        setup_session_state(gender_data["All"])
                        st.toast(
                            f"Reloaded {len(gender_data['All'])} names",
                            icon="✅",
                        )
                        st.rerun()

        with col2:
            if st.button(
                "Sync Names", help="Sync database with local submodule"
            ):
                with st.spinner("Syncing..."):
                    from utils import sync_names_from_submodule

                    inserted = sync_names_from_submodule()
                    if inserted > 0:
                        # Reload names after sync
                        gender_data = load_names_by_gender(
                            sync_with_submodule=False
                        )
                        if gender_data and "All" in gender_data:
                            st.session_state.all_names_data = gender_data
                            st.session_state.all_names = gender_data["All"]
                            setup_session_state(gender_data["All"])
                        st.rerun()

        with col3:
            if st.button("Check for Updates", help="Pull git updates and sync"):
                with st.spinner("Checking for updates..."):
                    if pull_submodule_updates(
                        classify_origins=classify_origins
                    ):
                        st.rerun()

        # Show submodule status
        if os.path.exists("godkendtefornavne"):
            try:
                import subprocess  # nosec: B404 - git commands are safe, no user input

                result = subprocess.run(  # nosec
                    [
                        "git",
                        "-C",
                        "godkendtefornavne",
                        "log",
                        "-1",
                        "--format=%cd",
                        "--date=short",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    last_update = result.stdout.strip()
                    st.caption(f"Last update: {last_update}")
            except Exception:  # nosec: B110 - non-critical, can safely ignore git log failure
                pass

        st.divider()

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

    # Get filtered names using database
    database.init_database()
    filtered_names = database.get_names_by_filters(
        gender=current_gender if current_gender != "All" else None,
        origins=origins_to_filter,
    )

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
