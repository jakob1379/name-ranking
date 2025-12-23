"""
Name Ranker - Main entry point.
Refactored version with modular imports.
"""

import os
from datetime import datetime

import streamlit as st

from data_loader import load_names_by_gender, save_ratings
from elo import initialize_ratings
from ui import render_similarity, render_tournament
from utils import pull_submodule_updates, setup_session_state


def main() -> None:
    st.set_page_config(page_title="Name Ranker", layout="wide")
    st.title("Name Preference Ranker")

    # Data Loading - Only from submodule
    with st.sidebar:
        # Auto-load from submodule on first run
        if "all_names_data" not in st.session_state:
            with st.spinner("Loading names from submodule..."):
                # Load gender-categorized data
                gender_data = load_names_by_gender()
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

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Reload Names"):
                with st.spinner("Reloading..."):
                    # Reload gender-categorized data
                    gender_data = load_names_by_gender()
                    if gender_data and "All" in gender_data:
                        # Update the full dataset
                        st.session_state.all_names_data = gender_data
                        st.session_state.all_names = gender_data["All"]

                        # Reinitialize ratings with all names
                        setup_session_state(gender_data["All"])
                        st.toast(
                            f"Reloaded {len(gender_data['All'])} total names",
                            icon="✅",
                        )

                        # Show breakdown
                        for gender in ["Male", "Female", "Unisex"]:
                            if gender in gender_data:
                                st.toast(
                                    f"{gender}: "
                                    f"{len(gender_data[gender])} names",
                                    icon="ℹ️",
                                )
                        st.rerun()

        with col2:
            if st.button("Check for Updates"):
                with st.spinner("Checking for updates..."):
                    if pull_submodule_updates():
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

    # Get filtered names for the current gender
    if current_gender in st.session_state.all_names_data:
        filtered_names = st.session_state.all_names_data[current_gender]
    else:
        filtered_names = st.session_state.all_names_data.get("All", [])

    if not filtered_names:
        st.toast(
            f"No names found for gender filter: {current_gender}",
            icon="⚠️",
        )
        return

    # Show filter info
    st.toast(
        f"Showing {len(filtered_names)} {current_gender.lower()} names "
        f"(out of {len(st.session_state.all_names)} total)",
        icon="ℹ️",
    )

    tab1, tab2 = st.tabs(["Tournament", "Similarity Search"])

    with tab1:
        render_tournament(filtered_names)

    with tab2:
        render_similarity(filtered_names)


if __name__ == "__main__":
    main()
