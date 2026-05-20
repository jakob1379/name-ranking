"""Streamlit-facing application actions."""

import logging
import subprocess
import time

import streamlit as st

from st_name_ranking import database
from st_name_ranking.data_loader import DataLoaderError, initialize_or_load_ratings
from st_name_ranking.database import initialize_ratings

logger = logging.getLogger(__name__)


def pull_submodule_updates(*, classify_origins: bool = False) -> bool:
    """Pull latest submodule data, sync names, and optionally classify origins."""
    logger.debug("Pulling submodule updates, classify_origins=%s", classify_origins)
    try:
        with st.spinner("Pulling latest name data from git submodule..."):
            result = subprocess.run(
                ["git", "-C", "godkendtefornavne", "pull"],  # noqa: S607
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
    except subprocess.SubprocessError as e:
        st.toast(
            f"Error pulling submodule: {e}",
            icon="❌",
            duration="long",
        )
        return False

    if result.returncode != 0:
        st.toast(
            f"Failed to pull submodule: {result.stderr}",
            icon="❌",
            duration="long",
        )
        return False

    st.toast("✅ Submodule updated successfully", icon="✅")
    if result.stdout:
        st.text(f"Output: {result.stdout[:200]}")

    _sync_after_submodule_pull()
    if classify_origins:
        _classify_origins_after_sync()

    st.toast("⏳ Reloading names in 2 seconds...", icon="⏳")
    time.sleep(2)
    return True


def setup_session_state(names: list[str]) -> None:
    """Initialize the Streamlit session keys needed by the app."""
    if "ratings" not in st.session_state:
        try:
            st.session_state["ratings"] = initialize_or_load_ratings(names)
        except DataLoaderError as e:
            st.toast(f"Could not load saved ratings: {e}", icon="⚠️")
            st.session_state["ratings"] = initialize_ratings(names)

    if "candidate_a" not in st.session_state:
        st.session_state["candidate_a"] = ""

    if "candidate_b" not in st.session_state:
        st.session_state["candidate_b"] = ""

    if "names" not in st.session_state:
        st.session_state["names"] = names


def sync_names_from_submodule() -> int:
    """Sync names from the submodule JSON and report the result in Streamlit."""
    try:
        database.init_database()
        with st.spinner("Syncing names from submodule..."):
            inserted = database.sync_names_with_submodule()
            if inserted > 0:
                st.toast(
                    f"✅ Added {inserted} new names to database",
                    icon="✅",
                )
            else:
                st.toast(
                    "Database already up to date with submodule",
                    icon="ℹ️",
                )
            return inserted
    except (RuntimeError, ValueError, subprocess.SubprocessError) as e:
        st.toast(
            f"Failed to sync names: {e}",
            icon="❌",
            duration="long",
        )
        return 0


def _sync_after_submodule_pull() -> None:
    with st.spinner("Syncing new names with database..."):
        try:
            database.init_database()
            inserted = database.sync_names_with_submodule()
            if inserted > 0:
                st.toast(
                    f"✅ Added {inserted} new names to database",
                    icon="✅",
                )
            else:
                st.toast("No new names to add", icon="ℹ️")
        except (RuntimeError, ValueError, subprocess.SubprocessError) as sync_error:
            st.toast(
                f"Failed to sync names: {sync_error}",
                icon="❌",
                duration="long",
            )


def _classify_origins_after_sync() -> None:
    try:
        database.init_database()
        from st_name_ranking.classify_origins import classify_all_names  # noqa: PLC0415

        with st.spinner("Classifying name origins..."):
            classified = classify_all_names(limit=None)
            if classified > 0:
                st.toast(
                    f"✅ Classified {classified} name origins",
                    icon="✅",
                )
            else:
                st.toast("No names needed classification", icon="ℹ️")
    except ImportError:
        st.toast(
            "ethnidata not installed. Run: pip install ethnidata",
            icon="⚠️",
        )
    except (RuntimeError, ValueError) as classify_error:
        st.toast(
            f"Failed to classify origins: {classify_error}",
            icon="❌",
            duration="long",
        )
