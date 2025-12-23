"""
Utility functions for the name ranking application.
"""

from typing import Dict, List, Tuple

import numpy as np
import streamlit as st

from data_loader import initialize_or_load_ratings, save_ratings
from elo import K_FACTOR, update_elo, update_elo_draw


def pull_submodule_updates() -> bool:
    """
    Pull latest updates from the git submodule.
    Returns True if successful.
    """
    try:
        import subprocess  # nosec: B404 - git commands are safe, no user input
        import time

        with st.spinner("Pulling latest name data from git submodule..."):
            result = subprocess.run(  # nosec
                ["git", "-C", "godkendtefornavne", "pull"],
                capture_output=True,
                text=True,
                timeout=30,
            )

        if result.returncode == 0:
            st.toast(
                "✅ Submodule updated successfully",
                icon="✅",
            )
            if result.stdout:
                st.text(f"Output: {result.stdout[:200]}")

            # Show reload message with slight delay
            st.toast("⏳ Reloading names in 2 seconds...", icon="⏳")
            time.sleep(2)

            return True
        else:
            st.toast(
                f"Failed to pull submodule: {result.stderr}",
                icon="❌",
                duration="long",
            )
            return False
    except Exception as e:
        st.toast(
            f"Error pulling submodule: {e}",
            icon="❌",
            duration="long",
        )
        return False


def setup_session_state(names: List[str]) -> None:
    if "ratings" not in st.session_state:
        st.session_state.ratings = initialize_or_load_ratings(names)

    if "candidate_a" not in st.session_state:
        st.session_state.candidate_a = ""

    if "candidate_b" not in st.session_state:
        st.session_state.candidate_b = ""

    if "names" not in st.session_state:
        st.session_state.names = names


def select_candidates(names: List[str]) -> Tuple[str, str]:
    """
    Simple random selection for candidates.
    In a more complex app, we'd pick names with low comparison counts or
    close ratings (TrueSkill style).
    """
    if len(names) < 2:
        return "", ""

    return tuple(np.random.choice(names, size=2, replace=False))


def update_elo_and_save(
    ratings: Dict[str, float], winner: str, loser: str, k: float = K_FACTOR
) -> Dict[str, float]:
    """
    Update Elo ratings and save in background.
    """
    updated_ratings = update_elo(ratings, winner, loser, k)

    # Save in background (non-blocking)
    try:
        # Use a simple thread or just save synchronously for now
        # In production, we might use asyncio or a proper background task
        save_ratings(updated_ratings)
    except Exception as e:
        # Don't break the UI if save fails
        st.toast(
            f"Failed to save ratings: {e}",
            icon="⚠️",
        )

    return updated_ratings


def update_elo_draw_and_save(
    ratings: Dict[str, float], player_a: str, player_b: str, k: float = K_FACTOR
) -> Dict[str, float]:
    """
    Update Elo ratings for a draw and save in background.
    """
    updated_ratings = update_elo_draw(ratings, player_a, player_b, k)

    # Save in background (non-blocking)
    try:
        save_ratings(updated_ratings)
    except Exception as e:
        st.toast(
            f"Failed to save ratings: {e}",
            icon="⚠️",
        )

    return updated_ratings
