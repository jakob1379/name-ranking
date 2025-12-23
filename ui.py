"""
UI rendering functions for the name ranking application.
"""

import logging
from typing import List, Literal, Optional, Union

import pandas as pd
import streamlit as st

from elo import INITIAL_RATING
from similarity import (
    get_string_similarity_scores,
    get_vector_similarity_scores,
    load_embedding_model,
)
from utils import (
    select_candidates,
    update_elo_and_save,
    update_elo_draw_and_save,
)

logger = logging.getLogger(__name__)


def display_name_with_rating(
    name: str, rating: float, delta: Optional[Union[int, float, str]] = None
) -> None:
    """
    Display name much larger than rating using st.metric with custom styling.
    CSS is injected in render_tournament to make the value (name) larger
    and label (rating) smaller.
    delta: difference in Elo compared to opponent (positive if higher,
    negative if lower).
    """
    # Format delta for display
    if delta is not None:
        # Convert numeric delta to string with sign
        if isinstance(delta, (int, float)):
            delta_str = f"{delta:+.0f}"
        else:
            delta_str = str(delta)
    else:
        delta_str = None

    # Use st.metric with swapped label/value to get desired visual hierarchy
    # Value will be large (name), label will be smaller (rating)
    st.metric(value=name, label=f"{rating:.0f}", delta=delta_str, border=True)


def render_tournament(names: List[str]) -> None:
    logger.debug("Rendering tournament with %d names", len(names))
    st.header("Elo Rating Tournament")
    st.write(f"Comparing {len(names)} names")
    st.caption(
        "💡 **Keyboard shortcuts**: Left arrow (←) for left name, "
        "Right arrow (→) for right name, Up arrow (↑) for draw"
    )

    # Pick candidates if not set or just reset
    if not st.session_state.candidate_a or not st.session_state.candidate_b:
        c_a, c_b = select_candidates(names)
        st.session_state.candidate_a = c_a
        st.session_state.candidate_b = c_b

    # Inject CSS for metric styling and equal column heights
    st.markdown(
        """
    <style>
    /* Style for st.metric display */
    div[data-testid="stMetricValue"] p {
        font-size: 48px !important;
        font-weight: bold !important;
        text-align: center !important;
        margin-bottom: 5px !important;
    }
    div[data-testid="stMetricLabel"] p {
        font-size: 24px !important;
        color: #666 !important;
        text-align: center !important;
        margin-top: 0 !important;
    }
    div[data-testid="stMetric"] {
        text-align: center !important;
    }
    div[data-testid="stMetricDelta"] svg {
        width: 20px !important;
        height: 20px !important;
    }

    /* Ensure columns have equal height */
    div[data-testid="column"] {
        display: flex !important;
        flex-direction: column !important;
        min-height: 300px !important;
    }
    div[data-testid="column"] > div[data-testid="stVerticalBlock"] {
        flex-grow: 1 !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    _, col_left, _, col_right, _ = st.columns([0.8, 1, 0.4, 1, 0.8])

    # Get both ratings before displaying
    rating_a = st.session_state.ratings.get(
        st.session_state.candidate_a, INITIAL_RATING
    )
    rating_b = st.session_state.ratings.get(
        st.session_state.candidate_b, INITIAL_RATING
    )

    # Calculate rating differences for delta display
    delta_a = rating_a - rating_b  # Positive if A is higher rated
    delta_b = rating_b - rating_a  # Positive if B is higher rated

    with col_left:
        display_name_with_rating(
            st.session_state.candidate_a, rating_a, delta=delta_a
        )

        # Centered button container
        button_container = st.container()
        with button_container:
            st.markdown(
                "<div style='text-align: center'>", unsafe_allow_html=True
            )
            button_clicked = st.button(
                f"👈 Prefer {st.session_state.candidate_a}",
                key="vote_a",
                width="stretch",
                type="primary",
                shortcut="Left",
                help=(
                    f"Select {st.session_state.candidate_a} as preferred "
                    "(Left arrow key)"
                ),
            )
            st.markdown("</div>", unsafe_allow_html=True)

        if button_clicked:
            update_elo_and_save(
                st.session_state.ratings,
                st.session_state.candidate_a,
                st.session_state.candidate_b,
            )
            st.session_state.candidate_a, st.session_state.candidate_b = (
                select_candidates(names)
            )
            st.rerun()

    with col_right:
        display_name_with_rating(
            st.session_state.candidate_b, rating_b, delta=delta_b
        )

        # Centered button container (same as left side)
        button_container = st.container()
        with button_container:
            st.markdown(
                "<div style='text-align: center'>", unsafe_allow_html=True
            )
            button_clicked = st.button(
                f"Prefer {st.session_state.candidate_b} 👉",
                key="vote_b",
                width="stretch",
                type="primary",
                shortcut="Right",
                help=(
                    f"Select {st.session_state.candidate_b} as preferred "
                    "(Right arrow key)"
                ),
            )
            st.markdown("</div>", unsafe_allow_html=True)

        if button_clicked:
            update_elo_and_save(
                st.session_state.ratings,
                st.session_state.candidate_b,
                st.session_state.candidate_a,
            )
            st.session_state.candidate_a, st.session_state.candidate_b = (
                select_candidates(names)
            )
            st.rerun()

    # Draw button centered below both names
    _, col_mid, _ = st.columns([1, 0.4, 1])
    with col_mid:
        draw_clicked = st.button(
            "🤝 Draw / Equal Preference",
            key="vote_draw",
            width="stretch",
            type="secondary",
            shortcut="Up",
            help="Mark both names as equally preferred (Up arrow key)",
        )

    if draw_clicked:
        st.toast("you chose a draw!", duration="long")
        update_elo_draw_and_save(
            st.session_state.ratings,
            st.session_state.candidate_a,
            st.session_state.candidate_b,
        )
        st.session_state.candidate_a, st.session_state.candidate_b = (
            select_candidates(names)
        )
        st.rerun()

    st.divider()
    st.subheader("Current Top 10")
    sorted_ratings = sorted(
        st.session_state.ratings.items(), key=lambda x: x[1], reverse=True
    )
    df = pd.DataFrame(sorted_ratings[:10], columns=["Name", "Rating"])
    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        column_config={
            "ratings": st.column_config.NumberColumn(
                "Rating",
                help="Higher is better",
                format="%d",
                pinned=True,
                width="small",
            )
        },
    )


def render_similarity(names: List[str]) -> None:
    logger.debug("Rendering similarity search with %d names", len(names))
    st.header("Similarity Search")

    search_type: Literal["String", "Vector"] = st.radio(
        "Search Method", ["String (Levenshtein)", "Vector (LLM Embedding)"]
    )

    query = st.text_input("Reference Name", value="Alma")

    if st.button("Find Similar") and query:
        if search_type == "String (Levenshtein)":
            results = get_string_similarity_scores(query, names, limit=10)
            st.dataframe(
                pd.DataFrame(results, columns=["Name", "Similarity Score"]),
                width="stretch",
                hide_index=True,
            )
        else:
            with st.spinner("Loading Embedding Model (first run is slow)..."):
                model = load_embedding_model()

            with st.spinner("Computing Similarities..."):
                results = get_vector_similarity_scores(
                    model, query, names, limit=10
                )

            st.dataframe(
                pd.DataFrame(results, columns=["Name", "Cosine Similarity"]),
                width="stretch",
                hide_index=True,
            )
