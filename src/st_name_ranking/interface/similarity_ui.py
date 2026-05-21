"""Similarity-search screen rendering."""

import logging
from typing import Literal

import polars as pl
import streamlit as st

from st_name_ranking.similarity import (
    get_string_similarity_scores,
    get_vector_similarity_scores,
    load_embedding_model,
)

logger = logging.getLogger(__name__)


def render_similarity(names: list[str]) -> None:
    """Render similarity search interface.

    Allows searching for names similar to a given name.
    """
    logger.debug("Rendering similarity search with %d names", len(names))
    st.header("Similarity Search")

    search_type: Literal["String (Levenshtein)", "Vector (LLM Embedding)"] = st.radio(
        "Search Method",
        ["String (Levenshtein)", "Vector (LLM Embedding)"],
    )

    query = st.text_input("Reference Name", value="Alma")

    if st.button("Find Similar") and query:
        if search_type == "String (Levenshtein)":
            results = get_string_similarity_scores(query, names, limit=10)
            st.dataframe(
                pl.DataFrame(results, schema=["Name", "Similarity Score"], orient="row"),
                width="stretch",
                hide_index=True,
            )
        else:
            with st.spinner("Loading Embedding Model (first run is slow)..."):
                try:
                    model = load_embedding_model()
                except RuntimeError as e:
                    st.warning(str(e))
                    return

            with st.spinner("Computing Similarities..."):
                results = get_vector_similarity_scores(
                    model,
                    query,
                    names,
                    limit=10,
                )

            st.dataframe(
                pl.DataFrame(results, schema=["Name", "Cosine Similarity"], orient="row"),
                width="stretch",
                hide_index=True,
            )
