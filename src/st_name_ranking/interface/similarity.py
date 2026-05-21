"""Similarity functions for name matching."""

import importlib
import logging
from functools import cache
from typing import Any, Protocol, cast

import numpy as np
from rapidfuzz import fuzz, process

from st_name_ranking.types import SimilarityScore

logger = logging.getLogger(__name__)

SENTENCE_TRANSFORMER_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
SEMANTIC_SEARCH_EXTRA = "st-name-ranking[semantic-search]"


class EmbeddingModel(Protocol):
    """Model interface used by semantic similarity search."""

    def encode(self, sentences: list[str]) -> np.ndarray:
        """Encode strings into embedding vectors."""


def get_string_similarity_scores(
    target: str,
    candidates: list[str],
    limit: int = 10,
) -> list[SimilarityScore]:
    """Uses RapidFuzz (Levenshtein) to find similar names.
    Returns list of SimilarityScore.
    """
    logger.debug(
        "String similarity search: target='%s', candidates=%d, limit=%d",
        target,
        len(candidates),
        limit,
    )
    if not candidates:
        return []

    # process.extract returns (match, score, index)
    results = process.extract(
        target,
        candidates,
        scorer=fuzz.ratio,
        limit=limit,
    )
    return [SimilarityScore(name=item[0], score=float(item[1])) for item in results]


def load_embedding_model() -> EmbeddingModel:
    """Load the optional sentence-transformer model for semantic search."""
    transformer_cls = _get_sentence_transformer_class()
    return transformer_cls(SENTENCE_TRANSFORMER_MODEL)


@cache
def _get_sentence_transformer_class() -> type[Any]:
    try:
        sentence_transformers = importlib.import_module("sentence_transformers")
    except ImportError as e:
        msg = (
            "Semantic search requires the optional sentence-transformers stack. "
            f"Install it with `uv sync --extra semantic-search` or `{SEMANTIC_SEARCH_EXTRA}`."
        )
        raise RuntimeError(msg) from e

    return cast("type[Any]", sentence_transformers.SentenceTransformer)


def get_vector_similarity_scores(
    model: EmbeddingModel,
    target: str,
    candidates: list[str],
    limit: int = 10,
) -> list[SimilarityScore]:
    """Uses LLM embeddings to find semantic similarity.
    Returns list of SimilarityScore.
    """
    logger.debug(
        "Vector similarity search: target='%s', candidates=%d, limit=%d",
        target,
        len(candidates),
        limit,
    )
    if not candidates:
        return []

    # Encode target and all candidates
    target_embedding = model.encode([target])
    candidate_embeddings = model.encode(candidates)

    # Compute Cosine Similarity
    # (N, 1) dot (1, N) -> (1, N)
    scores = np.dot(candidate_embeddings, target_embedding.T).flatten()

    # Get indices of top scores
    top_indices = np.argsort(scores)[::-1][:limit]

    return [SimilarityScore(name=candidates[i], score=float(scores[i])) for i in top_indices]
