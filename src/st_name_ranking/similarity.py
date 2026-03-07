"""Similarity functions for name matching."""

import logging

import numpy as np
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer

from st_name_ranking.types import SimilarityScore

logger = logging.getLogger(__name__)


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


def load_embedding_model() -> SentenceTransformer:
    # 'paraphrase-multilingual-MiniLM-L12-v2' is good for Danish/English
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def get_vector_similarity_scores(
    model: SentenceTransformer,
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
