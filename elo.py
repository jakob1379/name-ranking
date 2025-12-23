"""
Elo rating system implementation.
Pure calculation functions with no side effects.
"""

from typing import Dict, List

K_FACTOR: float = 32.0
INITIAL_RATING: float = 1500.0


def expected_score(rating_a: float, rating_b: float) -> float:
    """
    Calculate expected score for A.
    Formula: E_A = 1 / (1 + 10 ^ ((R_B - R_A) / 400))
    """
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_elo_generic(
    ratings: Dict[str, float],
    player_a: str,
    player_b: str,
    score_a: float,
    score_b: float,
    k: float = K_FACTOR,
) -> Dict[str, float]:
    """
    Generic Elo update for any outcome.
    score_a is the result for player_a (typically 1.0 for win,
    0.5 for draw, 0.0 for loss).
    """
    if player_a not in ratings or player_b not in ratings:
        return ratings

    r_a = ratings[player_a]
    r_b = ratings[player_b]

    e_a = expected_score(r_a, r_b)
    e_b = 1.0 - e_a

    ratings[player_a] = r_a + k * (score_a - e_a)
    ratings[player_b] = r_b + k * (score_b - e_b)

    return ratings


def update_elo(
    ratings: Dict[str, float], winner: str, loser: str, k: float = K_FACTOR
) -> Dict[str, float]:
    """
    Update Elo ratings based on a binary outcome (1 for winner, 0 for loser).
    """
    return update_elo_generic(ratings, winner, loser, 1.0, 0.0, k)


def update_elo_draw(
    ratings: Dict[str, float], player_a: str, player_b: str, k: float = K_FACTOR
) -> Dict[str, float]:
    """
    Update Elo ratings for a draw (0.5 points each).
    """
    return update_elo_generic(ratings, player_a, player_b, 0.5, 0.5, k)


def initialize_ratings(names: List[str]) -> Dict[str, float]:
    return {name.strip(): INITIAL_RATING for name in names if name.strip()}
