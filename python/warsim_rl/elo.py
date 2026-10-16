"""Elo ratings for league members (scores: 1 win, 0.5 draw, 0 loss)."""

from __future__ import annotations

import math
from typing import Iterable, Optional


def expected_score(rating_a: float, rating_b: float) -> float:
    """Probability-like expected score of A against B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_elo(rating_a: float, rating_b: float, score_a: float, k: float = 16.0) -> tuple:
    """Return new (rating_a, rating_b) after one game; zero-sum update."""
    if not 0.0 <= score_a <= 1.0:
        raise ValueError("score_a must be in [0, 1]")
    delta = k * (score_a - expected_score(rating_a, rating_b))
    return rating_a + delta, rating_b - delta


def performance_rating(opponent_rating: float, score: float, games: int) -> float:
    """Rating implied by scoring ``score`` (total points) in ``games`` games
    against one opponent. Clipped to stay finite at 0% / 100%."""
    if games <= 0:
        raise ValueError("games must be positive")
    eps = 0.5 / games
    p = min(max(score / games, eps), 1.0 - eps)
    return opponent_rating + 400.0 * math.log10(p / (1.0 - p))


class EloTable:
    """Ratings by player id. Anchored players keep a fixed rating."""

    def __init__(
        self, initial: float = 1000.0, k: float = 16.0, anchors: Optional[Iterable[str]] = None
    ) -> None:
        self.initial = initial
        self.k = k
        self.ratings: dict = {}
        self.anchors = set(anchors or ())

    def get(self, player: str) -> float:
        return self.ratings.get(player, self.initial)

    def set(self, player: str, rating: float) -> None:
        self.ratings[player] = float(rating)

    def record(self, a: str, b: str, score_a: float) -> None:
        ra, rb = self.get(a), self.get(b)
        na, nb = update_elo(ra, rb, score_a, self.k)
        if a not in self.anchors:
            self.ratings[a] = na
        if b not in self.anchors:
            self.ratings[b] = nb

    def state_dict(self) -> dict:
        return {
            "initial": self.initial,
            "k": self.k,
            "ratings": dict(self.ratings),
            "anchors": sorted(self.anchors),
        }

    def load_state_dict(self, state: dict) -> None:
        self.initial = float(state["initial"])
        self.k = float(state["k"])
        self.ratings = {str(p): float(r) for p, r in state["ratings"].items()}
        self.anchors = set(state["anchors"])
