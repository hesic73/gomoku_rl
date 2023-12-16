import numpy as np
from scipy.special import logit


def compute_elo_ratings(payoff: np.ndarray, average_rating: float = 1200) -> np.ndarray:
    """compute Elo ratings from the payoff matrix

    Args:
        payoff (np.ndarray): win rate matrix of shpe (n,n).
        average_rating (float, optional): I play Honor of Kings, so its default value is 1200.

    Returns:
        np.ndarray: (n,) estimated elo ratings
    """
    assert len(payoff.shape) == 2 and payoff.shape[0] == payoff.shape[1]
    assert np.allclose(payoff, 1 - payoff.T)
    payoff = payoff.clip(min=1e-8, max=1 - 1e-8)
    payoff = logit(payoff)
    elo_ratings = payoff.mean(axis=-1)
    elo_ratings = elo_ratings * (400 / np.log(10)) + average_rating
    return elo_ratings


def compute_expected_score(rating_0: float, rating_1: float) -> float:
    return 1 / (1 + 10.0 ** ((rating_1 - rating_0) / 400))


class Elo:
    def __init__(self) -> None:
        self.players: dict[str, float] = {}

    def addPlayer(self, name: str, rating: float = 1200):
        assert name not in self.players
        self.players[name] = rating

    def expected_score(self, player_0: str, player_1: str) -> float:
        rating_0 = self.players[player_0]
        rating_1 = self.players[player_1]
        return compute_expected_score(rating_0, rating_1)

    def update(self, player_0: str, player_1: str, score: float, K: float = 64):
        e = self.expected_score(player_0, player_1)
        tmp = K * (score - e)
        self.players[player_0] = self.players[player_0] + tmp
        self.players[player_1] = self.players[player_1] - tmp
