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
