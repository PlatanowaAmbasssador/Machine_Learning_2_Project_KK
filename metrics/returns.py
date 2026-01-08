"""
Strategy returns calculation from predictions.
"""

import numpy as np


def get_strategy_returns(predictions: np.ndarray, prices: np.ndarray, threshold: float = 0.5) -> tuple:
    """
    Generate trading signals and calculate returns.
    
    Parameters:
    -----------
    predictions : np.ndarray
        Model predictions (probabilities)
    prices : np.ndarray
        Price series (close prices)
    threshold : float
        Probability threshold for buy signal
    
    Returns:
    --------
    returns : np.ndarray
        Strategy returns
    signals : np.ndarray
        Trading signals (1 = buy, 0 = hold cash)
    """
    signals = (predictions > threshold).astype(int)
    # Calculate returns: buy when signal=1, hold cash when signal=0
    # Forward fill signals to align with price changes
    returns = np.diff(prices) / prices[:-1] * signals[:-1]
    return returns, signals[:-1]

