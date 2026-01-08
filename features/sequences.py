"""
Sequence creation for sequential models (LSTM, CNN).
"""

import numpy as np
from typing import Tuple


def create_sequences(data: np.ndarray, target: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for sequential model training.
    
    Parameters:
    -----------
    data : np.ndarray
        Features array (n_samples, n_features)
    target : np.ndarray
        Target variable (n_samples,)
    seq_length : int
        Length of sequence (lookback window)
    
    Returns:
    --------
    X : np.ndarray
        Sequences (n_samples - seq_length, seq_length, n_features)
    y : np.ndarray
        Targets (n_samples - seq_length,)
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(target[i+seq_length])
    return np.array(X), np.array(y)

