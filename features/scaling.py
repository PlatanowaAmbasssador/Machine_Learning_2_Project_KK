"""
Scaling utilities with strict temporal separation.
Scalers are fit only on training data.
"""

from sklearn.preprocessing import StandardScaler
import numpy as np
import polars as pl
from typing import Tuple


def fit_scaler(train_data: pl.DataFrame, feature_cols: list) -> StandardScaler:
    """
    Fit scaler ONLY on training data.
    
    Parameters:
    -----------
    train_data : pl.DataFrame
        Training data (only data used for fitting)
    feature_cols : list
        Feature column names
        
    Returns:
    --------
    scaler : StandardScaler
        Fitted scaler
    """
    X_train = train_data.select(feature_cols).to_numpy()
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def transform_data(
    scaler: StandardScaler,
    data: pl.DataFrame,
    feature_cols: list
) -> np.ndarray:
    """
    Transform data using pre-fitted scaler.
    
    Parameters:
    -----------
    scaler : StandardScaler
        Pre-fitted scaler (from training data)
    data : pl.DataFrame
        Data to transform
    feature_cols : list
        Feature column names
        
    Returns:
    --------
    X_scaled : np.ndarray
        Scaled features
    """
    X = data.select(feature_cols).to_numpy()
    return scaler.transform(X)

