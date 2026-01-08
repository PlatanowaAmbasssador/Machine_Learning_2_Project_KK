"""
Regression metrics for price prediction evaluation.
"""

import numpy as np


def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Squared Error.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
        
    Returns:
    --------
    mse : float
        Mean Squared Error
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0.0
    
    # Ensure same length
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    
    return float(np.mean((y_true - y_pred) ** 2))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
        
    Returns:
    --------
    mae : float
        Mean Absolute Error
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0.0
    
    # Ensure same length
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    
    return float(np.mean(np.abs(y_true - y_pred)))


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
        
    Returns:
    --------
    rmse : float
        Root Mean Squared Error
    """
    mse = calculate_mse(y_true, y_pred)
    return float(np.sqrt(mse))


def calculate_price_predictions_from_probabilities(
    probabilities: np.ndarray,
    current_prices: np.ndarray,
    method: str = "weighted"
) -> np.ndarray:
    """
    Convert probability predictions to price predictions.
    
    Since models predict probabilities (0-1) for binary classification,
    we convert these to price predictions for MSE/MAE calculation.
    
    The prediction at time t predicts the price at time t+1.
    We compare predicted_price[t] to actual_price[t+1].
    
    Parameters:
    -----------
    probabilities : np.ndarray
        Model predictions (probabilities between 0 and 1) for each time step
    current_prices : np.ndarray
        Current prices at each time step (used to predict next price)
    method : str
        Method to convert probabilities to prices:
        - "weighted": Use probability as weight for expected price change (default)
        - "threshold": Use threshold to determine direction, then apply average change
        - "linear": Linear mapping from probability to price change
        
    Returns:
    --------
    predicted_prices : np.ndarray
        Predicted prices (same length as probabilities)
        predicted_prices[i] is the predicted price for the next period after current_prices[i]
    """
    if len(probabilities) == 0 or len(current_prices) == 0:
        return np.array([])
    
    # Ensure same length
    min_len = min(len(probabilities), len(current_prices))
    probabilities = probabilities[:min_len]
    current_prices = current_prices[:min_len]
    
    # Calculate historical price changes for context
    if len(current_prices) > 1:
        # Calculate percentage changes
        price_changes_pct = np.diff(current_prices) / current_prices[:-1]
        if len(price_changes_pct) > 0:
            avg_abs_change = np.mean(np.abs(price_changes_pct))
        else:
            avg_abs_change = 0.01
    else:
        avg_abs_change = 0.01
    
    if method == "weighted":
        # Use probability to weight the expected price change
        # High probability (close to 1) = expect price increase
        # Low probability (close to 0) = expect price decrease
        # Probability 0.5 = expect no change
        
        # Map probability [0, 1] to price change multiplier [-1, 1]
        # Probability 0.5 -> 0 change, probability 1 -> +avg_change, probability 0 -> -avg_change
        price_change_multiplier = (probabilities - 0.5) * 2  # Maps [0,1] to [-1,1]
        
        # Predicted next price = current price * (1 + expected_change)
        predicted_prices = current_prices * (1 + price_change_multiplier * avg_abs_change)
        
    elif method == "threshold":
        # Use threshold to determine direction
        threshold = 0.5
        directions = (probabilities > threshold).astype(float) * 2 - 1  # -1 or 1
        
        # Predicted next price = current price * (1 + direction * avg_change)
        predicted_prices = current_prices * (1 + directions * avg_abs_change)
        
    else:  # linear
        # Linear mapping: probability 0 -> price decrease, probability 1 -> price increase
        # Map probability to price change: 0 -> -avg_change, 1 -> +avg_change
        price_change_pct = (probabilities - 0.5) * 2 * avg_abs_change
        predicted_prices = current_prices * (1 + price_change_pct)
    
    return predicted_prices

