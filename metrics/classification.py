"""
Classification metrics for binary direction prediction.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Calculate classification metrics for binary direction prediction.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True binary labels (0 or 1)
    y_pred : np.ndarray
        Predicted probabilities
    threshold : float
        Probability threshold for binary classification
        
    Returns:
    --------
    metrics : dict
        Dictionary with accuracy, precision, recall, f1, and confusion matrix
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]]
        }
    
    # Ensure same length
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    
    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Ensure y_true is binary
    y_true_binary = (y_true >= 0.5).astype(int) if y_true.dtype != int else y_true
    
    # Calculate metrics
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    cm = confusion_matrix(y_true_binary, y_pred_binary).tolist()
    
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm
    }


def create_binary_labels_from_prices(prices: np.ndarray) -> np.ndarray:
    """
    Convert price series to binary labels (1 if price goes up, 0 if price goes down).
    
    Parameters:
    -----------
    prices : np.ndarray
        Price series
        
    Returns:
    --------
    labels : np.ndarray
        Binary labels (1 = price increase, 0 = price decrease)
        Length is len(prices) - 1 (one less than input)
    """
    if len(prices) < 2:
        return np.array([])
    
    # Calculate price changes
    price_changes = np.diff(prices)
    
    # Convert to binary: 1 if price goes up, 0 if price goes down
    labels = (price_changes > 0).astype(int)
    
    return labels

