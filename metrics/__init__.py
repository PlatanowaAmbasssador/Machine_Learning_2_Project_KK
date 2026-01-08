"""
Trading-aware performance metrics.
"""

from .ir2 import calculate_ir2_from_returns, IR2
from .returns import get_strategy_returns
from .regression import calculate_mse, calculate_mae, calculate_rmse, calculate_price_predictions_from_probabilities
from .classification import calculate_classification_metrics, create_binary_labels_from_prices

__all__ = [
    'calculate_ir2_from_returns', 
    'IR2', 
    'get_strategy_returns',
    'calculate_mse',
    'calculate_mae',
    'calculate_rmse',
    'calculate_price_predictions_from_probabilities',
    'calculate_classification_metrics',
    'create_binary_labels_from_prices'
]

