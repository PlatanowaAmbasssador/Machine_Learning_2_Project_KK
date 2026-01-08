"""
VectorBT backtesting utilities.
"""

import numpy as np
import pandas as pd
import polars as pl
import vectorbt as vbt
from typing import Tuple, List


def vectorbt_backtest(
    predictions: np.ndarray,
    prices: pd.Series,
    threshold: float = 0.5,
    initial_cash: float = 1000000,
    commission: float = 0.001
) -> Tuple[vbt.Portfolio, vbt.Portfolio]:
    """
    Backtest strategy using VectorBT.
    
    Parameters:
    -----------
    predictions : np.ndarray
        Model predictions (probabilities)
    prices : pd.Series
        Price series with datetime index
    threshold : float
        Probability threshold for buy signal
    initial_cash : float
        Initial capital
    commission : float
        Commission rate (e.g., 0.001 = 0.1%)
        
    Returns:
    --------
    strategy_pf : vbt.Portfolio
        Strategy portfolio
    buyhold_pf : vbt.Portfolio
        Buy & hold portfolio
    """
    # Ensure prices is a pandas Series with datetime index
    if isinstance(prices, pd.Series):
        prices_series = prices
    else:
        price_index = pd.date_range(start="2024-01-01", periods=len(prices), freq="1min")
        prices_series = pd.Series(prices, index=price_index)
    
    # Align predictions length to prices_series length
    if len(predictions) < len(prices_series):
        padded = np.zeros(len(prices_series))
        start_idx = len(prices_series) - len(predictions)
        padded[start_idx:] = predictions
        predictions = padded
    elif len(predictions) > len(prices_series):
        predictions = predictions[:len(prices_series)]
    
    signals = (predictions > threshold).astype(int)
    signals_series = pd.Series(signals, index=prices_series.index)
    
    # Strategy: Buy when signal=1, hold cash when signal=0
    strategy_pf = vbt.Portfolio.from_signals(
        prices_series,
        entries=signals_series == 1,
        exits=signals_series == 0,
        init_cash=initial_cash,
        fees=commission,
        freq='1min'
    )
    
    # Buy & Hold: Buy at start, hold until end
    buyhold_pf = vbt.Portfolio.from_holding(
        prices_series,
        init_cash=initial_cash,
        fees=commission,
        freq='1min'
    )
    
    return strategy_pf, buyhold_pf


def aggregate_signals(
    all_test_signals: List[Tuple[pd.DatetimeIndex, np.ndarray]]
) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    """
    Aggregate signals from all test windows into a single continuous series.
    
    Parameters:
    -----------
    all_test_signals : List[Tuple[pd.DatetimeIndex, np.ndarray]]
        List of (dates, signals) tuples from each test window
        
    Returns:
    --------
    aggregated_dates : pd.DatetimeIndex
        Aggregated datetime index
    aggregated_signals : np.ndarray
        Aggregated signals array
    """
    if not all_test_signals:
        return pd.DatetimeIndex([]), np.array([])
    
    # Concatenate all dates and signals, ensuring they have matching lengths
    all_dates = []
    all_sigs = []
    
    for dates, signals in all_test_signals:
        # Ensure dates and signals have the same length
        min_len = min(len(dates), len(signals))
        if min_len == 0:
            continue
        
        # Convert to lists/arrays for easier handling
        dates_list = list(dates[:min_len])
        signals_list = list(signals[:min_len]) if isinstance(signals, np.ndarray) else signals[:min_len]
        
        all_dates.extend(dates_list)
        all_sigs.extend(signals_list)
    
    if not all_dates:
        return pd.DatetimeIndex([]), np.array([])
    
    # Convert to arrays and ensure same length
    dates_array = pd.DatetimeIndex(all_dates)
    signals_array = np.array(all_sigs)
    
    # Ensure arrays have the same length
    min_len = min(len(dates_array), len(signals_array))
    dates_array = dates_array[:min_len]
    signals_array = signals_array[:min_len]
    
    # Sort by date
    sort_idx = dates_array.argsort()
    aggregated_dates = dates_array[sort_idx]
    aggregated_signals = signals_array[sort_idx]
    
    return aggregated_dates, aggregated_signals

