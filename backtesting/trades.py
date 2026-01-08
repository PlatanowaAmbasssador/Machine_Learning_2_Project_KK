"""
Trade extraction and persistence utilities.
Stores entry/exit timestamps and position direction.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd
import polars as pl


@dataclass
class TradeRecord:
    """Single trade record with entry/exit information."""
    entry_timestamp: pd.Timestamp
    exit_timestamp: pd.Timestamp
    position_direction: int  # 1 = long, -1 = short, 0 = flat
    entry_price: float
    exit_price: float
    pnl: float
    return_pct: float


def extract_trades(
    predictions: np.ndarray,
    prices: pd.Series,
    dates: pd.DatetimeIndex,
    threshold: float = 0.5
) -> List[TradeRecord]:
    """
    Extract individual trades from predictions and prices.
    
    Parameters:
    -----------
    predictions : np.ndarray
        Model predictions (probabilities)
    prices : pd.Series
        Price series with datetime index
    dates : pd.DatetimeIndex
        Dates corresponding to predictions (aligned after sequence creation)
    threshold : float
        Probability threshold for buy signal
        
    Returns:
    --------
    trades : List[TradeRecord]
        List of trade records
    """
    signals = (predictions > threshold).astype(int)
    trades = []
    
    # Align signals with dates
    if len(signals) != len(dates):
        # If predictions are shorter (due to sequence creation), pad with zeros
        if len(signals) < len(dates):
            padded_signals = np.zeros(len(dates))
            start_idx = len(dates) - len(signals)
            padded_signals[start_idx:] = signals
            signals = padded_signals
    
    # Track current position
    in_position = False
    entry_idx = None
    entry_price = None
    entry_date = None
    
    for i in range(len(signals) - 1):
        current_signal = signals[i]
        next_signal = signals[i + 1]
        
        # Entry: signal changes from 0 to 1
        if not in_position and current_signal == 1:
            in_position = True
            entry_idx = i
            entry_price = prices.iloc[i] if i < len(prices) else prices.iloc[-1]
            entry_date = dates[i] if i < len(dates) else dates[-1]
        
        # Exit: signal changes from 1 to 0
        elif in_position and next_signal == 0:
            exit_idx = i + 1
            exit_price = prices.iloc[exit_idx] if exit_idx < len(prices) else prices.iloc[-1]
            exit_date = dates[exit_idx] if exit_idx < len(dates) else dates[-1]
            
            # Calculate P&L (assuming long positions only for now)
            pnl = exit_price - entry_price
            return_pct = (exit_price / entry_price - 1) * 100
            
            trades.append(TradeRecord(
                entry_timestamp=entry_date,
                exit_timestamp=exit_date,
                position_direction=1,  # Long position
                entry_price=entry_price,
                exit_price=exit_price,
                pnl=pnl,
                return_pct=return_pct
            ))
            
            in_position = False
    
    # Handle open position at the end
    if in_position and entry_idx is not None:
        exit_idx = len(signals) - 1
        exit_price = prices.iloc[-1]
        exit_date = dates[-1]
        pnl = exit_price - entry_price
        return_pct = (exit_price / entry_price - 1) * 100
        
        trades.append(TradeRecord(
            entry_timestamp=entry_date,
            exit_timestamp=exit_date,
            position_direction=1,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            return_pct=return_pct
        ))
    
    return trades

