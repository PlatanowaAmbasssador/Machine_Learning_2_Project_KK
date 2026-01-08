"""
Backtesting and trade persistence utilities.
"""

from .backtest import vectorbt_backtest, aggregate_signals
from .trades import extract_trades, TradeRecord
from .visualization import plot_equity_comparison

__all__ = [
    'vectorbt_backtest',
    'aggregate_signals',
    'extract_trades',
    'TradeRecord',
    'plot_equity_comparison'
]

