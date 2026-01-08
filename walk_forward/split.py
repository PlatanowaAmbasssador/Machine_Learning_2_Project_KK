"""
Walk-forward splitting with strict temporal separation.
"""

import polars as pl
from typing import Generator, Tuple, Union


def walk_forward_split(
    data: Union[pl.DataFrame, 'pd.DataFrame'],
    lookback_bars: int,
    validation_bars: int,
    testing_bars: int,
    step_size: int = None
) -> Generator[Tuple[int, slice, slice, slice, pl.DataFrame, pl.DataFrame, pl.DataFrame], None, None]:
    """
    Generator function that yields training, validation, and testing indices for walk-forward analysis.

    Supports:
    - polars.DataFrame (uses .slice)
    - pandas.DataFrame / Series (uses .iloc)

    Yields:
    -------
    fold_num : int
    train_idx : slice
    val_idx : slice
    test_idx : slice
    train_data : same type as input
    val_data : same type as input
    test_data : same type as input
    """
    # Default: step by testing window size to avoid overlapping test sets
    if step_size is None:
        step_size = testing_bars

    # Length handling for Polars vs Pandas
    n = data.height if isinstance(data, pl.DataFrame) else len(data)

    # Calculate how many folds we can fit
    max_start = n - (validation_bars + testing_bars) + 1

    # fold starts = where validation begins
    fold_starts = list(range(lookback_bars, max_start, step_size))

    if len(fold_starts) == 0:
        print("Warning: No valid folds can be created with the given parameters.")
        return

    for fold_num, val_start_idx in enumerate(fold_starts, start=1):
        # Training window
        train_start_idx = val_start_idx - lookback_bars
        train_end_idx = val_start_idx

        # Validation window
        val_end_idx = val_start_idx + validation_bars

        # Testing window
        test_start_idx = val_end_idx
        test_end_idx = val_end_idx + testing_bars

        # Create slices (works for both)
        train_idx = slice(train_start_idx, train_end_idx)
        val_idx = slice(val_start_idx, val_end_idx)
        test_idx = slice(test_start_idx, test_end_idx)

        # Extract subsets
        train_data = data.slice(train_start_idx, lookback_bars)
        val_data = data.slice(val_start_idx, validation_bars)
        test_data = data.slice(test_start_idx, testing_bars)

        yield fold_num, train_idx, val_idx, test_idx, train_data, val_data, test_data

