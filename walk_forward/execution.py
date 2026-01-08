"""
Walk-forward execution pipeline with trade persistence and aggregated backtesting.
"""

import numpy as np
import pandas as pd
import polars as pl
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import os
import gc

from .split import walk_forward_split
from .checkpoint import save_checkpoint, load_checkpoint
from models.base import BaseModel
from tuning.tuner import tune_model_random_search, select_best_by_ir2
from backtesting.trades import extract_trades, TradeRecord
from backtesting.backtest import vectorbt_backtest, aggregate_signals
from backtesting.visualization import plot_equity_comparison


@dataclass
class FoldResult:
    """Results for a single fold."""
    fold_num: int
    train_ir2: float
    val_ir2: float
    test_ir2: float
    train_mse: float = 0.0
    val_mse: float = 0.0
    test_mse: float = 0.0
    train_mae: float = 0.0
    val_mae: float = 0.0
    test_mae: float = 0.0
    hyperparameters: Dict[str, Any] = None
    test_predictions: np.ndarray = None
    test_dates: pd.DatetimeIndex = None
    test_trades: List[TradeRecord] = None


def execute_walk_forward(
    model: BaseModel,
    data: pl.DataFrame,
    feature_cols: list,
    target_col: str,
    lookback_bars: int,
    validation_bars: int,
    testing_bars: int,
    hyperparameter_space: Dict[str, Any],
    max_trials: int = 20,
    max_folds: Optional[int] = None,
    step_size: Optional[int] = None,
    use_keras_tuner: bool = True,
    verbose: int = 0,
    checkpoint_dir: Optional[str] = "checkpoints",
    resume: bool = True,
    retrain_on_train_val: bool = True,
    model_selection_method: str = "val_ir2_with_gap_penalty"
) -> Tuple[pd.DataFrame, List[FoldResult], pd.Series, pd.Series]:
    """
    Execute full walk-forward analysis with hyperparameter tuning and trade persistence.
    
    Parameters:
    -----------
    model : BaseModel
        Model instance to use
    data : pl.DataFrame
        Full dataset
    feature_cols : list
        Feature column names
    target_col : str
        Target column name
    lookback_bars : int
        Training window size
    validation_bars : int
        Validation window size
    testing_bars : int
        Testing window size
    hyperparameter_space : dict
        Hyperparameter search space (defined in Main.ipynb)
    max_trials : int
        Maximum hyperparameter trials per fold
    max_folds : int, optional
        Maximum number of folds to process
    step_size : int, optional
        Step size between folds (default: testing_bars)
    use_keras_tuner : bool
        Whether to use KerasTuner (always True for deep learning models)
    verbose : int
        Verbosity level
    checkpoint_dir : str, optional
        Directory to save/load checkpoints (default: "checkpoints")
    resume : bool
        Whether to resume from checkpoint if available (default: True)
    retrain_on_train_val : bool
        Whether to retrain best model on train+val combined before testing (default: True)
    model_selection_method : str
        Model selection method:
        - "val_ir2": Select by highest validation IR2 (fastest, may overfit)
        - "val_ir2_with_gap_penalty": Select by val IR2, penalize large train-val gap (default, prevents overfitting)
        - "combined": Select by average of train and val IR2 (balanced)
        
    Returns:
    --------
    results_df : pd.DataFrame
        Results DataFrame with fold, train_IR2, val_IR2, test_IR2, hyperparameters
    fold_results : List[FoldResult]
        Detailed results for each fold
    aggregated_signals : pd.Series
        Aggregated signals from all test windows
    aggregated_predictions : pd.Series
        Aggregated predictions from all test windows
    """
    # Try to load checkpoint if resume is enabled
    start_fold = 1
    results = []
    fold_results = []
    all_test_signals = []
    all_test_dates = []
    all_test_predictions = []
    
    if resume and checkpoint_dir:
        if verbose > 0:
            print(f"\nChecking for checkpoint in: {checkpoint_dir}")
        checkpoint = load_checkpoint(checkpoint_dir)
        if checkpoint:
            start_fold = checkpoint["last_completed_fold"] + 1
            results = checkpoint["results"]
            
            # Reconstruct FoldResult objects from serialized data
            fold_results = []
            for fr_dict in checkpoint["fold_results"]:
                try:
                    # Reconstruct test_dates
                    test_dates = pd.DatetimeIndex(fr_dict["test_dates"])
                    # Reconstruct test_predictions
                    test_predictions = np.array(fr_dict["test_predictions"])
                    # Reconstruct trades
                    from backtesting.trades import TradeRecord
                    test_trades = [
                        TradeRecord(
                            entry_timestamp=pd.Timestamp(trade["entry_timestamp"]),
                            exit_timestamp=pd.Timestamp(trade["exit_timestamp"]),
                            position_direction=trade["position_direction"],
                            entry_price=trade["entry_price"],
                            exit_price=trade["exit_price"],
                            pnl=trade["pnl"],
                            return_pct=trade["return_pct"]
                        )
                        for trade in fr_dict["test_trades"]
                    ]
                    
                    fold_result = FoldResult(
                        fold_num=fr_dict["fold_num"],
                        train_ir2=fr_dict["train_ir2"],
                        val_ir2=fr_dict["val_ir2"],
                        test_ir2=fr_dict["test_ir2"],
                        train_mse=fr_dict.get("train_mse", 0.0),
                        val_mse=fr_dict.get("val_mse", 0.0),
                        test_mse=fr_dict.get("test_mse", 0.0),
                        train_mae=fr_dict.get("train_mae", 0.0),
                        val_mae=fr_dict.get("val_mae", 0.0),
                        test_mae=fr_dict.get("test_mae", 0.0),
                        hyperparameters=fr_dict["hyperparameters"],
                        test_predictions=test_predictions,
                        test_dates=test_dates,
                        test_trades=test_trades
                    )
                    fold_results.append(fold_result)
                except Exception as e:
                    if verbose > 0:
                        print(f"Warning: Failed to reconstruct fold {fr_dict.get('fold_num', 'unknown')}: {e}")
                    continue
            
            all_test_signals = checkpoint.get("all_test_signals", [])
            all_test_predictions = checkpoint.get("all_test_predictions", [])
            all_test_dates = checkpoint.get("all_test_dates", [])
            
            if verbose > 0:
                print(f"\n{'='*80}")
                print(f"✓ RESUMING FROM CHECKPOINT")
                print(f"{'='*80}")
                print(f"Starting at fold {start_fold}")
                print(f"Already completed {len(results)} folds")
                print(f"Loaded {len(fold_results)} fold results")
                print(f"{'='*80}\n")
        else:
            if verbose > 0:
                print(f"No checkpoint found. Starting from fold 1.")
    
    # Get sequence length for alignment
    # Handle case where hyperparameter_space is a function (KerasTuner) or dict
    if callable(hyperparameter_space):
        # For KerasTuner, create a dummy hp object to extract seq_length
        import keras_tuner as kt
        dummy_hp = kt.HyperParameters()
        hp_dict = hyperparameter_space(dummy_hp)
        seq_length = model.get_sequence_length(hp_dict)
    else:
        # For dict (non-callable hyperparameter space)
        seq_length = model.get_sequence_length(hyperparameter_space)
    
    for fold_num, train_idx, val_idx, test_idx, train_data, val_data, test_data in \
        walk_forward_split(data, lookback_bars, validation_bars, testing_bars, step_size):
        
        # Skip folds already completed
        if fold_num < start_fold:
            continue
        
        if max_folds and fold_num > max_folds:
            break
        
        if verbose > 0:
            print(f"\n{'='*80}")
            print(f"Fold {fold_num} | Tuning (max_trials={max_trials})")
            print(f"{'='*80}")
        
        # Hyperparameter tuning
        models, hyperparameters, metrics, scaler = tune_model_random_search(
            model=model,
            train_data=train_data,
            val_data=val_data,
            feature_cols=feature_cols,
            target_col=target_col,
            hyperparameter_space=hyperparameter_space,
            max_trials=max_trials,
            use_keras_tuner=use_keras_tuner,
            project_name=f"{model.model_type.lower()}_fold_{fold_num}",
            overwrite=True,
            verbose=verbose
        )
        
        # Select best model by IR2
        best_model, best_hp, best_metrics = select_best_by_ir2(
            models, hyperparameters, metrics, selection_method=model_selection_method
        )
        
        if best_model is None:
            print(f"Warning: No valid model for fold {fold_num}")
            continue
        
        if scaler is None:
            print(f"Warning: No scaler returned from tuning for fold {fold_num}")
            continue
        
        # Ensure epochs and patience are included in best_hp (they might be fixed values)
        # KerasTuner only returns tunable hyperparameters, so we need to add fixed ones
        if callable(hyperparameter_space):
            import keras_tuner as kt
            dummy_hp = kt.HyperParameters()
            full_hp_dict = hyperparameter_space(dummy_hp)
            # Add fixed values (like epochs, patience) that might not be in best_hp
            for key, value in full_hp_dict.items():
                if key not in best_hp:
                    # Check if it's a fixed value (not a HyperParameter object)
                    try:
                        # If it's a HyperParameter object, skip it (it's tunable)
                        if isinstance(value, kt.engine.hyperparameters.HyperParameter):
                            continue
                    except:
                        pass
                    # It's a fixed value, add it
                    best_hp[key] = value
        else:
            # For dict-based hyperparameter spaces, add fixed values
            for key, value in hyperparameter_space.items():
                if key not in best_hp:
                    # Skip lists (they might be tunable parameter lists)
                    if not isinstance(value, list):
                        best_hp[key] = value
        
        train_ir2 = best_metrics["train_ir2"]
        val_ir2 = best_metrics["val_ir2"]
        
        if verbose > 0:
            print(f"Best val IR2: {val_ir2:.4f}")
            print(f"Best hyperparameters: {best_hp}")
        
        # Retrain on train+val combined if requested
        if retrain_on_train_val:
            if verbose > 0:
                print(f"Retraining best model on train+val combined...")
            
            # Combine train and validation data
            train_val_combined = pl.concat([train_data, val_data])
            
            # Retrain with best hyperparameters on combined data
            final_model, final_scaler, _ = model.train_model(
                train_val_combined, train_val_combined,  # Use same data for train/val (no validation split)
                feature_cols, target_col, best_hp, verbose=verbose
            )
            
            if final_model is None:
                print(f"Warning: Failed to retrain model for fold {fold_num}")
                continue
            
            # Clear old model from memory
            del best_model
            best_model = final_model
            scaler = final_scaler
            
            if verbose > 0:
                print(f"Model retrained on train+val combined")
        else:
            if verbose > 0:
                print(f"Using model from tuning (no retraining)")
        
        # Clear tuning models from memory
        del models
        gc.collect()
        
        # Clear Keras backend cache
        try:
            from utils.tf_config import clear_tf_session
            clear_tf_session()
        except:
            try:
                import tensorflow as tf
                tf.keras.backend.clear_session()
            except:
                pass
        
        # Evaluate on test set
        test_predictions = model.predict(
            best_model, scaler, test_data, feature_cols, target_col
        )
        
        # Clear model from memory after prediction
        del best_model
        gc.collect()
        
        # Calculate test IR2
        price_col = "close" if "close" in test_data.columns else "Close"
        test_prices = test_data.select(price_col).to_numpy().ravel()
        if seq_length > 1:
            test_prices = test_prices[seq_length:]
        
        # Align test_prices with test_predictions length
        # test_predictions length is determined by the model's sequence handling
        # Ensure they match before calculating returns
        min_len = min(len(test_predictions), len(test_prices))
        test_predictions = test_predictions[:min_len]
        test_prices = test_prices[:min_len]
        
        from metrics.returns import get_strategy_returns
        from metrics.ir2 import calculate_ir2_from_returns
        from metrics.regression import (
            calculate_mse, calculate_mae,
            calculate_price_predictions_from_probabilities
        )
        
        test_returns, test_signals = get_strategy_returns(test_predictions, test_prices)
        test_ir2 = calculate_ir2_from_returns(test_returns)
        
        # Calculate MSE and MAE for price prediction
        # Convert probability predictions to price predictions
        # Note: predictions at time t predict price at time t+1
        # So we compare predicted_prices[t] to actual_prices[t+1]
        test_price_preds = calculate_price_predictions_from_probabilities(
            test_predictions, test_prices, method="weighted"
        )
        
        # Compare predicted prices to actual next prices
        # test_price_preds[i] predicts test_prices[i+1]
        if len(test_prices) > 1 and len(test_price_preds) > 0:
            # Align: predicted_prices[:-1] should match actual_prices[1:]
            actual_next_prices = test_prices[1:len(test_price_preds)+1]
            predicted_for_comparison = test_price_preds[:len(actual_next_prices)]
            test_mse = calculate_mse(actual_next_prices, predicted_for_comparison)
            test_mae = calculate_mae(actual_next_prices, predicted_for_comparison)
        else:
            # Fallback: compare same-length arrays
            test_mse = calculate_mse(test_prices, test_price_preds)
            test_mae = calculate_mae(test_prices, test_price_preds)
        
        # Get test dates (aligned after sequence creation)
        # Note: test_signals has length len(test_predictions) - 1 due to get_strategy_returns
        # So we need to align dates accordingly
        test_dates = test_data.select("date").to_series().to_pandas()
        if seq_length > 1:
            test_dates = test_dates.iloc[seq_length:].reset_index(drop=True)
        
        # Align dates with signals (signals are one shorter due to returns calculation)
        # test_signals has length len(test_predictions) - 1, so we need to drop the last date
        if len(test_dates) > len(test_signals):
            test_dates = test_dates[:len(test_signals)]
        elif len(test_dates) < len(test_signals):
            # This shouldn't happen, but handle it gracefully
            test_signals = test_signals[:len(test_dates)]
        
        # Align test_prices with test_dates (they should have the same length now)
        if len(test_prices) > len(test_dates):
            test_prices = test_prices[:len(test_dates)]
        elif len(test_prices) < len(test_dates):
            # This shouldn't happen, but handle it gracefully
            test_dates = test_dates[:len(test_prices)]
            test_signals = test_signals[:len(test_prices)]
        
        test_dates = pd.DatetimeIndex(test_dates)
        
        # Extract trades
        test_prices_series = pd.Series(test_prices, index=test_dates)
        test_trades = extract_trades(test_predictions, test_prices_series, test_dates)
        
        if verbose > 0:
            print(f"Test IR2: {test_ir2:.4f}")
            print(f"Test MSE: {test_mse:.2f}")
            print(f"Test MAE: {test_mae:.2f}")
            print(f"Number of trades: {len(test_trades)}")
        
        # Store results
        results.append({
            "fold": fold_num,
            "train_IR2": train_ir2,
            "val_IR2": val_ir2,
            "test_IR2": test_ir2,
            "train_MSE": best_metrics.get("train_mse", 0.0),
            "val_MSE": best_metrics.get("val_mse", 0.0),
            "test_MSE": test_mse,
            "train_MAE": best_metrics.get("train_mae", 0.0),
            "val_MAE": best_metrics.get("val_mae", 0.0),
            "test_MAE": test_mae,
            "hyperparameters": str(best_hp)
        })
        
        fold_result = FoldResult(
            fold_num=fold_num,
            train_ir2=train_ir2,
            val_ir2=val_ir2,
            test_ir2=test_ir2,
            train_mse=best_metrics.get("train_mse", 0.0),
            val_mse=best_metrics.get("val_mse", 0.0),
            test_mse=test_mse,
            train_mae=best_metrics.get("train_mae", 0.0),
            val_mae=best_metrics.get("val_mae", 0.0),
            test_mae=test_mae,
            hyperparameters=best_hp,
            test_predictions=test_predictions,
            test_dates=test_dates,
            test_trades=test_trades
        )
        fold_results.append(fold_result)
        
        # Store signals and predictions for aggregation
        # Note: test_signals is already aligned with test_dates (we fixed that above)
        # For predictions, we need to align them with test_dates too
        test_predictions_aligned = test_predictions[:len(test_dates)]
        all_test_signals.append((test_dates, test_signals))
        all_test_predictions.append((test_dates, test_predictions_aligned))
        all_test_dates.append(test_dates)
        
        # Save checkpoint after each fold
        if checkpoint_dir:
            try:
                save_checkpoint(
                    checkpoint_dir,
                    fold_num,
                    results,
                    fold_results,
                    all_test_signals,
                    all_test_predictions,
                    all_test_dates
                )
                if verbose > 0:
                    print(f"✓ Checkpoint saved successfully for fold {fold_num}")
            except Exception as e:
                print(f"Warning: Failed to save checkpoint for fold {fold_num}: {e}")
                import traceback
                traceback.print_exc()
        
        # Clear variables to free memory
        del test_predictions, test_signals, test_prices, test_dates
        gc.collect()
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Aggregate all test signals
    aggregated_dates, aggregated_signals = aggregate_signals(all_test_signals)
    aggregated_signals_series = pd.Series(aggregated_signals, index=aggregated_dates)
    
    # Aggregate predictions
    aggregated_pred_dates, aggregated_preds = aggregate_signals(all_test_predictions)
    aggregated_predictions_series = pd.Series(aggregated_preds, index=aggregated_pred_dates)
    
    return results_df, fold_results, aggregated_signals_series, aggregated_predictions_series

