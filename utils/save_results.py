"""
Utilities for saving comprehensive model results including backtests and metrics.
"""

import os
import pickle
import pandas as pd
import numpy as np
import polars as pl
from typing import List, Dict, Any
import vectorbt as vbt
from backtesting.backtest import vectorbt_backtest
from backtesting.trades import extract_trades


def save_comprehensive_results(
    model_type: str,
    output_dir: str,
    results_df: pd.DataFrame,
    fold_results: List[Any],
    aggregated_predictions: pd.Series,
    data: pl.DataFrame,
    feature_cols: list,
    target_col: str,
    problem_type: str = 'regression'
):
    """
    Save comprehensive results including backtests, equity curves, and trade statistics.
    
    Parameters:
    -----------
    model_type : str
        Model type (LSTM, CNN, ANN)
    output_dir : str
        Output directory
    results_df : pd.DataFrame
        Results DataFrame
    fold_results : List[FoldResult]
        Fold results
    aggregated_predictions : pd.Series
        Aggregated predictions
    data : pl.DataFrame
        Full dataset
    feature_cols : list
        Feature columns
    target_col : str
        Target column
    problem_type : str
        Problem type: 'regression' or 'classification' (default: 'regression')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save basic results
    results_df.to_csv(f"{output_dir}/results.csv", index=False)
    
    # 2. Save aggregated predictions
    aggregated_predictions.to_csv(f"{output_dir}/aggregated_predictions.csv")
    
    # 3. Perform aggregated backtest
    if len(aggregated_predictions) > 0:
        test_dates = aggregated_predictions.index
        first_test_date = test_dates[0]
        last_test_date = test_dates[-1]
        
        df_test_period = data.filter(
            (pl.col("date") >= first_test_date) & 
            (pl.col("date") <= last_test_date)
        )
        
        price_col = "close" if "close" in df_test_period.columns else "Close"
        test_prices_df = df_test_period.select(["date", price_col]).to_pandas()
        test_prices_df.set_index("date", inplace=True)
        test_prices = test_prices_df[price_col]
        
        # Align predictions with prices
        aligned_predictions = aggregated_predictions.reindex(test_prices.index, method='ffill').fillna(0.5)
        
        # Perform backtest
        strategy_pf, buyhold_pf = vectorbt_backtest(
            aligned_predictions.values,
            test_prices,
            threshold=0.5,
            initial_cash=1000000,
            commission=0.001
        )
        
        # 4. Calculate comprehensive metrics
        strategy_equity = strategy_pf.value()
        buyhold_equity = buyhold_pf.value()
        
        # Strategy metrics
        strategy_returns = strategy_pf.returns()
        strategy_total_return = strategy_pf.total_return()
        strategy_sharpe = strategy_pf.sharpe_ratio() if hasattr(strategy_pf, 'sharpe_ratio') else np.nan
        strategy_max_dd = strategy_pf.max_drawdown() if hasattr(strategy_pf, 'max_drawdown') else np.nan
        
        # Get trade statistics
        try:
            trades = strategy_pf.trades
            if hasattr(trades, 'records') and len(trades.records) > 0:
                strategy_win_rate = trades.win_rate()
                strategy_num_trades = len(trades.records)
            else:
                strategy_win_rate = np.nan
                strategy_num_trades = 0
        except:
            strategy_win_rate = np.nan
            strategy_num_trades = 0
        
        # Buy & Hold metrics
        buyhold_total_return = buyhold_pf.total_return()
        buyhold_sharpe = buyhold_pf.sharpe_ratio() if hasattr(buyhold_pf, 'sharpe_ratio') else np.nan
        buyhold_max_dd = buyhold_pf.max_drawdown() if hasattr(buyhold_pf, 'max_drawdown') else np.nan
        buyhold_returns = buyhold_pf.returns()
        
        # Calculate IR2 for aggregated period
        from metrics.ir2 import calculate_ir2_from_returns
        aggregated_ir2 = calculate_ir2_from_returns(strategy_returns.values)
        buyhold_ir2 = calculate_ir2_from_returns(buyhold_returns.values)
        
        # 5. Save equity curves
        equity_df = pd.DataFrame({
            'date': strategy_equity.index,
            'strategy_equity': strategy_equity.values,
            'buyhold_equity': buyhold_equity.values
        })
        equity_df.to_csv(f"{output_dir}/equity_curves.csv", index=False)
        
        # 6. Save returns
        returns_df = pd.DataFrame({
            'date': strategy_returns.index,
            'strategy_returns': strategy_returns.values,
            'buyhold_returns': buyhold_pf.returns().values
        })
        returns_df.to_csv(f"{output_dir}/returns.csv", index=False)
        
        # 7. Aggregate all trades
        all_trades = []
        for fold_result in fold_results:
            for trade in fold_result.test_trades:
                all_trades.append({
                    "fold": fold_result.fold_num,
                    "entry_timestamp": trade.entry_timestamp,
                    "exit_timestamp": trade.exit_timestamp,
                    "position_direction": trade.position_direction,
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "pnl": trade.pnl,
                    "return_pct": trade.return_pct
                })
        
        if all_trades:
            trades_df = pd.DataFrame(all_trades)
            trades_df.to_csv(f"{output_dir}/all_trades.csv", index=False)
            
            # Trade statistics
            trade_stats = {
                "total_trades": len(trades_df),
                "winning_trades": len(trades_df[trades_df['return_pct'] > 0]),
                "losing_trades": len(trades_df[trades_df['return_pct'] < 0]),
                "win_rate": (trades_df['return_pct'] > 0).sum() / len(trades_df) * 100,
                "avg_return_per_trade": trades_df['return_pct'].mean(),
                "avg_winning_trade": trades_df[trades_df['return_pct'] > 0]['return_pct'].mean() if (trades_df['return_pct'] > 0).any() else 0,
                "avg_losing_trade": trades_df[trades_df['return_pct'] < 0]['return_pct'].mean() if (trades_df['return_pct'] < 0).any() else 0,
                "largest_win": trades_df['return_pct'].max(),
                "largest_loss": trades_df['return_pct'].min(),
            }
        else:
            trade_stats = {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "avg_return_per_trade": 0,
                "avg_winning_trade": 0,
                "avg_losing_trade": 0,
                "largest_win": 0,
                "largest_loss": 0,
            }
        
        # 8. Save comprehensive metrics
        metrics = {
            "model_type": model_type,
            "aggregated_test_ir2": aggregated_ir2,
            "buyhold_ir2": buyhold_ir2,
            "strategy_total_return": float(strategy_total_return),
            "strategy_sharpe_ratio": float(strategy_sharpe) if not np.isnan(strategy_sharpe) else 0.0,
            "strategy_max_drawdown": float(strategy_max_dd) if not np.isnan(strategy_max_dd) else 0.0,
            "strategy_num_trades": int(strategy_num_trades),
            "buyhold_total_return": float(buyhold_total_return),
            "buyhold_sharpe_ratio": float(buyhold_sharpe) if not np.isnan(buyhold_sharpe) else 0.0,
            "buyhold_max_drawdown": float(buyhold_max_dd) if not np.isnan(buyhold_max_dd) else 0.0,
            "average_test_ir2": float(results_df['test_IR2'].mean()),
            "average_val_ir2": float(results_df['val_IR2'].mean()),
            "average_train_ir2": float(results_df['train_IR2'].mean()),
            "average_test_mse": float(results_df['test_MSE'].mean()) if 'test_MSE' in results_df.columns else 0.0,
            "average_val_mse": float(results_df['val_MSE'].mean()) if 'val_MSE' in results_df.columns else 0.0,
            "average_train_mse": float(results_df['train_MSE'].mean()) if 'train_MSE' in results_df.columns else 0.0,
            "average_test_mae": float(results_df['test_MAE'].mean()) if 'test_MAE' in results_df.columns else 0.0,
            "average_val_mae": float(results_df['val_MAE'].mean()) if 'val_MAE' in results_df.columns else 0.0,
            "average_train_mae": float(results_df['train_MAE'].mean()) if 'train_MAE' in results_df.columns else 0.0,
            "num_folds": len(results_df),
            **trade_stats
        }
        
        # Add classification metrics if problem_type is classification
        if problem_type == 'classification':
            from metrics.classification import calculate_classification_metrics
            
            # Calculate classification metrics for each fold and aggregate
            all_test_accuracies = []
            all_test_precisions = []
            all_test_recalls = []
            all_test_f1s = []
            
            for fold_result in fold_results:
                if hasattr(fold_result, 'test_predictions') and hasattr(fold_result, 'test_dates'):
                    # Get true labels for test period
                    test_dates = fold_result.test_dates
                    test_data_subset = data.filter(pl.col("date").is_in(test_dates))
                    
                    if target_col in test_data_subset.columns:
                        true_labels = test_data_subset.select(target_col).to_numpy().ravel()
                        pred_probs = fold_result.test_predictions
                        
                        # Align lengths
                        min_len = min(len(true_labels), len(pred_probs))
                        if min_len > 0:
                            true_labels_aligned = true_labels[:min_len]
                            pred_probs_aligned = pred_probs[:min_len]
                            
                            # Calculate metrics
                            fold_metrics = calculate_classification_metrics(
                                true_labels_aligned, pred_probs_aligned
                            )
                            all_test_accuracies.append(fold_metrics['accuracy'])
                            all_test_precisions.append(fold_metrics['precision'])
                            all_test_recalls.append(fold_metrics['recall'])
                            all_test_f1s.append(fold_metrics['f1'])
            
            # Add aggregated classification metrics
            if all_test_accuracies:
                metrics['average_test_accuracy'] = float(np.mean(all_test_accuracies))
                metrics['average_test_precision'] = float(np.mean(all_test_precisions))
                metrics['average_test_recall'] = float(np.mean(all_test_recalls))
                metrics['average_test_f1'] = float(np.mean(all_test_f1s))
            else:
                metrics['average_test_accuracy'] = 0.0
                metrics['average_test_precision'] = 0.0
                metrics['average_test_recall'] = 0.0
                metrics['average_test_f1'] = 0.0
        
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(f"{output_dir}/comprehensive_metrics.csv", index=False)
        
        # 9. Try to save portfolio objects (for dashboard)
        # Note: VectorBT portfolio objects may contain cached methods that can't be pickled
        # If pickling fails, we skip it since we already have all essential data saved
        try:
            with open(f"{output_dir}/strategy_portfolio.pkl", "wb") as f:
                pickle.dump(strategy_pf, f)
            portfolio_saved = True
        except Exception as e:
            print(f"Warning: Could not pickle strategy portfolio (this is OK, essential data is already saved): {e}")
            portfolio_saved = False
        
        try:
            with open(f"{output_dir}/buyhold_portfolio.pkl", "wb") as f:
                pickle.dump(buyhold_pf, f)
            buyhold_portfolio_saved = True
        except Exception as e:
            print(f"Warning: Could not pickle buyhold portfolio (this is OK, essential data is already saved): {e}")
            buyhold_portfolio_saved = False
        
        print(f"\nComprehensive results saved to {output_dir}/")
        print(f"  - results.csv")
        print(f"  - comprehensive_metrics.csv")
        print(f"  - equity_curves.csv")
        print(f"  - returns.csv")
        print(f"  - all_trades.csv")
        print(f"  - aggregated_predictions.csv")
        if portfolio_saved and buyhold_portfolio_saved:
            print(f"  - strategy_portfolio.pkl")
            print(f"  - buyhold_portfolio.pkl")
        else:
            print(f"  - (Portfolio objects skipped - can be recreated from saved data if needed)")
        
        return metrics
    else:
        print(f"Warning: No aggregated predictions to backtest")
        return None

