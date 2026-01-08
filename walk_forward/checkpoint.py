"""
Checkpoint and resume functionality for walk-forward analysis.
"""

import os
import pickle
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path


def save_checkpoint(
    checkpoint_dir: str,
    fold_num: int,
    results: List[Dict],
    fold_results: List[Any],
    all_test_signals: List,
    all_test_predictions: List,
    all_test_dates: List
):
    """
    Save checkpoint after completing a fold.
    
    Parameters:
    -----------
    checkpoint_dir : str
        Directory to save checkpoints
    fold_num : int
        Current fold number
    results : List[Dict]
        Results list so far
    fold_results : List[FoldResult]
        Fold results so far
    all_test_signals : List
        Test signals collected so far
    all_test_predictions : List
        Test predictions collected so far
    all_test_dates : List
        Test dates collected so far
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save results DataFrame
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(
            os.path.join(checkpoint_dir, "results.csv"),
            index=False
        )
    
    # Save fold results (without model objects - they're too large)
    fold_results_serializable = []
    for fr in fold_results:
        fr_dict = {
            "fold_num": fr.fold_num,
            "train_ir2": fr.train_ir2,
            "val_ir2": fr.val_ir2,
            "test_ir2": fr.test_ir2,
            "train_mse": getattr(fr, 'train_mse', 0.0),
            "val_mse": getattr(fr, 'val_mse', 0.0),
            "test_mse": getattr(fr, 'test_mse', 0.0),
            "train_mae": getattr(fr, 'train_mae', 0.0),
            "val_mae": getattr(fr, 'val_mae', 0.0),
            "test_mae": getattr(fr, 'test_mae', 0.0),
            "hyperparameters": fr.hyperparameters,
            "test_predictions": fr.test_predictions.tolist() if isinstance(fr.test_predictions, np.ndarray) else fr.test_predictions,
            "test_dates": fr.test_dates.tolist() if hasattr(fr.test_dates, 'tolist') else list(fr.test_dates),
            "test_trades": [
                {
                    "entry_timestamp": str(trade.entry_timestamp),
                    "exit_timestamp": str(trade.exit_timestamp),
                    "position_direction": trade.position_direction,
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "pnl": trade.pnl,
                    "return_pct": trade.return_pct
                }
                for trade in (fr.test_trades if fr.test_trades else [])
            ]
        }
        fold_results_serializable.append(fr_dict)
    
    with open(
        os.path.join(checkpoint_dir, "fold_results.pkl"),
        "wb"
    ) as f:
        pickle.dump(fold_results_serializable, f)
    
    # Save test signals and predictions
    with open(
        os.path.join(checkpoint_dir, "test_signals.pkl"),
        "wb"
    ) as f:
        pickle.dump(all_test_signals, f)
    
    with open(
        os.path.join(checkpoint_dir, "test_predictions.pkl"),
        "wb"
    ) as f:
        pickle.dump(all_test_predictions, f)
    
    with open(
        os.path.join(checkpoint_dir, "test_dates.pkl"),
        "wb"
    ) as f:
        pickle.dump(all_test_dates, f)
    
    # Save metadata
    metadata = {
        "last_completed_fold": fold_num,
        "num_folds_completed": len(results)
    }
    
    # Save log file path if provided
    import sys
    if hasattr(sys.stdout, 'log_file') or (hasattr(sys.stdout, 'file') and hasattr(sys.stdout.file, 'name')):
        try:
            if hasattr(sys.stdout, 'file'):
                log_file = sys.stdout.file.name
            else:
                log_file = None
            if log_file and os.path.exists(log_file):
                metadata["log_file"] = log_file
        except:
            pass
    
    with open(
        os.path.join(checkpoint_dir, "metadata.json"),
        "w"
    ) as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Checkpoint saved: Fold {fold_num} completed")


def load_checkpoint(checkpoint_dir: str) -> Optional[Dict[str, Any]]:
    """
    Load checkpoint if it exists.
    
    Parameters:
    -----------
    checkpoint_dir : str
        Directory containing checkpoints
        
    Returns:
    --------
    checkpoint_data : dict or None
        Dictionary with checkpoint data, or None if no checkpoint exists
    """
    metadata_path = os.path.join(checkpoint_dir, "metadata.json")
    
    if not os.path.exists(metadata_path):
        print(f"No checkpoint found at {checkpoint_dir}")
        return None
    
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        print(f"Found checkpoint: Last completed fold = {metadata.get('last_completed_fold', 'unknown')}")
        
        # Load results
        results_df_path = os.path.join(checkpoint_dir, "results.csv")
        results = []
        if os.path.exists(results_df_path):
            results_df = pd.read_csv(results_df_path)
            results = results_df.to_dict("records")
            print(f"Loaded {len(results)} completed folds from checkpoint")
        else:
            print(f"Warning: results.csv not found in checkpoint")
        
        # Load fold results
        fold_results_path = os.path.join(checkpoint_dir, "fold_results.pkl")
        fold_results = []
        if os.path.exists(fold_results_path):
            with open(fold_results_path, "rb") as f:
                fold_results = pickle.load(f)
            print(f"Loaded {len(fold_results)} fold result objects")
        else:
            print(f"Warning: fold_results.pkl not found in checkpoint")
        
        # Load test signals and predictions
        test_signals_path = os.path.join(checkpoint_dir, "test_signals.pkl")
        all_test_signals = []
        if os.path.exists(test_signals_path):
            with open(test_signals_path, "rb") as f:
                all_test_signals = pickle.load(f)
            print(f"Loaded {len(all_test_signals)} test signal sets")
        else:
            print(f"Warning: test_signals.pkl not found in checkpoint")
        
        test_predictions_path = os.path.join(checkpoint_dir, "test_predictions.pkl")
        all_test_predictions = []
        if os.path.exists(test_predictions_path):
            with open(test_predictions_path, "rb") as f:
                all_test_predictions = pickle.load(f)
            print(f"Loaded {len(all_test_predictions)} test prediction sets")
        else:
            print(f"Warning: test_predictions.pkl not found in checkpoint")
        
        test_dates_path = os.path.join(checkpoint_dir, "test_dates.pkl")
        all_test_dates = []
        if os.path.exists(test_dates_path):
            with open(test_dates_path, "rb") as f:
                all_test_dates = pickle.load(f)
            print(f"Loaded {len(all_test_dates)} test date sets")
        else:
            print(f"Warning: test_dates.pkl not found in checkpoint")
        
        return {
            "last_completed_fold": metadata["last_completed_fold"],
            "results": results,
            "fold_results": fold_results,
            "all_test_signals": all_test_signals,
            "all_test_predictions": all_test_predictions,
            "all_test_dates": all_test_dates
        }
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None

