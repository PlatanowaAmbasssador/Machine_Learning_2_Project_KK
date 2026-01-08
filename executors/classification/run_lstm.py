#!/usr/bin/env python3
"""
Standalone script to run LSTM model walk-forward analysis for CLASSIFICATION.
Uses ETH-USD data and binary direction labels (up/down).
"""

import os
import sys

# Add project root to Python path
project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import polars as pl
import keras_tuner as kt

# Configure TensorFlow for optimal performance (minimal config for speed)
from utils.tf_config import configure_tensorflow
print("Configuring TensorFlow...")
has_gpu = configure_tensorflow(minimal=True)  # Minimal config = faster startup

from models import LSTMModel
from walk_forward import execute_walk_forward
from metrics.classification import create_binary_labels_from_prices
from utils.output_logger import OutputLogger, get_log_file_path, check_existing_log

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_TYPE = 'LSTM'
PROBLEM_TYPE = 'classification'

# ============================================================================
# SETUP OUTPUT LOGGING
# ============================================================================
checkpoint_dir = f"checkpoints/{PROBLEM_TYPE}/{MODEL_TYPE.lower()}"
log_file = get_log_file_path(checkpoint_dir, MODEL_TYPE, PROBLEM_TYPE)
log_exists = check_existing_log(log_file)

# Start logging (append if resuming)
logger = OutputLogger(log_file, append=log_exists)
logger.start()

# ============================================================================
# DATA LOADING
# ============================================================================
print("Loading ETH-USD data...")
PATH_TO_ETH = "./inputs/ETH-USD.csv"
df_ETH = pl.read_csv(PATH_TO_ETH)

df_ETH = df_ETH.with_columns(
    pl.col("date")
      .str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S%z", strict=False)
      .alias("date")
)

# Group data to hourly
df_ETH_hourly = df_ETH.group_by(
    pl.col("date").dt.truncate("1h")
).agg([
    pl.first("close").alias("close"),
    pl.first("high").alias("high"),
    pl.first("low").alias("low"),
    pl.first("open").alias("open"),
    pl.first("tradesDone").alias("tradesDone"),
    pl.first("volume").alias("volume"),
    pl.first("volumeNotional").alias("volumeNotional"),
    pl.first("ticker").alias("ticker"),
])

df_ETH = df_ETH_hourly.sort('date')

# ============================================================================
# CREATE BINARY LABELS FOR CLASSIFICATION
# ============================================================================
print("Creating binary direction labels...")
prices = df_ETH.select("close").to_numpy().ravel()
binary_labels = create_binary_labels_from_prices(prices)

# Add binary labels to dataframe (aligned with next period)
# Label at index i corresponds to price direction from i to i+1
df_ETH = df_ETH.with_columns([
    pl.Series("direction", np.concatenate([binary_labels, [0]]))  # Pad last value
])

# Remove last row since it doesn't have a next period
df_ETH = df_ETH.slice(0, len(df_ETH) - 1)


# Walk-forward parameters
train_bars = 5000
val_bars = 2500
test_bars = 1250
step_size = None  # Default: test_bars (non-overlapping test sets)
max_folds = None  # None = all folds

# Hyperparameter tuning
max_trials = 10  # Number of hyperparameter trials per fold

# Feature and target columns
feature_cols = ['close', 'high', 'low', 'open', 'tradesDone', 'volume', 'volumeNotional']
target_col = 'direction'  # Binary labels for classification

# Sort data by date
df_ETH = df_ETH.sort("date")

# ============================================================================
# HYPERPARAMETER SEARCH SPACE
# ============================================================================
def define_lstm_hyperparameter_space(hp):
    """
    Hyperparameter search space for ETH classification.
    """
    return {
        "seq_length": hp.Int("seq_length", 24, 168, step=24),  # Hourly: 1-7 days context
        "hidden_units": hp.Int("hidden_units", 50, 200, step=25),
        "num_layers": hp.Int("num_layers", 1, 3),
        "dropout_rate": hp.Float("dropout_rate", 0.2, 0.5, step=0.05),
        "bidirectional": hp.Boolean("bidirectional"),
        "learning_rate": hp.Float("learning_rate", 5e-4, 5e-3, sampling="log"),
        "dense_units": hp.Choice("dense_units", [32, 64, 128]),
        "l2_reg": hp.Choice("l2_reg", [0.0, 1e-5, 1e-4]),
        "batch_size": hp.Int("batch_size", 32, 128, step=32),
        "optimizer": hp.Choice("optimizer", ["adam"]),
        "activation": hp.Choice("activation", ["tanh"]),
        "epochs": 30,
        "patience": 5
    }

hyperparameter_space = define_lstm_hyperparameter_space
model = LSTMModel()
use_keras_tuner = True

print(f"Running {MODEL_TYPE} model for {PROBLEM_TYPE.upper()}")
print(f"Using KerasTuner: {use_keras_tuner}")
print(f"Max trials per fold: {max_trials}")
print(f"Max folds: {max_folds if max_folds else 'All'}")

# ============================================================================
# EXECUTE WALK-FORWARD ANALYSIS
# ============================================================================
checkpoint_dir = f"checkpoints/{PROBLEM_TYPE}/{MODEL_TYPE.lower()}"

results_df, fold_results, aggregated_signals, aggregated_predictions = execute_walk_forward(
    model=model,
    data=df_ETH,
    feature_cols=feature_cols,
    target_col=target_col,
    lookback_bars=train_bars,
    validation_bars=val_bars,
    testing_bars=test_bars,
    hyperparameter_space=hyperparameter_space,
    max_trials=max_trials,
    max_folds=max_folds,
    step_size=step_size,
    use_keras_tuner=use_keras_tuner,
    verbose=1,
    checkpoint_dir=checkpoint_dir,
    resume=True,
    retrain_on_train_val=True,
    model_selection_method="val_ir2_with_gap_penalty"  # Prevents overfitting
)

# ============================================================================
# SAVE COMPREHENSIVE RESULTS
# ============================================================================
output_dir = f"results/{PROBLEM_TYPE}/{MODEL_TYPE.lower()}"
os.makedirs(output_dir, exist_ok=True)

from utils.save_results import save_comprehensive_results

metrics = save_comprehensive_results(
    model_type=MODEL_TYPE,
    output_dir=output_dir,
    results_df=results_df,
    fold_results=fold_results,
    aggregated_predictions=aggregated_predictions,
    data=df_ETH,
    feature_cols=feature_cols,
    target_col=target_col,
    problem_type=PROBLEM_TYPE
)

# Print summary
print("\n" + "="*80)
print(f"{MODEL_TYPE} WALK-FORWARD RESULTS SUMMARY ({PROBLEM_TYPE.upper()})")
print("="*80)
print(results_df)
if metrics:
    print(f"\nAggregated Test IR2: {metrics['aggregated_test_ir2']:.4f}")
    print(f"Buy & Hold IR2: {metrics['buyhold_ir2']:.4f}")
    print(f"Average Test IR2: {metrics['average_test_ir2']:.4f}")
    print(f"Average Val IR2: {metrics['average_val_ir2']:.4f}")
    if 'average_test_accuracy' in metrics:
        print(f"Average Test Accuracy: {metrics['average_test_accuracy']:.4f}")
        print(f"Average Test F1: {metrics['average_test_f1']:.4f}")
    print(f"Strategy Total Return: {metrics['strategy_total_return']:.2%}")
    print(f"Buy & Hold Total Return: {metrics['buyhold_total_return']:.2%}")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.2f}%")
print("="*80)

# Stop logging
logger.stop()
print(f"\nOutput log saved to: {log_file}")
print(f"To convert to PDF, run: python utils/log_to_pdf.py {log_file}")

