#!/usr/bin/env python3
"""
Standalone script to run LSTM model walk-forward analysis.
Can be run in parallel with other model scripts.
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
from utils.output_logger import OutputLogger, get_log_file_path, check_existing_log

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_TYPE = 'LSTM'
PROBLEM_TYPE = 'regression'

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
print("Loading data...")
PATH_TO_BTC = "./inputs/BTC-USD.csv"
df_BTC = pl.read_csv(PATH_TO_BTC)

df_BTC = df_BTC.with_columns(
    pl.col("date")
      .str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S%z", strict=False)
      .alias("date")
)

# Group data to hourly
df_BTC_hourly = df_BTC.group_by(
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

df_BTC = df_BTC_hourly.sort('date')

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
target_col = 'close'

# Sort data by date
df_BTC = df_BTC.sort("date")

# ============================================================================
# HYPERPARAMETER SEARCH SPACE
# ============================================================================
def define_lstm_hyperparameter_space(hp):
    """
    Research-backed search space for BTC hourly LSTM (e.g., Sakarya Univ study [web:14]).
    Tuned for RandomSearch(10 trials): broad ranges for key params like dropout (high for hourly),
    learning_rate (large), seq_length; includes optimizer/activation per BTC papers.
    """
    return {
        "seq_length": hp.Int("seq_length", 24, 168, step=24),  # Hourly: 1-7 days context effective [web:14]
        "hidden_units": hp.Int("hidden_units", 50, 200, step=25),  # Balanced for crypto volatility [web:14]
        "num_layers": hp.Int("num_layers", 1, 3),
        "dropout_rate": hp.Float("dropout_rate", 0.2, 0.5, step=0.05),  # Large for hourly BTC [web:14]
        "bidirectional": hp.Boolean("bidirectional"),
        "learning_rate": hp.Float("learning_rate", 5e-4, 5e-3, sampling="log"),  # Large rates best [web:14]
        "dense_units": hp.Choice("dense_units", [32, 64, 128]),
        "l2_reg": hp.Choice("l2_reg", [0.0, 1e-5, 1e-4]),
        "batch_size": hp.Int("batch_size", 32, 128, step=32),  # Efficient for 4000 train bars [memory:18]
        "optimizer": hp.Choice("optimizer", ["adam"]),  # RMSprop strong for BTC [web:14]
        "activation": hp.Choice("activation", ["tanh"]),  # Tanh superior for hourly [web:14]
        "epochs": 30,
        "patience": 5
    }

hyperparameter_space = define_lstm_hyperparameter_space
model = LSTMModel()
use_keras_tuner = True

print(f"Running {MODEL_TYPE} model")
print(f"Using KerasTuner: {use_keras_tuner}")
print(f"Max trials per fold: {max_trials}")
print(f"Max folds: {max_folds if max_folds else 'All'}")

# ============================================================================
# EXECUTE WALK-FORWARD ANALYSIS
# ============================================================================

results_df, fold_results, aggregated_signals, aggregated_predictions = execute_walk_forward(
    model=model,
    data=df_BTC,
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
output_dir = f"results/{MODEL_TYPE.lower()}"
os.makedirs(output_dir, exist_ok=True)

from utils.save_results import save_comprehensive_results

metrics = save_comprehensive_results(
    model_type=MODEL_TYPE,
    output_dir=output_dir,
    results_df=results_df,
    fold_results=fold_results,
    aggregated_predictions=aggregated_predictions,
    data=df_BTC,
    feature_cols=feature_cols,
    target_col=target_col,
    problem_type='regression'
)

# Print summary
print("\n" + "="*80)
print(f"{MODEL_TYPE} WALK-FORWARD RESULTS SUMMARY")
print("="*80)
print(results_df)
if metrics:
    print(f"\nAggregated Test IR2: {metrics['aggregated_test_ir2']:.4f}")
    print(f"Buy & Hold IR2: {metrics['buyhold_ir2']:.4f}")
    print(f"Average Test IR2: {metrics['average_test_ir2']:.4f}")
    print(f"Average Val IR2: {metrics['average_val_ir2']:.4f}")
    print(f"Strategy Total Return: {metrics['strategy_total_return']:.2%}")
    print(f"Buy & Hold Total Return: {metrics['buyhold_total_return']:.2%}")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.2f}%")
print("="*80)

# Stop logging
logger.stop()
print(f"\nOutput log saved to: {log_file}")
print(f"To convert to PDF, run: python utils/log_to_pdf.py {log_file}")

