# Walk-Forward Trading System with Hyperparameter Tuning

A modular, extensible, and research-ready architecture for walk-forward trading system evaluation with hyperparameter tuning and IR2-based model selection.

## Architecture Overview

The codebase is organized into modular components:

```
Project/
├── models/              # Model definitions (LSTM, CNN, ANN)
│   ├── base.py         # Base model interface
│   ├── lstm_model.py
│   ├── cnn_model.py
│   └── ann_model.py
├── walk_forward/        # Walk-forward analysis
│   ├── split.py        # Walk-forward splitting
│   ├── visualization.py # Plotting utilities
│   └── execution.py    # Full pipeline execution
├── features/            # Feature engineering
│   ├── scaling.py      # Temporal scaling (no data leakage)
│   └── sequences.py     # Sequence creation
├── metrics/             # Trading metrics
│   ├── ir2.py          # IR2 calculation
│   └── returns.py      # Strategy returns
├── backtesting/         # Backtesting utilities
│   ├── backtest.py     # VectorBT backtesting
│   ├── trades.py       # Trade extraction
│   └── visualization.py # Equity curve plotting
├── tuning/              # Hyperparameter tuning
│   └── tuner.py        # IR2-based model selection
└── Main.ipynb          # Main execution notebook
```

## Key Features

### 1. Model-Agnostic Architecture
- All models implement a common `BaseModel` interface
- Easy switching between LSTM, CNN, and ANN
- Common methods: `build_model()`, `train_model()`, `predict()`

### 2. Hyperparameter Tuning
- Search spaces defined in `Main.ipynb` (not in model functions)
- Uses KerasTuner for all deep learning models
- Batch size is tunable as a hyperparameter

### 3. IR2-Based Model Selection
- Models optimize `val_loss` internally (standard ML loss)
- Final model selection based on validation IR2 (trading-aware)
- Stores train_IR2, val_IR2, and test_IR2 for each fold

### 4. Trade Persistence
- Stores entry/exit timestamps for each trade
- Records position direction (long/short)
- Aggregates trades across all test windows

### 5. Aggregated Backtesting
- Combines all test window signals into continuous series
- Performs final backtest over "all test data combined"
- Compares strategy vs Buy & Hold using VectorBT

### 6. No Data Leakage
- Scalers fit only on training data
- Validation and test use training scaler
- Strict temporal separation maintained

## Usage

### Basic Workflow

1. **Load and prepare data** (see `Main.ipynb` cells 0-5)

2. **Define hyperparameter search space** (cell 9):
   ```python
   def define_lstm_hyperparameter_space(hp):
       return {
           "seq_length": 60,
           "hidden_units": hp.Int("hidden_units", 32, 128, step=32),
           "batch_size": hp.Int("batch_size", 16, 128, step=16),
           # ... more hyperparameters
       }
   ```

3. **Select model and execute walk-forward** (cells 8-10):
   ```python
   model = LSTMModel()  # or CNNModel(), ANNModel()
   results_df, fold_results, aggregated_signals, aggregated_predictions = \
       execute_walk_forward(...)
   ```

4. **Analyze results**:
   - `results_df`: DataFrame with fold, train_IR2, val_IR2, test_IR2, hyperparameters
   - `fold_results`: Detailed results per fold including trades
   - `aggregated_predictions`: Combined predictions for final backtest

### Switching Models

Simply change `MODEL_TYPE` in `Main.ipynb`:
```python
MODEL_TYPE = 'LSTM'  # or 'CNN', 'ANN'
```

The system automatically:
- Selects the appropriate model class
- Uses the correct hyperparameter space
- Uses KerasTuner for hyperparameter tuning

## Results Structure

### Results DataFrame
Contains for each fold:
- `fold`: Fold number
- `train_IR2`: Training IR2
- `val_IR2`: Validation IR2 (used for model selection)
- `test_IR2`: Test IR2 (out-of-sample performance)
- `hyperparameters`: Selected hyperparameters (as string)

### Fold Results
Each `FoldResult` contains:
- `fold_num`: Fold number
- `train_ir2`, `val_ir2`, `test_ir2`: IR2 metrics
- `hyperparameters`: Selected hyperparameters (as dict)
- `test_predictions`: Predictions on test set
- `test_dates`: Dates for test predictions
- `test_trades`: List of `TradeRecord` objects

### Trade Records
Each `TradeRecord` contains:
- `entry_timestamp`: Entry time
- `exit_timestamp`: Exit time
- `position_direction`: 1 (long), -1 (short), 0 (flat)
- `entry_price`, `exit_price`: Trade prices
- `pnl`: Profit/loss
- `return_pct`: Return percentage

## Research Integrity

- **No data leakage**: Scalers fit only on training data
- **Temporal separation**: Strict walk-forward splitting
- **Reproducible**: All hyperparameters stored in results
- **Extensible**: Easy to add new models or metrics
- **Research-ready**: Suitable for Master's thesis

## Extending the System

### Adding a New Model

1. Create a new model class in `models/`:
   ```python
   class NewModel(BaseModel):
       def build_model(self, hyperparams, input_shape):
           # Build model architecture
           pass
       
       def train_model(self, train_data, val_data, ...):
           # Train and return model, scaler, metrics
           pass
       
       def predict(self, model, scaler, data, ...):
           # Generate predictions
           pass
       
       def get_sequence_length(self, hyperparams):
           # Return sequence length (1 for non-sequential)
           return 1
   ```

2. Add hyperparameter space function in `Main.ipynb`

3. Update model selection logic in `Main.ipynb`

### Adding New Metrics

1. Add metric calculation in `metrics/`
2. Update `train_model()` to compute and return new metrics
3. Update results DataFrame structure

## Dependencies

- `polars`: Fast DataFrame operations
- `tensorflow`: Deep learning models
- `keras_tuner`: Hyperparameter tuning for deep models
- `vectorbt`: Backtesting
- `plotly`: Visualization
- `numpy`, `pandas`: Numerical operations

## Notes

- The original `Functions_DSA_Model.py` is preserved for reference
- All new code follows the modular architecture
- Hyperparameter search spaces are defined in `Main.ipynb` for easy modification
- IR2-based selection ensures trading-aware model evaluation

