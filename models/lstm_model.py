"""
LSTM model implementation following BaseModel interface.
"""

import numpy as np
import polars as pl
from typing import Dict, Any, Tuple
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from .base import BaseModel
from features.scaling import fit_scaler, transform_data
from features.sequences import create_sequences
from metrics.returns import get_strategy_returns
from metrics.ir2 import calculate_ir2_from_returns


class LSTMModel(BaseModel):
    """LSTM model for time series prediction."""
    
    def __init__(self):
        super().__init__("LSTM")
    
    def build_model(
        self,
        hyperparams: Dict[str, Any],
        input_shape: Tuple
    ) -> keras.Model:
        """Build LSTM model from hyperparameters."""
        hidden_units = hyperparams.get("hidden_units", 64)
        num_layers = hyperparams.get("num_layers", 1)
        dropout_rate = hyperparams.get("dropout_rate", 0.3)
        bidirectional = hyperparams.get("bidirectional", False)
        learning_rate = hyperparams.get("learning_rate", 1e-3)
        dense_units = hyperparams.get("dense_units", 0)
        l2_reg = hyperparams.get("l2_reg", 0.0)
        # New research-driven hyperparameters
        optimizer_name = hyperparams.get("optimizer", "adam")
        lstm_activation = hyperparams.get("activation", "tanh")
        
        reg = regularizers.l2(l2_reg) if l2_reg and l2_reg > 0 else None
        
        inp = keras.Input(shape=input_shape)
        x = inp
        
        for layer_i in range(num_layers):
            return_seq = layer_i < (num_layers - 1)
            
            lstm = layers.LSTM(
                hidden_units,
                return_sequences=return_seq,
                activation=lstm_activation,
                kernel_regularizer=reg,
                recurrent_regularizer=reg
            )
            
            x = layers.Bidirectional(lstm)(x) if bidirectional else lstm(x)
            x = layers.Dropout(dropout_rate)(x)
        
        if dense_units and dense_units > 0:
            x = layers.Dense(dense_units, activation="relu", kernel_regularizer=reg)(x)
            x = layers.Dropout(dropout_rate)(x)
        
        # Output (binary classification)
        out = layers.Dense(1, activation="sigmoid")(x)
        
        model = keras.Model(inp, out)
        
        # Select optimizer based on hyperparameters
        if optimizer_name.lower() == "rmsprop":
            optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            # Default to Adam
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        
        return model
    
    def train_model(
        self,
        train_data: pl.DataFrame,
        val_data: pl.DataFrame,
        feature_cols: list,
        target_col: str,
        hyperparams: Dict[str, Any],
        verbose: int = 0
    ) -> Tuple[Any, Any, Dict[str, float]]:
        """Train LSTM model and return model, scaler, and metrics."""
        # Extract features and targets
        X_train = train_data.select(feature_cols).to_numpy()
        y_train = train_data.select(target_col).to_numpy().ravel()
        X_val = val_data.select(feature_cols).to_numpy()
        y_val = val_data.select(target_col).to_numpy().ravel()
        
        # Scale (fit only on training data)
        scaler = fit_scaler(train_data, feature_cols)
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Get sequence length
        seq_length = self.get_sequence_length(hyperparams)
        
        # Create sequences
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, seq_length)
        X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, seq_length)
        
        if len(X_train_seq) == 0 or len(X_val_seq) == 0:
            return None, None, {
                "train_ir2": 0.0,
                "val_ir2": 0.0,
                "train_mse": 0.0,
                "val_mse": 0.0,
                "train_mae": 0.0,
                "val_mae": 0.0
            }
        
        # Build model
        input_shape = (seq_length, len(feature_cols))
        model = self.build_model(hyperparams, input_shape)
        
        # Training callbacks
        patience = hyperparams.get("patience", 10)
        epochs = hyperparams.get("epochs", 50)
        batch_size = hyperparams.get("batch_size", 32)
        
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
                verbose=verbose
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=max(2, patience // 2),
                min_lr=1e-7,
                verbose=verbose
            )
        ]
        
        # Train
        # If train_data and val_data are the same (combined data), don't use validation split
        use_validation = not (len(train_data) == len(val_data) and 
                             train_data.equals(val_data))
        
        if use_validation:
            model.fit(
                X_train_seq, y_train_seq,
                validation_data=(X_val_seq, y_val_seq),
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose,
                callbacks=callbacks
            )
        else:
            # For combined train+val, use a small validation split instead
            model.fit(
                X_train_seq, y_train_seq,
                validation_split=0.1,  # Use 10% for validation
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose,
                callbacks=callbacks
            )
        
        # Calculate IR2 metrics
        train_preds = model.predict(X_train_seq, verbose=0).flatten()
        val_preds = model.predict(X_val_seq, verbose=0).flatten()
        
        # Get prices for returns calculation
        price_col = "close" if "close" in train_data.columns else "Close"
        train_prices = train_data.select(price_col).to_numpy().ravel()[seq_length:]
        val_prices = val_data.select(price_col).to_numpy().ravel()[seq_length:]
        
        train_returns, _ = get_strategy_returns(train_preds, train_prices)
        val_returns, _ = get_strategy_returns(val_preds, val_prices)
        
        train_ir2 = calculate_ir2_from_returns(train_returns)
        val_ir2 = calculate_ir2_from_returns(val_returns)
        
        # Calculate MSE and MAE for price prediction
        from metrics.regression import (
            calculate_mse, calculate_mae,
            calculate_price_predictions_from_probabilities
        )
        
        # Convert probability predictions to price predictions
        # Predictions at time t predict price at time t+1
        train_price_preds = calculate_price_predictions_from_probabilities(
            train_preds, train_prices, method="weighted"
        )
        val_price_preds = calculate_price_predictions_from_probabilities(
            val_preds, val_prices, method="weighted"
        )
        
        # Compare predicted prices to actual next prices
        # predicted_prices[i] should match actual_prices[i+1]
        if len(train_prices) > 1 and len(train_price_preds) > 0:
            train_actual_next = train_prices[1:len(train_price_preds)+1]
            train_pred_for_comp = train_price_preds[:len(train_actual_next)]
            train_mse = calculate_mse(train_actual_next, train_pred_for_comp)
            train_mae = calculate_mae(train_actual_next, train_pred_for_comp)
        else:
            train_mse = calculate_mse(train_prices, train_price_preds)
            train_mae = calculate_mae(train_prices, train_price_preds)
        
        if len(val_prices) > 1 and len(val_price_preds) > 0:
            val_actual_next = val_prices[1:len(val_price_preds)+1]
            val_pred_for_comp = val_price_preds[:len(val_actual_next)]
            val_mse = calculate_mse(val_actual_next, val_pred_for_comp)
            val_mae = calculate_mae(val_actual_next, val_pred_for_comp)
        else:
            val_mse = calculate_mse(val_prices, val_price_preds)
            val_mae = calculate_mae(val_prices, val_price_preds)
        
        return model, scaler, {
            "train_ir2": train_ir2,
            "val_ir2": val_ir2,
            "train_mse": train_mse,
            "val_mse": val_mse,
            "train_mae": train_mae,
            "val_mae": val_mae
        }
    
    def predict(
        self,
        model: Any,
        scaler: Any,
        data: pl.DataFrame,
        feature_cols: list,
        target_col: str
    ) -> np.ndarray:
        """Generate predictions on data."""
        X = data.select(feature_cols).to_numpy()
        X_scaled = scaler.transform(X)
        
        # Get sequence length from model input shape
        seq_length = model.input_shape[1]
        
        # Create sequences
        X_seq, _ = create_sequences(X_scaled, np.zeros(len(X_scaled)), seq_length)
        
        if len(X_seq) == 0:
            return np.array([])
        
        predictions = model.predict(X_seq, verbose=0).flatten()
        return predictions
    
    def get_sequence_length(self, hyperparams: Dict[str, Any]) -> int:
        """Get sequence length from hyperparameters."""
        return hyperparams.get("seq_length", 60)

