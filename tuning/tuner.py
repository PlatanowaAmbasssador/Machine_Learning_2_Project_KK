"""
Hyperparameter tuning with IR2-based model selection.
Uses KerasTuner for deep learning models (LSTM, CNN, ANN).
"""

import numpy as np
import polars as pl
from typing import Dict, Any, List, Tuple, Optional
import keras_tuner as kt
from tensorflow import keras

from models.base import BaseModel
from metrics.returns import get_strategy_returns
from metrics.ir2 import calculate_ir2_from_returns


def tune_model_random_search(
    model: BaseModel,
    train_data: pl.DataFrame,
    val_data: pl.DataFrame,
    feature_cols: list,
    target_col: str,
    hyperparameter_space: Dict[str, Any],
    max_trials: int = 20,
    use_keras_tuner: bool = True,
    project_name: str = "tuning",
    overwrite: bool = True,
    verbose: int = 0
) -> Tuple[List[Any], List[Dict[str, Any]], List[Dict[str, float]], Any]:
    """
    Perform random search hyperparameter tuning.
    
    Parameters:
    -----------
    model : BaseModel
        Model instance to tune
    train_data : pl.DataFrame
        Training data
    val_data : pl.DataFrame
        Validation data
    feature_cols : list
        Feature column names
    target_col : str
        Target column name
    hyperparameter_space : dict or callable
        Hyperparameter search space (defined in Main.ipynb)
        For KerasTuner: function that takes hp and returns a dict with kt hyperparameter objects
    max_trials : int
        Maximum number of trials
    use_keras_tuner : bool
        Whether to use KerasTuner (always True for deep learning models)
    project_name : str
        Project name for KerasTuner
    overwrite : bool
        Whether to overwrite existing tuner results
    verbose : int
        Verbosity level
        
    Returns:
    --------
    models : List[Model]
        List of trained models (one per trial)
    hyperparameters : List[dict]
        List of hyperparameter dicts (one per trial)
    metrics : List[dict]
        List of metric dicts with train_ir2 and val_ir2
    scaler : Scaler
        Fitted scaler (fit on training data, same for all models)
    """
    # All models use KerasTuner for hyperparameter tuning
    models, hyperparameters, metrics, scaler = _tune_with_kerastuner(
        model, train_data, val_data, feature_cols, target_col,
        hyperparameter_space, max_trials, project_name, overwrite, verbose
    )
    
    return models, hyperparameters, metrics, scaler


def _tune_with_kerastuner(
    model: BaseModel,
    train_data: pl.DataFrame,
    val_data: pl.DataFrame,
    feature_cols: list,
    target_col: str,
    hyperparameter_space: Dict[str, Any],
    max_trials: int,
    project_name: str,
    overwrite: bool,
    verbose: int
) -> Tuple[List[Any], List[Dict[str, Any]], List[Dict[str, float]], Any]:
    """Tune using KerasTuner. Returns models, hyperparameters, metrics, and scaler."""
    # Extract sequence length for input shape
    # If hyperparameter_space is a function, we need to call it with a dummy hp
    # to get seq_length. For now, assume it's in the dict or use default.
    if callable(hyperparameter_space):
        # Create a dummy hp to extract seq_length
        import keras_tuner as kt
        dummy_hp = kt.HyperParameters()
        hp_dict = hyperparameter_space(dummy_hp)
        seq_length = model.get_sequence_length(hp_dict)
    else:
        seq_length = model.get_sequence_length(hyperparameter_space)
    input_shape = (seq_length, len(feature_cols)) if seq_length > 1 else (len(feature_cols),)
    
    # Create build function for KerasTuner
    # hyperparameter_space should be a function that takes hp and returns a dict
    # OR a dict with fixed values (for non-KerasTuner models)
    def build_fn(hp):
        if callable(hyperparameter_space):
            # hyperparameter_space is a function that defines the search space
            hp_dict = hyperparameter_space(hp)
        else:
            # hyperparameter_space is a dict - use as fixed values
            # This shouldn't happen for KerasTuner, but handle it gracefully
            hp_dict = hyperparameter_space.copy()
        
        return model.build_model(hp_dict, input_shape)
    
    # Create tuner
    tuner = kt.RandomSearch(
        hypermodel=build_fn,
        objective="val_loss",  # Internal objective (we'll rerank by IR2)
        max_trials=max_trials,
        directory="kt_runs",
        project_name=project_name,
        overwrite=overwrite
    )
    
    # Prepare data
    from features.scaling import fit_scaler, transform_data
    from features.sequences import create_sequences
    
    X_train = train_data.select(feature_cols).to_numpy()
    y_train = train_data.select(target_col).to_numpy().ravel()
    X_val = val_data.select(feature_cols).to_numpy()
    y_val = val_data.select(target_col).to_numpy().ravel()
    
    scaler = fit_scaler(train_data, feature_cols)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    if seq_length > 1:
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, seq_length)
        X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, seq_length)
    else:
        X_train_seq, y_train_seq = X_train_scaled, y_train
        X_val_seq, y_val_seq = X_val_scaled, y_val
    
    # Get training hyperparameters (epochs, batch_size, patience)
    if callable(hyperparameter_space):
        import keras_tuner as kt
        dummy_hp = kt.HyperParameters()
        hp_dict = hyperparameter_space(dummy_hp)
        epochs = hp_dict.get("epochs", 50)
        batch_size = hp_dict.get("batch_size", 32)
        patience = hp_dict.get("patience", 10)
    else:
        epochs = hyperparameter_space.get("epochs", 50)
        batch_size = hyperparameter_space.get("batch_size", 32)
        patience = hyperparameter_space.get("patience", 10)
    
    # Search
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=0
        )
    ]
    
    tuner.search(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose
    )
    
    # Get all trials and evaluate by IR2
    models = []
    hyperparameters = []
    metrics = []
    
    # Get top models by val_loss (we'll rerank by IR2)
    top_k = min(max_trials, len(tuner.oracle.trials))
    candidate_models = tuner.get_best_models(num_models=top_k)
    candidate_hps = tuner.get_best_hyperparameters(num_trials=top_k)
    
    # Evaluate each candidate by IR2
    price_col = "close" if "close" in val_data.columns else "Close"
    val_prices = val_data.select(price_col).to_numpy().ravel()
    if seq_length > 1:
        val_prices = val_prices[seq_length:]
    
    for m, hp in zip(candidate_models, candidate_hps):
        # Get predictions
        val_preds = m.predict(X_val_seq, verbose=0).flatten()
        
        # Calculate IR2
        val_returns, _ = get_strategy_returns(val_preds, val_prices)
        val_ir2 = calculate_ir2_from_returns(val_returns)
        
        # Also calculate train IR2
        train_preds = m.predict(X_train_seq, verbose=0).flatten()
        train_prices = train_data.select(price_col).to_numpy().ravel()
        if seq_length > 1:
            train_prices = train_prices[seq_length:]
        train_returns, _ = get_strategy_returns(train_preds, train_prices)
        train_ir2 = calculate_ir2_from_returns(train_returns)
        
        # Calculate MSE and MAE for price prediction
        from metrics.regression import (
            calculate_mse, calculate_mae,
            calculate_price_predictions_from_probabilities
        )
        
        # Convert probability predictions to price predictions
        # Predictions at time t predict price at time t+1
        # For validation
        val_price_preds = calculate_price_predictions_from_probabilities(
            val_preds, val_prices, method="weighted"
        )
        # Compare predicted prices to actual next prices
        if len(val_prices) > 1 and len(val_price_preds) > 0:
            val_actual_next = val_prices[1:len(val_price_preds)+1]
            val_pred_for_comp = val_price_preds[:len(val_actual_next)]
            val_mse = calculate_mse(val_actual_next, val_pred_for_comp)
            val_mae = calculate_mae(val_actual_next, val_pred_for_comp)
        else:
            val_mse = calculate_mse(val_prices, val_price_preds)
            val_mae = calculate_mae(val_prices, val_price_preds)
        
        # For training
        train_price_preds = calculate_price_predictions_from_probabilities(
            train_preds, train_prices, method="weighted"
        )
        # Compare predicted prices to actual next prices
        if len(train_prices) > 1 and len(train_price_preds) > 0:
            train_actual_next = train_prices[1:len(train_price_preds)+1]
            train_pred_for_comp = train_price_preds[:len(train_actual_next)]
            train_mse = calculate_mse(train_actual_next, train_pred_for_comp)
            train_mae = calculate_mae(train_actual_next, train_pred_for_comp)
        else:
            train_mse = calculate_mse(train_prices, train_price_preds)
            train_mae = calculate_mae(train_prices, train_price_preds)
        
        models.append(m)
        hyperparameters.append(hp.values)
        metrics.append({
            "train_ir2": train_ir2,
            "val_ir2": val_ir2,
            "train_mse": train_mse,
            "val_mse": val_mse,
            "train_mae": train_mae,
            "val_mae": val_mae
        })
    
    # Return scaler (same for all models, fit on training data)
    return models, hyperparameters, metrics, scaler


def select_best_by_ir2(
    models: List[Any],
    hyperparameters: List[Dict[str, Any]],
    metrics: List[Dict[str, float]],
    selection_method: str = "val_ir2"
) -> Tuple[Any, Dict[str, Any], Dict[str, float]]:
    """
    Select best model by IR2 with configurable selection strategy.
    
    Parameters:
    -----------
    models : List[Model]
        List of trained models
    hyperparameters : List[dict]
        List of hyperparameter dicts
    metrics : List[dict]
        List of metric dicts with train_ir2 and val_ir2
    selection_method : str
        Selection method:
        - "val_ir2": Select by highest validation IR2 (default, fastest)
        - "val_ir2_with_gap_penalty": Select by val IR2, penalize large train-val gap (prevents overfitting)
        - "combined": Select by (val_ir2 + train_ir2) / 2 (balanced)
        
    Returns:
    --------
    best_model : Model
        Best model
    best_hyperparameters : dict
        Best hyperparameters
    best_metrics : dict
        Best metrics
    """
    if not models:
        return None, {}, {
            "train_ir2": 0.0,
            "val_ir2": 0.0,
            "train_mse": 0.0,
            "val_mse": 0.0,
            "train_mae": 0.0,
            "val_mae": 0.0
        }
    
    if selection_method == "val_ir2":
        # Simple: highest validation IR2
        best_idx = max(range(len(metrics)), key=lambda i: metrics[i]["val_ir2"])
    
    elif selection_method == "val_ir2_with_gap_penalty":
        # Select by val IR2, but penalize models with large train-val gap (overfitting)
        # Score = val_ir2 - penalty_factor * max(0, train_ir2 - val_ir2)
        penalty_factor = 0.5  # Penalize 50% of the gap
        scores = []
        for m in metrics:
            val_ir2 = m["val_ir2"]
            train_ir2 = m["train_ir2"]
            gap = max(0, train_ir2 - val_ir2)  # Only penalize if train > val
            score = val_ir2 - penalty_factor * gap
            scores.append(score)
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
    
    elif selection_method == "combined":
        # Average of train and val IR2
        scores = [
            (metrics[i]["train_ir2"] + metrics[i]["val_ir2"]) / 2
            for i in range(len(metrics))
        ]
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
    
    else:
        raise ValueError(f"Unknown selection_method: {selection_method}")
    
    return models[best_idx], hyperparameters[best_idx], metrics[best_idx]

