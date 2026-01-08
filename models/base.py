"""
Base model interface for all trading models.
All models must implement this interface for consistency.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np
import polars as pl


class BaseModel(ABC):
    """
    Abstract base class for all trading models.
    
    All models must implement:
    - build_model: Create model architecture
    - train_model: Train on data and return model + scaler
    - predict: Generate predictions
    """
    
    def __init__(self, model_type: str):
        """
        Initialize base model.
        
        Parameters:
        -----------
        model_type : str
            Type of model (e.g., 'LSTM', 'CNN', 'ANN')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = None
    
    @abstractmethod
    def build_model(self, hyperparams: Dict[str, Any], input_shape: Tuple) -> Any:
        """
        Build model architecture from hyperparameters.
        
        Parameters:
        -----------
        hyperparams : dict
            Hyperparameter dictionary
        input_shape : tuple
            Input shape for the model
            
        Returns:
        --------
        model : Model object
            Compiled model ready for training
        """
        pass
    
    @abstractmethod
    def train_model(
        self,
        train_data: pl.DataFrame,
        val_data: pl.DataFrame,
        feature_cols: list,
        target_col: str,
        hyperparams: Dict[str, Any],
        verbose: int = 0
    ) -> Tuple[Any, Any, Dict[str, float]]:
        """
        Train model on training data and evaluate on validation.
        
        Parameters:
        -----------
        train_data : pl.DataFrame
            Training data
        val_data : pl.DataFrame
            Validation data
        feature_cols : list
            List of feature column names
        target_col : str
            Target column name
        hyperparams : dict
            Hyperparameters for training
        verbose : int
            Verbosity level
            
        Returns:
        --------
        model : Model object
            Trained model
        scaler : Scaler object
            Fitted scaler (fit only on training data)
        metrics : dict
            Dictionary with train_ir2 and val_ir2
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        model: Any,
        scaler: Any,
        data: pl.DataFrame,
        feature_cols: list,
        target_col: str
    ) -> np.ndarray:
        """
        Generate predictions on data.
        
        Parameters:
        -----------
        model : Model object
            Trained model
        scaler : Scaler object
            Fitted scaler
        data : pl.DataFrame
            Data to predict on
        feature_cols : list
            List of feature column names
        target_col : str
            Target column name (for alignment, not used in prediction)
            
        Returns:
        --------
        predictions : np.ndarray
            Model predictions
        """
        pass
    
    @abstractmethod
    def get_sequence_length(self, hyperparams: Dict[str, Any]) -> int:
        """
        Get sequence length required by this model.
        Returns 1 for non-sequential models (ANN).
        
        Parameters:
        -----------
        hyperparams : dict
            Hyperparameters
            
        Returns:
        --------
        seq_length : int
            Sequence length required
        """
        pass

