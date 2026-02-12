"""
RUL (Remaining Useful Life) prediction model for turbofan engines.

Uses Random Forest regression as the primary model for predicting
remaining operational cycles before engine failure.
"""

import os
import numpy as np
import joblib
from typing import Dict, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class RULPredictor:
    """
    Remaining Useful Life predictor using Random Forest.
    
    Predicts the number of remaining operational cycles before
    an engine requires maintenance or fails.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 15,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        random_state: int = 42
    ):
        """
        Initialize the RUL predictor.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at leaf node
            random_state: Random seed for reproducibility
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1  # Use all CPU cores
        )
        self._is_fitted = False
    
    def train(self, X: np.ndarray, y: np.ndarray) -> 'RULPredictor':
        """
        Train the RUL prediction model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target RUL values (n_samples,)
            
        Returns:
            Self for chaining
        """
        self.model.fit(X, y)
        self._is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict RUL for given feature data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Predicted RUL values (n_samples,)
        """
        if not self._is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        predictions = self.model.predict(X)
        # Ensure non-negative predictions
        return np.maximum(predictions, 0)
    
    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Feature matrix
            y_true: Ground truth RUL values
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(X)
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Scoring function from PHM08 challenge
        # Penalizes late predictions more than early predictions
        s_score = self._compute_s_score(y_true, y_pred)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            's_score': s_score
        }
    
    def _compute_s_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute asymmetric scoring function from PHM08 challenge.
        
        Late predictions (underestimating RUL) are penalized more heavily
        than early predictions (overestimating RUL).
        """
        d = y_pred - y_true
        
        score = 0
        for diff in d:
            if diff < 0:
                # Late prediction (more dangerous)
                score += np.exp(-diff / 13) - 1
            else:
                # Early prediction
                score += np.exp(diff / 10) - 1
        
        return score
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance scores.
        
        Returns:
            Array of feature importance values
        """
        if not self._is_fitted:
            raise ValueError("Model must be trained first")
        
        return self.model.feature_importances_
    
    def save(self, filepath: str) -> None:
        """
        Save model to disk.
        
        Args:
            filepath: Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'is_fitted': self._is_fitted
        }, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'RULPredictor':
        """
        Load model from disk.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded RULPredictor instance
        """
        data = joblib.load(filepath)
        predictor = cls()
        predictor.model = data['model']
        predictor._is_fitted = data['is_fitted']
        return predictor
