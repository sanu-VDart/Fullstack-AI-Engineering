"""
Engine state classifier for turbofan engines.

Classifies engines into health states: healthy, degrading, or critical
based on sensor readings and predicted RUL.
"""

import os
import numpy as np
import joblib
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from ..feature_config import STATE_LABELS, STATE_THRESHOLDS


class StateClassifier:
    """
    Engine state classifier using Random Forest.
    
    Classifies engine health into three states:
    - Healthy: RUL > 125 cycles (normal operation)
    - Degrading: 50 < RUL <= 125 cycles (maintenance recommended)
    - Critical: RUL <= 50 cycles (immediate attention required)
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 12,
        random_state: int = 42
    ):
        """
        Initialize the state classifier.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            random_state: Random seed for reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'  # Handle class imbalance
        )
        self.state_labels = STATE_LABELS
        self._is_fitted = False
    
    @staticmethod
    def rul_to_state(rul: np.ndarray) -> np.ndarray:
        """
        Convert RUL values to state labels.
        
        Args:
            rul: Array of RUL values
            
        Returns:
            Array of state labels
        """
        states = np.empty(len(rul), dtype=object)
        
        states[rul > STATE_THRESHOLDS['healthy']] = 'healthy'
        states[(rul > STATE_THRESHOLDS['degrading']) & (rul <= STATE_THRESHOLDS['healthy'])] = 'degrading'
        states[rul <= STATE_THRESHOLDS['degrading']] = 'critical'
        
        return states
    
    def train(self, X: np.ndarray, y_rul: np.ndarray) -> 'StateClassifier':
        """
        Train the state classifier.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y_rul: Target RUL values (used to derive state labels)
            
        Returns:
            Self for chaining
        """
        # Convert RUL to state labels
        y_state = self.rul_to_state(y_rul)
        
        self.model.fit(X, y_state)
        self._is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict engine state for given feature data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Predicted state labels (n_samples,)
        """
        if not self._is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get probability distribution over states.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary mapping state labels to probability arrays
        """
        if not self._is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        probs = self.model.predict_proba(X)
        classes = self.model.classes_
        
        return {
            state: probs[:, i] 
            for i, state in enumerate(classes)
        }
    
    def evaluate(self, X: np.ndarray, y_rul: np.ndarray) -> Dict:
        """
        Evaluate classifier performance.
        
        Args:
            X: Feature matrix
            y_rul: Ground truth RUL values
            
        Returns:
            Dictionary with evaluation metrics
        """
        y_true = self.rul_to_state(y_rul)
        y_pred = self.predict(X)
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_true, y_pred, labels=self.state_labels)
        }
    
    def get_state_description(self, state: str) -> str:
        """
        Get human-readable description of engine state.
        
        Args:
            state: State label
            
        Returns:
            Description string
        """
        descriptions = {
            'healthy': "Engine is operating normally. No immediate maintenance required. "
                      "Continue monitoring sensor readings.",
            'degrading': "Engine shows signs of degradation. Schedule maintenance within "
                        "the next 50-125 operational cycles. Monitor critical sensors closely.",
            'critical': "Engine requires immediate attention! RUL estimated at less than "
                       "50 cycles. Recommend grounding for inspection and maintenance."
        }
        return descriptions.get(state, "Unknown state")
    
    def save(self, filepath: str) -> None:
        """Save model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'is_fitted': self._is_fitted
        }, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'StateClassifier':
        """Load model from disk."""
        data = joblib.load(filepath)
        classifier = cls()
        classifier.model = data['model']
        classifier._is_fitted = data['is_fitted']
        return classifier
