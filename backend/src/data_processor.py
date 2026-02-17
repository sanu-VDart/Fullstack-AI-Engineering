"""
Data processing module for C-MAPSS turbofan engine dataset.

Handles loading, preprocessing, and feature engineering for the NASA C-MAPSS dataset.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List
from sklearn.preprocessing import StandardScaler

from .feature_config import (
    COLUMN_NAMES, FEATURE_COLUMNS, RUL_CAP, 
    STATE_THRESHOLDS, STATE_LABELS, ROLLING_WINDOWS
)


class CMAPSSDataLoader:
    """
    Data loader for C-MAPSS turbofan engine degradation dataset.
    
    Attributes:
        data_path: Path to the CMAPSSData directory
        dataset_id: Dataset identifier (FD001, FD002, FD003, FD004)
    """
    
    def __init__(self, data_path: str, dataset_id: str = "FD001"):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to CMAPSSData directory
            dataset_id: Which dataset to load (FD001-FD004)
        """
        self.data_path = data_path
        self.dataset_id = dataset_id
        self._validate_dataset_id()
    
    def _validate_dataset_id(self):
        """Validate that dataset_id is valid."""
        valid_ids = ['FD001', 'FD002', 'FD003', 'FD004']
        if self.dataset_id not in valid_ids:
            raise ValueError(f"dataset_id must be one of {valid_ids}, got {self.dataset_id}")
    
    def load_train_data(self) -> pd.DataFrame:
        """
        Load training data for the specified dataset.
        
        Returns:
            DataFrame with training data including RUL labels
        """
        file_path = os.path.join(self.data_path, f"train_{self.dataset_id}.txt")
        df = pd.read_csv(file_path, sep=r'\s+', header=None, names=COLUMN_NAMES)
        
        # Calculate RUL for training data
        df = self._add_rul(df)
        
        return df
    
    def load_test_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load test data and ground truth RUL values.
        
        Returns:
            Tuple of (test_data DataFrame, ground_truth RUL Series)
        """
        # Load test data
        test_path = os.path.join(self.data_path, f"test_{self.dataset_id}.txt")
        df = pd.read_csv(test_path, sep=r'\s+', header=None, names=COLUMN_NAMES)
        
        # Load ground truth RUL
        rul_path = os.path.join(self.data_path, f"RUL_{self.dataset_id}.txt")
        rul = pd.read_csv(rul_path, header=None, names=['rul'])['rul']
        
        return df, rul
    
    def _add_rul(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Remaining Useful Life (RUL) column to training data.
        
        For training data, RUL is calculated as:
        RUL = max_cycle - current_cycle for each engine unit.
        
        Args:
            df: Training DataFrame
            
        Returns:
            DataFrame with RUL column added
        """
        # Get max cycle for each unit (when engine fails)
        max_cycles = df.groupby('unit_id')['cycle'].max()
        
        # Calculate RUL
        df = df.copy()
        df['rul'] = df.apply(
            lambda row: max_cycles[row['unit_id']] - row['cycle'], 
            axis=1
        )
        
        return df


class FeatureEngineer:
    """
    Feature engineering for C-MAPSS data.
    
    Creates additional features from raw sensor data:
    - Rolling statistics (mean, std)
    - Degradation trend indicators
    - Time-based features
    """
    
    def __init__(self, window_sizes: List[int] = None):
        """
        Initialize feature engineer.
        
        Args:
            window_sizes: List of rolling window sizes
        """
        self.window_sizes = window_sizes or ROLLING_WINDOWS
    
    def create_features(self, df: pd.DataFrame, sensor_columns: List[str] = None) -> pd.DataFrame:
        """
        Create engineered features from raw data.
        
        Args:
            df: Input DataFrame with sensor data
            sensor_columns: List of sensor columns to use
            
        Returns:
            DataFrame with additional engineered features
        """
        df = df.copy()
        sensor_columns = sensor_columns or [c for c in df.columns if c.startswith('sensor_')]
        
        # Add rolling statistics for each unit
        for unit_id in df['unit_id'].unique():
            unit_mask = df['unit_id'] == unit_id
            
            for window in self.window_sizes:
                for col in sensor_columns:
                    # Rolling mean
                    roll_mean = df.loc[unit_mask, col].rolling(window=window, min_periods=1).mean()
                    df.loc[unit_mask, f'{col}_mean_{window}'] = roll_mean
                    
                    # Rolling std
                    roll_std = df.loc[unit_mask, col].rolling(window=window, min_periods=1).std()
                    df.loc[unit_mask, f'{col}_std_{window}'] = roll_std.fillna(0)
        
        # Add normalized cycle (percentage of typical lifespan)
        df['cycle_norm'] = df.groupby('unit_id')['cycle'].transform(
            lambda x: x / x.max()
        )
        
        return df
    
    def add_state_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add state classification labels based on RUL.
        
        Args:
            df: DataFrame with RUL column
            
        Returns:
            DataFrame with 'state' column added
        """
        df = df.copy()
        
        conditions = [
            df['rul'] > STATE_THRESHOLDS['healthy'],
            (df['rul'] > STATE_THRESHOLDS['degrading']) & (df['rul'] <= STATE_THRESHOLDS['healthy']),
            df['rul'] <= STATE_THRESHOLDS['degrading']
        ]
        
        df['state'] = np.select(conditions, STATE_LABELS, default='unknown')
        
        return df


class DataPreprocessor:
    """
    Data preprocessing pipeline for ML model training.
    
    Handles normalization, feature selection, and data splitting.
    """
    
    def __init__(self, rul_cap: int = RUL_CAP):
        """
        Initialize preprocessor.
        
        Args:
            rul_cap: Maximum RUL value (capping improves model performance)
        """
        self.rul_cap = rul_cap
        self.scaler = StandardScaler()
        self.feature_columns = None
        self._is_fitted = False
    
    def fit(self, df: pd.DataFrame, feature_columns: List[str] = None) -> 'DataPreprocessor':
        """
        Fit the preprocessor on training data.
        
        Args:
            df: Training DataFrame
            feature_columns: Columns to use as features
            
        Returns:
            Self for chaining
        """
        self.feature_columns = feature_columns or FEATURE_COLUMNS
        
        # Fit scaler on feature columns
        self.scaler.fit(df[self.feature_columns])
        self._is_fitted = True
        
        return self
    
    def transform(self, df: pd.DataFrame, include_rul: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transform data for model input.
        
        Args:
            df: Input DataFrame
            include_rul: Whether to include RUL target (for training)
            
        Returns:
            Tuple of (features array, RUL array or None)
        """
        if not self._is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Scale features
        X = self.scaler.transform(df[self.feature_columns])
        
        # Get RUL if available and requested
        y = None
        if include_rul and 'rul' in df.columns:
            y = df['rul'].values.copy()
            # Cap RUL values
            y = np.clip(y, 0, self.rul_cap)
        
        return X, y
    
    def fit_transform(self, df: pd.DataFrame, feature_columns: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit and transform training data.
        
        Args:
            df: Training DataFrame
            feature_columns: Columns to use as features
            
        Returns:
            Tuple of (features array, RUL array)
        """
        self.fit(df, feature_columns)
        return self.transform(df)


def prepare_training_data(data_path: str, dataset_id: str = "FD001") -> Dict:
    """
    Complete data preparation pipeline for model training.
    
    Args:
        data_path: Path to CMAPSSData directory
        dataset_id: Dataset to load (FD001-FD004)
        
    Returns:
        Dictionary containing prepared data and fitted objects
    """
    # Load data
    loader = CMAPSSDataLoader(data_path, dataset_id)
    train_df = loader.load_train_data()
    test_df, test_rul = loader.load_test_data()
    
    # Feature engineering
    feature_eng = FeatureEngineer()
    train_df = feature_eng.create_features(train_df)
    train_df = feature_eng.add_state_labels(train_df)
    test_df = feature_eng.create_features(test_df)
    
    # Get all feature columns (original + engineered)
    feature_columns = [c for c in train_df.columns 
                      if c not in ['unit_id', 'cycle', 'rul', 'state']]
    
    # Preprocess
    preprocessor = DataPreprocessor()
    X_train, y_train = preprocessor.fit_transform(train_df, feature_columns)
    
    # For test data, get last cycle of each unit (point of prediction)
    test_last_cycle = test_df.groupby('unit_id').last().reset_index()
    X_test, _ = preprocessor.transform(test_last_cycle, include_rul=False)
    y_test = test_rul.values
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'train_df': train_df,
        'test_df': test_df,
        'preprocessor': preprocessor,
        'feature_columns': feature_columns,
        'loader': loader,
        'feature_engineer': feature_eng
    }
