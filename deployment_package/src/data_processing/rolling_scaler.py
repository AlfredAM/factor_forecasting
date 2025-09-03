"""
Rolling normalization for time series data processing.
Provides temporal-aware scaling using only historical data.
"""
import numpy as np
import pandas as pd
from collections import deque
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class RollingStandardScaler:
    """Rolling standard scaler using only historical data."""
    
    def __init__(self, window_size: int = 252, min_periods: int = 30):
        """
        Args:
            window_size: Rolling window size in trading days
            min_periods: Minimum historical data requirement
        """
        self.window_size = window_size
        self.min_periods = min_periods
        self.history = deque(maxlen=window_size)
        self._mean = None
        self._std = None
        
    def partial_fit(self, X: np.ndarray) -> 'RollingStandardScaler':
        """Update historical data buffer."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Add to historical record
        self.history.append(X.copy())
        
        # Recalculate statistics
        if len(self.history) >= self.min_periods:
            historical_data = np.vstack(list(self.history))
            self._mean = np.mean(historical_data, axis=0)
            self._std = np.std(historical_data, axis=0)
            # Avoid division by zero
            self._std = np.where(self._std == 0, 1.0, self._std)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using historical statistics."""
        if self._mean is None or self._std is None:
            logger.warning("Scaler not fitted with sufficient data, returning original values")
            return X.copy()
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        return (X - self._mean) / self._std
    
    def fit_transform(self, X: np.ndarray, update_history: bool = True) -> np.ndarray:
        """Fit and transform data."""
        if update_history:
            self.partial_fit(X)
        return self.transform(X)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        return {
            'mean': self._mean.copy() if self._mean is not None else None,
            'std': self._std.copy() if self._std is not None else None,
            'history_size': len(self.history),
            'is_fitted': self._mean is not None
        }


class RollingRobustScaler:
    """Rolling robust scaler using median and IQR."""
    
    def __init__(self, window_size: int = 252, min_periods: int = 30):
        self.window_size = window_size
        self.min_periods = min_periods
        self.history = deque(maxlen=window_size)
        self._median = None
        self._scale = None
        
    def partial_fit(self, X: np.ndarray) -> 'RollingRobustScaler':
        """Update historical data buffer."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        self.history.append(X.copy())
        
        if len(self.history) >= self.min_periods:
            historical_data = np.vstack(list(self.history))
            self._median = np.median(historical_data, axis=0)
            
            # Calculate interquartile range
            q75 = np.percentile(historical_data, 75, axis=0)
            q25 = np.percentile(historical_data, 25, axis=0)
            self._scale = q75 - q25
            self._scale = np.where(self._scale == 0, 1.0, self._scale)
            
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using historical statistics."""
        if self._median is None or self._scale is None:
            logger.warning("Robust scaler not fitted with sufficient data, returning original values")
            return X.copy()
            
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        return (X - self._median) / self._scale
    
    def fit_transform(self, X: np.ndarray, update_history: bool = True) -> np.ndarray:
        """Fit and transform data."""
        if update_history:
            self.partial_fit(X)
        return self.transform(X)


class TimeSeriesDataProcessor:
    """Time series data processor ensuring no data leakage."""
    
    def __init__(self, 
                 factor_scaler_type: str = 'robust',
                 target_scaler_type: str = 'standard',
                 window_size: int = 252,
                 min_periods: int = 30):
        """
        Args:
            factor_scaler_type: Factor scaler type ('standard' or 'robust')
            target_scaler_type: Target scaler type
            window_size: Rolling window size
            min_periods: Minimum historical data requirement
        """
        self.window_size = window_size
        self.min_periods = min_periods
        
        # Create scalers
        if factor_scaler_type == 'standard':
            self.factor_scaler = RollingStandardScaler(window_size, min_periods)
        elif factor_scaler_type == 'robust':
            self.factor_scaler = RollingRobustScaler(window_size, min_periods)
        else:
            raise ValueError(f"Unsupported scaler type: {factor_scaler_type}")
            
        if target_scaler_type == 'standard':
            self.target_scaler = RollingStandardScaler(window_size, min_periods)
        elif target_scaler_type == 'robust':
            self.target_scaler = RollingRobustScaler(window_size, min_periods)
        else:
            raise ValueError(f"Unsupported scaler type: {target_scaler_type}")
    
    def process_daily_data(self, 
                          df: pd.DataFrame,
                          factor_columns: list,
                          target_columns: list,
                          update_scalers: bool = True) -> pd.DataFrame:
        """
        Process daily data ensuring temporal integrity.
        
        Args:
            df: Data to process
            factor_columns: Factor column names
            target_columns: Target column names
            update_scalers: Whether to update scaler history
            
        Returns:
            Processed data
        """
        df_processed = df.copy()
        
        # Process factor data
        if factor_columns:
            factor_data = df[factor_columns].values
            
            if update_scalers:
                # Scale with history, then update history
                factor_scaled = self.factor_scaler.transform(factor_data)
                self.factor_scaler.partial_fit(factor_data)
            else:
                # Only scale, don't update history
                factor_scaled = self.factor_scaler.transform(factor_data)
            
            df_processed[factor_columns] = factor_scaled
        
        # Process target data
        if target_columns:
            target_data = df[target_columns].values
            
            if update_scalers:
                target_scaled = self.target_scaler.transform(target_data)
                self.target_scaler.partial_fit(target_data)
            else:
                target_scaled = self.target_scaler.transform(target_data)
                
            df_processed[target_columns] = target_scaled
        
        return df_processed
    
    def get_scaler_states(self) -> Dict[str, Dict]:
        """Get scaler states."""
        return {
            'factor_scaler': self.factor_scaler.get_stats(),
            'target_scaler': self.target_scaler.get_stats()
        }
    
    def save_scalers(self, filepath: str):
        """Save scaler states."""
        import pickle
        scaler_data = {
            'factor_scaler': self.factor_scaler,
            'target_scaler': self.target_scaler,
            'window_size': self.window_size,
            'min_periods': self.min_periods
        }
        with open(filepath, 'wb') as f:
            pickle.dump(scaler_data, f)
        logger.info(f"Scalers saved to {filepath}")
    
    def load_scalers(self, filepath: str):
        """Load scaler states."""
        import pickle
        with open(filepath, 'rb') as f:
            scaler_data = pickle.load(f)
        
        self.factor_scaler = scaler_data['factor_scaler']
        self.target_scaler = scaler_data['target_scaler']
        self.window_size = scaler_data['window_size']
        self.min_periods = scaler_data['min_periods']
        logger.info(f"Scalers loaded from {filepath}")


def validate_time_series_split(train_dates: list, val_dates: list, test_dates: list) -> bool:
    """Validate time series split correctness."""
    
    # Check for temporal overlap
    train_set = set(train_dates)
    val_set = set(val_dates)
    test_set = set(test_dates)
    
    if train_set & val_set:
        logger.error("Train and validation sets have overlapping dates")
        return False
    
    if train_set & test_set:
        logger.error("Train and test sets have overlapping dates")
        return False
        
    if val_set & test_set:
        logger.error("Validation and test sets have overlapping dates")
        return False
    
    # Check temporal order
    if train_dates:
        max_train_date = max(train_dates)
        if val_dates and min(val_dates) <= max_train_date:
            logger.error("Validation dates should be after all training dates")
            return False
            
        if test_dates and min(test_dates) <= max_train_date:
            logger.error("Test dates should be after all training dates")
            return False
    
    if val_dates and test_dates:
        max_val_date = max(val_dates)
        if min(test_dates) <= max_val_date:
            logger.error("Test dates should be after all validation dates")
            return False
    
    logger.info("Time series split validation passed")
    return True