"""
Enhanced data processing module for time series factor forecasting.
Supports multi-file training with proper temporal data handling and validation.
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import warnings
# Only suppress specific warnings that are known to be harmless
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, message='.*NumPy.*')
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import glob
from pathlib import Path
import os
from configs.config import config
from .rolling_scaler import TimeSeriesDataProcessor, validate_time_series_split

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiFileDataProcessor:
    """Multi-file data processor for time series training."""
    
    def __init__(self, config):
        self.config = config
        # Time series processor with rolling window normalization
        self.ts_processor = TimeSeriesDataProcessor(
            factor_scaler_type='robust',
            target_scaler_type='standard',
            window_size=getattr(config, 'scaler_window_size', 252),
            min_periods=getattr(config, 'scaler_min_periods', 30)
        )
        self.feature_dim = len(config.factor_columns)
        self.target_dim = len(config.target_columns)
        # Default to './data' if data_dir is missing in lightweight configs used by tests
        self.data_dir = Path(getattr(config, 'data_dir', './data'))
        
    def get_available_dates(self) -> List[str]:
        """Get list of available dates from parquet files."""
        pattern = "*.parquet"
        files = list(self.data_dir.glob(pattern))
        
        dates = []
        for file in files:
            try:
                date_str = file.stem
                # Try YYYY-MM-DD format first
                if len(date_str) == 10 and date_str[4] == '-' and date_str[7] == '-':
                    date = date_str
                    dates.append(date)
                # Try YYYYMMDD format
                elif len(date_str) == 8:
                    date = datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")
                    dates.append(date)
            except ValueError:
                continue
        
        dates.sort()
        return dates
    
    def filter_dates(self, dates: List[str], start_date: str, end_date: str) -> List[str]:
        """Filter dates based on start and end dates."""
        if start_date:
            dates = [d for d in dates if d >= start_date]
        if end_date:
            dates = [d for d in dates if d <= end_date]
        return dates
    
    def load_daily_data(self, date: str) -> pd.DataFrame:
        """Load data for a specific date."""
        # Try YYYY-MM-DD format first
        filename = date + ".parquet"
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            # Try YYYYMMDD format
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            filename = date_obj.strftime("%Y%m%d") + ".parquet"
            filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        df = pd.read_parquet(filepath)
        df = self._clean_data(df)
        df = self._preprocess_data(df, is_training=True)
        
        logger.info(f"Loaded {len(df)} records for {date}")
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Data cleaning"""
        # 1. Remove limit up/down data
        if (self.config.limit_up_down_column and 
            self.config.limit_up_down_column in df.columns):
            df = df[df[self.config.limit_up_down_column] != 1]
            logger.info(f"Data shape after removing limit up/down: {df.shape}")
        
        # 2. Handle missing values
        factor_cols = self.config.factor_columns
        df.loc[:, factor_cols] = df[factor_cols].ffill().bfill()
        
        target_cols = self.config.target_columns
        df = df.dropna(subset=target_cols)
        
        # 3. Handle infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        return df
    
    def _preprocess_data(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Data preprocessing with rolling window normalization."""
        
        df_processed = self.ts_processor.process_daily_data(
            df=df,
            factor_columns=self.config.factor_columns,
            target_columns=self.config.target_columns,
            update_scalers=is_training
        )
        
        logger.debug(f"Processed {len(df_processed)} records with rolling window scaling")
        return df_processed

    def load_and_preprocess(self, file_path: str) -> pd.DataFrame:
        """Load a parquet file, clean and preprocess it for training/inference.
        Compatible with tests expecting a simple one-shot preprocessing entrypoint.
        """
        df = pd.read_parquet(file_path)
        # Ensure required columns exist or minimal defaults
        if 'date' not in df.columns:
            df['date'] = pd.to_datetime('2018-01-01')
        if 'sid' not in df.columns:
            # create a dummy single stock id if missing
            df['sid'] = 0
        df = self._clean_data(df)
        df = self._preprocess_data(df, is_training=True)
        return df
    
    def build_sequences(self, df: pd.DataFrame) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Build time series sequences"""
        sequences_X = []
        sequences_y = []
        
        # Group by stock
        for sid, group in df.groupby(self.config.stock_id_column):
            if len(group) < self.config.min_sequence_length:
                continue
                
            # Sort by time
            group = group.sort_index()
            
            # Build sliding window sequences
            for i in range(len(group) - self.config.sequence_length - self.config.prediction_horizon + 1):
                seq_X = group.iloc[i:i + self.config.sequence_length][self.config.factor_columns].values
                seq_y = group.iloc[i + self.config.sequence_length:i + self.config.sequence_length + self.config.prediction_horizon][self.config.target_columns].values
                
                if not np.isnan(seq_X).any() and not np.isnan(seq_y).any():
                    sequences_X.append(seq_X)
                    sequences_y.append(seq_y)
        
        return sequences_X, sequences_y
    
# Multi-file data processing and streaming data loader implementation

class MultiFileDataset(Dataset):
    """Multi-file dataset for training with three targets"""
    
    def __init__(self, dataframes: List[pd.DataFrame], config, mode: str = 'train'):
        self.config = config
        self.mode = mode
        self.sequences = []
        
        # Load sequences from all dataframes
        for df in dataframes:
            sequences_X, sequences_y = self._build_sequences_from_df(df)
            self.sequences.extend(list(zip(sequences_X, sequences_y)))
        
        logger.info(f"Loaded {len(self.sequences)} sequences for {mode}")
    
    def _build_sequences_from_df(self, df: pd.DataFrame) -> Tuple[List[Dict], List[Dict[str, np.ndarray]]]:
        """Build sequences from dataframe with three targets"""
        sequences_X = []
        sequences_y = []
        
        # Group by stock ID
        for sid, group in df.groupby(self.config.stock_id_column):
            if len(group) < self.config.min_sequence_length:
                continue
            
            # Sort by date
            group = group.sort_values('date')
            
            # Build sliding window sequences
            for i in range(len(group) - self.config.sequence_length - self.config.prediction_horizon + 1):
                # Features (factors)
                seq_X = group.iloc[i:i + self.config.sequence_length][self.config.factor_columns].values
                
                # Targets (three targets)
                seq_y = {}
                for target_name in self.config.target_columns:
                    target_values = group.iloc[i + self.config.sequence_length:i + self.config.sequence_length + self.config.prediction_horizon][target_name].values
                    # If prediction_horizon is 1, take the first (and only) value
                    if self.config.prediction_horizon == 1:
                        seq_y[target_name] = target_values[0]  # Single scalar value
                    else:
                        seq_y[target_name] = target_values  # Array of values
                
                # Stock IDs
                stock_ids = group.iloc[i:i + self.config.sequence_length][self.config.stock_id_column].values
                
                if not np.isnan(seq_X).any() and all(not np.isnan(seq_y[target]).any() for target in seq_y):
                    sequences_X.append({
                        'features': seq_X,
                        'stock_ids': stock_ids
                    })
                    sequences_y.append(seq_y)
        
        return sequences_X, sequences_y
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        data_X, data_y = self.sequences[idx]
        
        features = torch.FloatTensor(data_X['features'])
        
        # Convert string stock IDs to integers
        stock_ids = data_X['stock_ids']
        if hasattr(stock_ids, 'dtype') and stock_ids.dtype.kind in {'U', 'O'}:  # String or object type
            stock_ids = pd.factorize(stock_ids)[0]
        stock_ids = torch.LongTensor(stock_ids)
        
        # Handle targets - they might be scalars or arrays
        targets = {}
        for target in self.config.target_columns:
            target_value = data_y[target]
            if isinstance(target_value, (int, float, np.number)):
                # Scalar value
                targets[target] = torch.FloatTensor([target_value])
            else:
                # Array value
                targets[target] = torch.FloatTensor(target_value)
        
        if self.mode == 'train' and self.config.use_data_augmentation:
            features = self._augment_data(features)
        
        return {
            'features': features,
            'stock_ids': stock_ids,
            'targets': targets
        }
    
    def _augment_data(self, x: torch.Tensor) -> torch.Tensor:
        """Data augmentation"""
        if self.config.noise_std > 0:
            noise = torch.randn_like(x) * self.config.noise_std
            x = x + noise
        
        if self.config.mask_probability > 0:
            mask = torch.rand_like(x) > self.config.mask_probability
            x = x * mask
        
        return x


# Streaming datasets for efficient data processing

class DataManager:
    """Enhanced data manager for multi-file time series processing."""
    
    def __init__(self, config):
        self.config = config
        self.multi_file_processor = MultiFileDataProcessor(config)
        self.datasets = {}
        self.dataloaders = {}
        
    def prepare_training_data(self) -> Dict[str, DataLoader]:
        """Prepare training data loaders using multi-file mode"""
        logger.info("Preparing training data using multi-file mode...")
        
        # Get available dates
        available_dates = self.multi_file_processor.get_available_dates()
        filtered_dates = self.multi_file_processor.filter_dates(
            available_dates, self.config.start_date, self.config.end_date
        )
        
        if len(filtered_dates) == 0:
            raise ValueError("No data files found in specified date range")
        
        # Split dates for train/val/test
        train_dates, val_dates, test_dates = self._split_dates(filtered_dates)
        
        # Create datasets
        train_dfs = [self.multi_file_processor.load_daily_data(date) for date in train_dates]
        val_dfs = [self.multi_file_processor.load_daily_data(date) for date in val_dates]
        test_dfs = [self.multi_file_processor.load_daily_data(date) for date in test_dates]
        train_dataset = MultiFileDataset(train_dfs, self.config, mode='train')
        val_dataset = MultiFileDataset(val_dfs, self.config, mode='val')
        test_dataset = MultiFileDataset(test_dfs, self.config, mode='test')
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True if self.config.device == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True if self.config.device == 'cuda' else False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True if self.config.device == 'cuda' else False
        )
        
        self.dataloaders = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
        
        logger.info(f"Training data prepared:")
        logger.info(f"  Train: {len(train_dataset)} samples from {len(train_dates)} days")
        logger.info(f"  Validation: {len(val_dataset)} samples from {len(val_dates)} days")
        logger.info(f"  Test: {len(test_dataset)} samples from {len(test_dates)} days")
        
        return self.dataloaders
    
    # Validation data preparation integrated into prepare_training_data method
    
    def _split_dates(self, dates: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """Split dates into train/val/test with temporal order preservation."""
        
        # Ensure dates are sorted chronologically
        sorted_dates = sorted(dates)
        n_dates = len(sorted_dates)
        
        train_end = int(n_dates * self.config.train_ratio)
        val_end = train_end + int(n_dates * self.config.val_ratio)
        
        train_dates = sorted_dates[:train_end]
        val_dates = sorted_dates[train_end:val_end]
        test_dates = sorted_dates[val_end:]
        
        # Validate temporal split integrity
        if not validate_time_series_split(train_dates, val_dates, test_dates):
            raise ValueError("Invalid time series split detected!")
        
        logger.info(f"Time series split validation:")
        logger.info(f"  Train: {train_dates[0] if train_dates else 'None'} to {train_dates[-1] if train_dates else 'None'} ({len(train_dates)} days)")
        logger.info(f"  Val:   {val_dates[0] if val_dates else 'None'} to {val_dates[-1] if val_dates else 'None'} ({len(val_dates)} days)")
        logger.info(f"  Test:  {test_dates[0] if test_dates else 'None'} to {test_dates[-1] if test_dates else 'None'} ({len(test_dates)} days)")
        
        return train_dates, val_dates, test_dates
    
    def get_scalers(self):
        """Get time series processor and scaler states."""
        return {
            'ts_processor': self.multi_file_processor.ts_processor,
            'scaler_states': self.multi_file_processor.ts_processor.get_scaler_states()
        }

def create_training_dataloaders(config) -> Tuple[Dict[str, DataLoader], Dict]:
    """Create training data loaders using selected mode (quantitative or legacy multi-file)"""
    # Convert dict config to object format if needed
    if isinstance(config, dict):
        import types
        config = types.SimpleNamespace(**config)
    
    # Check if using new quantitative approach
    if hasattr(config, 'training_mode') and config.training_mode == "quantitative":
        logger.info("Using quantitative finance data processor...")
        
        # Import quantitative processor
        from src.data_processing.quantitative_data_processor import (
            create_quantitative_dataloaders, QuantitativeDataConfig
        )
        
        # Create quantitative config
        quant_config = QuantitativeDataConfig(
            train_start_date=getattr(config, 'train_start_date', '2018-01-01'),
            train_end_date=getattr(config, 'train_end_date', '2021-12-31'),
            val_start_date=getattr(config, 'val_start_date', '2022-01-01'),
            val_end_date=getattr(config, 'val_end_date', '2022-06-30'),
            test_start_date=getattr(config, 'test_start_date', '2022-07-01'),
            test_end_date=getattr(config, 'test_end_date', '2022-12-31'),
            sequence_length=getattr(config, 'sequence_length', 20),
            prediction_horizon=getattr(config, 'prediction_horizon', 1),
            min_sequence_length=getattr(config, 'min_sequence_length', 5),
            factor_columns=getattr(config, 'factor_columns', [str(i) for i in range(100)]),
            target_columns=getattr(config, 'target_columns', ['intra30m', 'nextT1d', 'ema1d']),
            stock_id_column=getattr(config, 'stock_id_column', 'sid'),
            weight_column=getattr(config, 'weight_column', 'ADV50'),
            min_stock_history_days=getattr(config, 'min_stock_history_days', 252),
            max_missing_ratio=getattr(config, 'max_missing_ratio', 0.1),
            remove_limit_up_down=getattr(config, 'remove_limit_up_down', True),
            remove_suspended=getattr(config, 'remove_suspended', True),
            batch_size=getattr(config, 'batch_size', 512),
            num_workers=getattr(config, 'num_workers', 4),
            pin_memory=getattr(config, 'device', 'cpu') == 'cuda'
        )
        
        # Create quantitative dataloaders
        train_loader, val_loader, test_loader = create_quantitative_dataloaders(
            config.data_dir, quant_config
        )
        
        dataloaders = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
        
        # Create dummy scalers for compatibility
        scalers = {
            'ts_processor': None,
            'scaler_states': {}
        }
        
        return dataloaders, scalers
    
    else:
        # Use legacy multi-file approach
        logger.info("Using legacy multi-file data processor...")
        data_manager = DataManager(config)
        dataloaders = data_manager.prepare_training_data()
        scalers = data_manager.get_scalers()
        return dataloaders, scalers

# Multi-file validation data loading using create_training_dataloaders
