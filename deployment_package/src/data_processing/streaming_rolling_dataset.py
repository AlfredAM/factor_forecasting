#!/usr/bin/env python3
"""
StreamingRollingDataset: Hybrid dataset combining streaming data loading with rolling window training.
Optimizes for both memory efficiency and training performance.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import gc
import pyarrow.parquet as pq
from collections import OrderedDict
import warnings
# Only suppress specific warnings that are known to be harmless
warnings.filterwarnings('ignore', category=FutureWarning, module='pyarrow')
warnings.filterwarnings('ignore', category=UserWarning, message='.*NumPy.*')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamingRollingDataset(Dataset):
    """
    Hybrid dataset combining streaming data loading with rolling window training.
    Optimizes for both memory efficiency and training performance.
    """
    
    def __init__(self, data_loader, train_dates: List[str], 
                 test_dates: List[str], sequence_length: int = 20, 
                 prediction_horizon: int = 1, target_columns: List[str] = None,
                 cache_days: int = 5, batch_size: int = 50000,
                 enable_rolling_cache: bool = True, rolling_window_size: int = 3):
        """
        Initialize streaming rolling dataset.
        
        Args:
            data_loader: Daily data loader
            train_dates: Training dates
            test_dates: Test dates
            sequence_length: Length of input sequence
            prediction_horizon: Prediction horizon
            target_columns: Target columns to predict
            cache_days: Number of days to keep in LRU cache
            batch_size: PyArrow batch size for streaming
            enable_rolling_cache: Enable rolling window cache optimization
            rolling_window_size: Size of rolling window for cache optimization
        """
        self.data_loader = data_loader
        self.train_dates = train_dates
        self.test_dates = test_dates
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.target_columns = target_columns or ['intra30m', 'nextT1d', 'ema1d']
        self.cache_days = cache_days
        self.batch_size = batch_size
        self.enable_rolling_cache = enable_rolling_cache
        self.rolling_window_size = rolling_window_size
        
        # LRU cache for data (date -> DataFrame)
        self.data_cache = OrderedDict()
        
        # Rolling window cache for frequently accessed date ranges
        self.rolling_cache = OrderedDict()
        self.rolling_cache_max_size = rolling_window_size * 2
        
        # Global index: (date, offset, stock_id) -> sequence index
        self.global_index = self._create_global_index()
        
        # Preload critical training dates for better performance
        self._preload_critical_dates()
        
        logger.info(f"Created streaming rolling dataset with {len(self.global_index)} sequences")
        logger.info(f"Cache: {cache_days} days, Rolling cache: {rolling_window_size} windows")
    
    def _create_global_index(self) -> List[Tuple[str, int, int]]:
        """Create global index of all possible sequences."""
        global_index = []
        
        # Get all unique stock IDs from first few dates to estimate
        sample_dates = self.train_dates[:min(5, len(self.train_dates))]
        all_stocks = set()
        
        for date in sample_dates:
            try:
                # Use pyarrow to get metadata without loading full file
                date_obj = datetime.strptime(date, "%Y-%m-%d")
                filename = date_obj.strftime("%Y%m%d") + ".parquet"
                filepath = self.data_loader.data_dir / filename
                
                if filepath.exists():
                    # Get metadata only
                    parquet_file = pq.ParquetFile(filepath)
                    metadata = parquet_file.metadata
                    
                    # Estimate number of stocks
                    if metadata.num_rows > 0:
                        estimated_stocks = min(5000, metadata.num_rows // 100)
                        all_stocks.update(range(estimated_stocks))
                        
            except Exception as e:
                logger.warning(f"Failed to get metadata for {date}: {str(e)}")
                continue
        
        # Create sequence indices for training data
        for i in range(len(self.train_dates) - self.sequence_length - self.prediction_horizon + 1):
            for stock_id in sorted(all_stocks):
                global_index.append((self.train_dates[i], i, stock_id))
        
        # Create sequence indices for test data
        for i in range(len(self.test_dates) - self.sequence_length - self.prediction_horizon + 1):
            for stock_id in sorted(all_stocks):
                global_index.append((self.test_dates[i], i, stock_id))
        
        return global_index
    
    def _preload_critical_dates(self):
        """Preload critical training dates for better performance."""
        if not self.enable_rolling_cache:
            return
        
        # Preload first and last few dates of training data for better cache performance
        critical_dates = []
        
        # First few dates (for sequence start)
        critical_dates.extend(self.train_dates[:self.rolling_window_size])
        
        # Last few dates (for sequence end)
        if len(self.train_dates) > self.rolling_window_size:
            critical_dates.extend(self.train_dates[-self.rolling_window_size:])
        
        # Middle dates (for rolling window)
        if len(self.train_dates) > self.rolling_window_size * 2:
            mid_start = len(self.train_dates) // 2 - self.rolling_window_size // 2
            critical_dates.extend(self.train_dates[mid_start:mid_start + self.rolling_window_size])
        
        # Remove duplicates and limit
        critical_dates = list(set(critical_dates))[:self.rolling_window_size * 2]
        
        logger.info(f"Preloading critical dates: {critical_dates}")
        
        for date in critical_dates:
            try:
                self._load_date_data(date)
            except Exception as e:
                logger.warning(f"Failed to preload {date}: {str(e)}")
    
    def _load_date_data(self, date: str) -> pd.DataFrame:
        """
        Load data for a specific date with streaming and column projection.
        Enhanced with rolling window cache optimization.
        """
        # Check rolling cache first (for frequently accessed dates)
        if self.enable_rolling_cache and date in self.rolling_cache:
            self.rolling_cache.move_to_end(date)
            return self.rolling_cache[date]
        
        # Check regular cache
        if date in self.data_cache:
            self.data_cache.move_to_end(date)
            return self.data_cache[date]
        
        try:
            # Convert date to filename format
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            filename = date_obj.strftime("%Y%m%d") + ".parquet"
            filepath = self.data_loader.data_dir / filename
            
            if not filepath.exists():
                raise FileNotFoundError(f"Data file not found: {filepath}")
            
            # Use pyarrow streaming with column projection
            parquet_file = pq.ParquetFile(filepath)
            
            # Define required columns
            required_cols = ['sid'] + [f'factor_{i}' for i in range(100)] + self.target_columns
            
            # Filter columns that exist in the file
            available_cols = [col for col in required_cols if col in parquet_file.schema.names]
            
            # Load data in batches
            batches = []
            for batch in parquet_file.iter_batches(columns=available_cols, batch_size=self.batch_size):
                batch_df = batch.to_pandas()
                batches.append(batch_df)
            
            # Combine batches
            df = pd.concat(batches, ignore_index=True)
            
            # Add missing columns with zeros
            for col in required_cols:
                if col not in df.columns:
                    if col.startswith('factor_'):
                        df[col] = 0.0
                    elif col in self.target_columns:
                        df[col] = 0.0
            
            # Add date column
            df['date'] = pd.to_datetime(date)
            
            # Sort by stock ID
            df = df.sort_values(['sid']).reset_index(drop=True)
            
            # Filter limit-hit data if column exists
            if 'luld' in df.columns:
                df = df[df['luld'] != 1].copy()
            
            # Update appropriate cache
            self._update_cache(date, df)
            
            logger.debug(f"Streaming loaded {len(df)} records for {date}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data for {date}: {str(e)}")
            # Return empty DataFrame with correct structure
            empty_df = pd.DataFrame(columns=['sid', 'date'] + [f'factor_{i}' for i in range(100)] + self.target_columns)
            return empty_df
    
    def _update_cache(self, date: str, df: pd.DataFrame):
        """Update cache with new data, using rolling window optimization."""
        # Determine which cache to use based on access pattern
        if self.enable_rolling_cache and self._is_critical_date(date):
            # Use rolling cache for critical dates
            self.rolling_cache[date] = df
            
            # Remove oldest from rolling cache if full
            if len(self.rolling_cache) > self.rolling_cache_max_size:
                oldest_date = next(iter(self.rolling_cache))
                oldest_df = self.rolling_cache.pop(oldest_date)
                del oldest_df
                gc.collect()
        else:
            # Use regular LRU cache
            self.data_cache[date] = df
            
            # Remove oldest from regular cache if full
            if len(self.data_cache) > self.cache_days:
                oldest_date = next(iter(self.data_cache))
                oldest_df = self.data_cache.pop(oldest_date)
                del oldest_df
                gc.collect()
    
    def _is_critical_date(self, date: str) -> bool:
        """Check if a date is critical for rolling window training."""
        if not self.enable_rolling_cache:
            return False
        
        # Check if date is in critical ranges
        critical_ranges = [
            self.train_dates[:self.rolling_window_size],
            self.train_dates[-self.rolling_window_size:] if len(self.train_dates) > self.rolling_window_size else [],
        ]
        
        if len(self.train_dates) > self.rolling_window_size * 2:
            mid_start = len(self.train_dates) // 2 - self.rolling_window_size // 2
            critical_ranges.append(self.train_dates[mid_start:mid_start + self.rolling_window_size])
        
        return any(date in date_range for date_range in critical_ranges)
    
    def _get_stock_data_for_dates(self, dates: List[str], stock_id: int) -> List[pd.DataFrame]:
        """Get data for a specific stock across multiple dates."""
        stock_data = []
        
        for date in dates:
            df = self._load_date_data(date)
            if not df.empty:
                stock_df = df[df['sid'] == stock_id].copy()
                if not stock_df.empty:
                    stock_data.append(stock_df)
        
        return stock_data
    
    def __len__(self) -> int:
        return len(self.global_index)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single sequence with streaming data loading and rolling optimization."""
        date, offset, stock_id = self.global_index[idx]
        
        # Determine if this is training or test data
        if date in self.train_dates:
            date_list = self.train_dates
            start_idx = self.train_dates.index(date)
        else:
            date_list = self.test_dates
            start_idx = self.test_dates.index(date)
        
        # Get date range for this sequence
        input_dates = date_list[start_idx:start_idx + self.sequence_length]
        target_dates = date_list[start_idx + self.sequence_length:start_idx + self.sequence_length + self.prediction_horizon]
        
        # Load input sequence data
        input_data = self._get_stock_data_for_dates(input_dates, stock_id)
        
        # Load target data
        target_data = self._get_stock_data_for_dates(target_dates, stock_id)
        
        # Validate sequence length
        if len(input_data) < self.sequence_length or len(target_data) < self.prediction_horizon:
            # Return zero-padded sequence
            return self._create_zero_padded_sequence(stock_id, input_dates, target_dates)
        
        # Prepare input features
        factors_list = []
        stock_ids_list = []
        
        for df in input_data:
            # Extract factor columns
            factor_cols = [f'factor_{i}' for i in range(100)]
            factors = df[factor_cols].values
            
            # Shape fix
            if factors.shape[-1] < 100:
                pad = np.zeros((factors.shape[0], 100 - factors.shape[1]))
                factors = np.concatenate([factors, pad], axis=1)
            elif factors.shape[-1] > 100:
                factors = factors[:, :100]
            
            # Stock IDs
            stock_ids = df['sid'].values
            if hasattr(stock_ids, 'dtype') and stock_ids.dtype.kind in {'U', 'O'}:
                stock_ids = pd.factorize(stock_ids)[0]
            
            factors_list.append(factors)
            stock_ids_list.append(stock_ids)
        
        # Stack factors and stock IDs
        factors = np.stack(factors_list)  # (seq_len, num_stocks, num_factors)
        stock_ids = np.stack(stock_ids_list)  # (seq_len, num_stocks)
        
        # Prepare targets
        targets = {}
        for target_col in self.target_columns:
            target_values = []
            for df in target_data:
                if target_col in df.columns:
                    target_values.append(df[target_col].values)
                else:
                    # Fill with zeros if target not available
                    target_values.append(np.zeros(len(df)))
            
            targets[target_col] = np.stack(target_values)  # (horizon, num_stocks)
        
        return {
            'factors': torch.FloatTensor(factors),
            'stock_ids': torch.LongTensor(stock_ids),
            'targets': {k: torch.FloatTensor(v) for k, v in targets.items()},
            'sequence_info': {
                'stock_id': stock_id,
                'input_dates': [str(x) for x in input_dates],
                'target_dates': [str(x) for x in target_dates],
                'is_training': date in self.train_dates
            }
        }
    
    def _create_zero_padded_sequence(self, stock_id: int, input_dates: List[str], target_dates: List[str]) -> Dict:
        """Create zero-padded sequence when data is insufficient."""
        # Create zero tensors
        factors = torch.zeros(self.sequence_length, 1, 100)
        stock_ids = torch.full((self.sequence_length, 1), stock_id, dtype=torch.long)
        
        targets = {}
        for target_col in self.target_columns:
            targets[target_col] = torch.zeros(self.prediction_horizon, 1)
        
        return {
            'factors': factors,
            'stock_ids': stock_ids,
            'targets': targets,
            'sequence_info': {
                'stock_id': stock_id,
                'input_dates': [str(x) for x in input_dates],
                'target_dates': [str(x) for x in target_dates],
                'is_training': True
            }
        }
    
    def clear_cache(self):
        """Clear all caches to free memory."""
        self.data_cache.clear()
        self.rolling_cache.clear()
        gc.collect()
        logger.info("Cleared all caches")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics for monitoring."""
        return {
            'data_cache_size': len(self.data_cache),
            'rolling_cache_size': len(self.rolling_cache),
            'total_cached_dates': len(self.data_cache) + len(self.rolling_cache),
            'max_data_cache_size': self.cache_days,
            'max_rolling_cache_size': self.rolling_cache_max_size
        }
    
    def optimize_cache_for_rolling(self, current_train_years: List[int], next_test_year: int):
        """Optimize cache for rolling window training."""
        if not self.enable_rolling_cache:
            return
        
        # Preload data for next training window
        next_train_years = current_train_years[1:] + [next_test_year]
        
        for year in next_train_years:
            year_dates = [date for date in self.train_dates if date.startswith(str(year))]
            for date in year_dates[:self.rolling_window_size]:
                if date not in self.rolling_cache:
                    try:
                        self._load_date_data(date)
                        logger.debug(f"Preloaded {date} for rolling optimization")
                    except Exception as e:
                        logger.warning(f"Failed to preload {date}: {str(e)}")


class StreamingRollingDataLoader:
    """
    Data loader for rolling window training using streaming datasets.
    Combines the best of both approaches.
    """
    
    def __init__(self, data_dir: str = "/nas/feature_v2_10s",
                 rolling_window_years: int = 3, min_train_years: int = 2,
                 sequence_length: int = 20, prediction_horizon: int = 1,
                 batch_size: int = 64, num_workers: int = 8,
                 cache_days: int = 5, streaming_batch_size: int = 50000,
                 enable_rolling_cache: bool = True):
        """
        Initialize streaming rolling data loader.
        
        Args:
            data_dir: Directory containing daily parquet files
            rolling_window_years: Number of years to train before predicting next year
            min_train_years: Minimum years required for training
            sequence_length: Length of input sequence
            prediction_horizon: Prediction horizon
            batch_size: Training batch size
            num_workers: Number of worker processes
            cache_days: Number of days to keep in LRU cache
            streaming_batch_size: PyArrow batch size for streaming
            enable_rolling_cache: Enable rolling window cache optimization
        """
        self.data_dir = data_dir
        self.rolling_window_years = rolling_window_years
        self.min_train_years = min_train_years
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_days = cache_days
        self.streaming_batch_size = streaming_batch_size
        self.enable_rolling_cache = enable_rolling_cache
        
        # Initialize daily data loader
        from .data_pipeline import DailyDataLoader
        self.daily_loader = DailyDataLoader(data_dir)
        
        # Get available dates and split by years
        self.available_dates = self.daily_loader.available_dates
        self.yearly_dates = self._split_dates_by_year()
        
        logger.info(f"Initialized streaming rolling data loader")
        logger.info(f"Available years: {list(self.yearly_dates.keys())}")
        logger.info(f"Rolling window: {rolling_window_years} years")
    
    def _split_dates_by_year(self) -> Dict[int, List[str]]:
        """Split available dates by year."""
        yearly_dates = {}
        
        for date in self.available_dates:
            try:
                year = int(date[:4])
                if year not in yearly_dates:
                    yearly_dates[year] = []
                yearly_dates[year].append(date)
            except ValueError:
                continue
        
        # Sort dates within each year
        for year in yearly_dates:
            yearly_dates[year].sort()
        
        return yearly_dates
    
    def get_training_windows(self, prediction_years: List[int]) -> List[Tuple[List[int], int]]:
        """Get training windows for rolling window training."""
        training_windows = []
        
        for pred_year in prediction_years:
            if pred_year in self.yearly_dates:
                # Use rolling window of previous years
                start_year = max(min(self.yearly_dates.keys()), pred_year - self.rolling_window_years)
                train_years = list(range(start_year, pred_year))
                
                if len(train_years) >= self.min_train_years:
                    training_windows.append((train_years, pred_year))
        
        return training_windows
    
    def create_dataset_for_window(self, train_years: List[int], test_year: int) -> StreamingRollingDataset:
        """Create dataset for a specific training window."""
        # Collect training dates
        train_dates = []
        for year in train_years:
            if year in self.yearly_dates:
                train_dates.extend(self.yearly_dates[year])
        
        # Collect test dates
        test_dates = []
        if test_year in self.yearly_dates:
            test_dates.extend(self.yearly_dates[test_year])
        
        if not train_dates:
            raise ValueError(f"No training data available for years {train_years}")
        
        if not test_dates:
            raise ValueError(f"No test data available for year {test_year}")
        
        # Create streaming rolling dataset
        dataset = StreamingRollingDataset(
            self.daily_loader, train_dates, test_dates,
            self.sequence_length, self.prediction_horizon,
            cache_days=self.cache_days,
            batch_size=self.streaming_batch_size,
            enable_rolling_cache=self.enable_rolling_cache,
            rolling_window_size=self.rolling_window_years
        )
        
        return dataset
    
    def get_data_loaders_for_window(self, train_years: List[int], test_year: int) -> Tuple[DataLoader, DataLoader]:
        """Get data loaders for a specific training window."""
        # Create dataset
        dataset = self.create_dataset_for_window(train_years, test_year)
        
        # Split dataset into train and test
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )
        
        # Create data loaders with optimized settings
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            prefetch_factor=2,
            pin_memory=True,
            drop_last=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=max(2, self.num_workers // 2),
            prefetch_factor=2,
            pin_memory=True,
            drop_last=False
        )
        
        return train_loader, test_loader
    
    def optimize_cache_for_next_window(self, current_train_years: List[int], next_test_year: int):
        """Optimize cache for the next training window."""
        # This method can be called between training windows to preload data
        if hasattr(self, 'current_dataset'):
            self.current_dataset.optimize_cache_for_rolling(current_train_years, next_test_year)
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics for monitoring."""
        if hasattr(self, 'current_dataset'):
            return self.current_dataset.get_cache_stats()
        return {'data_cache_size': 0, 'rolling_cache_size': 0} 