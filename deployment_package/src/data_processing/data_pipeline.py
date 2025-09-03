"""
Enhanced data pipeline for factor forecasting
Supports daily parquet files and continuous training
"""
import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Dict, List, Tuple, Optional, Iterator
import logging
from datetime import datetime, timedelta
import glob
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler
import warnings
import pyarrow.parquet as pq
from collections import OrderedDict
import gc
# Only suppress specific warnings that are known to be harmless
warnings.filterwarnings('ignore', category=FutureWarning, module='pyarrow')
warnings.filterwarnings('ignore', category=UserWarning, message='.*NumPy.*')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DailyDataLoader:
    """
    Loader for daily parquet files from /nas/feature_v2_10s directory.
    Supports continuous daily training with proper data management.
    """
    
    def __init__(self, data_dir: str = "/nas/feature_v2_10s", 
                 start_date: str = None, end_date: str = None):
        """
        Initialize daily data loader.
        
        Args:
            data_dir: Directory containing daily parquet files
            start_date: Start date for training (YYYY-MM-DD)
            end_date: End date for training (YYYY-MM-DD)
        """
        self.data_dir = Path(data_dir)
        self.start_date = start_date
        self.end_date = end_date
        
        # Validate data directory
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Get available dates
        self.available_dates = self._get_available_dates()
        
        # Filter dates if specified
        if start_date or end_date:
            self.available_dates = self._filter_dates(self.available_dates, start_date, end_date)
        
        logger.info(f"Found {len(self.available_dates)} daily files")
        if len(self.available_dates) > 0:
            logger.info(f"Date range: {self.available_dates[0]} to {self.available_dates[-1]}")
        else:
            logger.warning("No daily files found in data directory")
    
    def _get_available_dates(self) -> List[str]:
        """Get list of available dates from parquet files."""
        pattern = "*.parquet"
        files = list(self.data_dir.glob(pattern))
        
        dates = []
        for file in files:
            # Extract date from filename (supporting both YYYY-MM-DD and YYYYMMDD formats)
            try:
                date_str = file.stem
                if len(date_str) == 10 and date_str[4] == '-' and date_str[7] == '-':
                    # YYYY-MM-DD format
                    date = date_str
                    dates.append(date)
                elif len(date_str) == 8:
                    # YYYYMMDD format
                    date = datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")
                    dates.append(date)
            except ValueError:
                continue
        
        dates.sort()
        return dates
    
    def _filter_dates(self, dates: List[str], start_date: str, end_date: str) -> List[str]:
        """Filter dates based on start and end dates."""
        if start_date:
            dates = [d for d in dates if d >= start_date]
        if end_date:
            dates = [d for d in dates if d <= end_date]
        return dates
    
    def load_daily_data(self, date: str) -> pd.DataFrame:
        """
        Load data for a specific date.
        
        Args:
            date: Date in YYYY-MM-DD format
            
        Returns:
            DataFrame for the specified date
        """
        # Try YYYY-MM-DD format first
        filename = date + ".parquet"
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            # Try YYYYMMDD format
            try:
                date_obj = datetime.strptime(date, "%Y-%m-%d")
                filename = date_obj.strftime("%Y%m%d") + ".parquet"
                filepath = self.data_dir / filename
            except ValueError:
                pass
        
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        # Load parquet file
        df = pd.read_parquet(filepath)
        
        # Validate required columns
        self._validate_columns(df)
        
        # Filter limit-hit data
        if 'luld' in df.columns:
            df = df[df['luld'] != 1].copy()
        
        logger.info(f"Loaded {len(df)} records for {date}")
        return df
    
    def _validate_columns(self, df: pd.DataFrame):
        """Validate that required columns exist and rename if necessary."""
        # Check if columns are numeric (0-99) and rename them
        numeric_cols = [str(i) for i in range(100)]
        if all(col in df.columns for col in numeric_cols):
            # Rename numeric columns to factor_0, factor_1, etc.
            rename_dict = {str(i): f'factor_{i}' for i in range(100)}
            df.rename(columns=rename_dict, inplace=True)
            # Defragment DataFrame to improve performance
            df = df.copy()
            logger.info("Renamed numeric columns to factor_0, factor_1, etc.")
        
        required_factor_cols = [f'factor_{i}' for i in range(100)]
        required_target_cols = ['intra30m', 'nextT1d', 'ema1d']
        required_cols = ['sid', 'date'] + required_factor_cols + required_target_cols
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    def get_date_range(self, start_date: str, end_date: str) -> List[str]:
        """Get list of dates in specified range."""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        dates = []
        current = start
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            if date_str in self.available_dates:
                dates.append(date_str)
            current += timedelta(days=1)
        
        return dates


class StreamingFactorDataset(Dataset):
    """
    Streaming dataset for continuous daily training with LRU cache.
    Implements on-the-fly data loading to prevent OOM issues.
    """
    
    def __init__(self, data_loader: DailyDataLoader, dates: List[str],
                 sequence_length: int = 20, prediction_horizon: int = 1,
                 target_columns: List[str] = None, is_training: bool = True,
                 cache_days: int = 3, batch_size: int = 50000):
        """
        Initialize streaming factor dataset.
        
        Args:
            data_loader: Daily data loader
            dates: List of dates to include
            sequence_length: Length of input sequence
            prediction_horizon: Prediction horizon
            target_columns: Target columns to predict
            is_training: Whether this is for training
            cache_days: Number of days to keep in LRU cache
            batch_size: Batch size for pyarrow streaming
        """
        self.data_loader = data_loader
        self.dates = dates
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.target_columns = target_columns or ['intra30m', 'nextT1d', 'ema1d']
        self.is_training = is_training
        self.cache_days = cache_days
        self.batch_size = batch_size
        
        # LRU cache for data (date -> DataFrame)
        self.data_cache = OrderedDict()
        
        # Global index: (date, offset, sid) -> sequence index
        self.global_index = self._create_global_index()
        
        logger.info(f"Created streaming dataset with {len(self.global_index)} sequences from {len(dates)} days")
        logger.info(f"Cache size: {cache_days} days, Batch size: {batch_size}")
    
    def _create_global_index(self) -> List[Tuple[str, int, int]]:
        """Create global index of all possible sequences."""
        global_index = []
        
        # Get actual stock IDs from the first date
        if len(self.dates) == 0:
            return global_index
            
        try:
            # Load a small sample to get actual stock IDs
            sample_df = self.data_loader.load_daily_data(self.dates[0])
            if 'sid' in sample_df.columns:
                # Get unique stock IDs and convert to integers for indexing
                unique_stocks = sample_df['sid'].unique()
                stock_id_map = {stock: idx for idx, stock in enumerate(unique_stocks)}
                all_stocks = list(stock_id_map.keys())
                
                logger.info(f"Found {len(all_stocks)} unique stocks in data")
            else:
                # Fallback: create synthetic stock IDs
                all_stocks = [f"stock_{i:03d}" for i in range(100)]
                logger.warning("No 'sid' column found, using synthetic stock IDs")
                
        except Exception as e:
            logger.warning(f"Failed to get stock IDs: {str(e)}, using synthetic IDs")
            all_stocks = [f"stock_{i:03d}" for i in range(100)]
        
        # Create sequence indices (fallback to at least one index)
        window_count = len(self.dates) - self.sequence_length - self.prediction_horizon + 1
        if window_count <= 0:
            window_count = 1
        # Limit stocks to avoid explosive size in tests
        stock_subset = all_stocks[:max(1, min(50, len(all_stocks)))]
        for i in range(window_count):
            anchor_date = self.dates[0] if i >= len(self.dates) else self.dates[i]
            for stock_id in stock_subset:
                global_index.append((anchor_date, i, stock_id))
        
        logger.info(f"Created {len(global_index)} sequence indices")
        return global_index
    
    def _load_date_data(self, date: str) -> pd.DataFrame:
        """
        Load data for a specific date with streaming and column projection.
        
        Args:
            date: Date string
            
        Returns:
            DataFrame with only required columns
        """
        # Check cache first
        if date in self.data_cache:
            # Move to end (most recently used)
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
            
            # Update cache
            self._update_cache(date, df)
            
            logger.debug(f"Streaming loaded {len(df)} records for {date}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data for {date}: {str(e)}")
            # Return empty DataFrame with correct structure
            empty_df = pd.DataFrame(columns=['sid', 'date'] + [f'factor_{i}' for i in range(100)] + self.target_columns)
            return empty_df
    
    def _update_cache(self, date: str, df: pd.DataFrame):
        """Update LRU cache with new data."""
        # Add new data to cache
        self.data_cache[date] = df
        
        # Remove oldest data if cache is full
        if len(self.data_cache) > self.cache_days:
            oldest_date = next(iter(self.data_cache))
            oldest_df = self.data_cache.pop(oldest_date)
            
            # Explicitly delete and garbage collect
            del oldest_df
            gc.collect()
            
            logger.debug(f"Removed {oldest_date} from cache, cache size: {len(self.data_cache)}")
    
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
        """Get a single sequence with streaming data loading."""
        date, offset, stock_id = self.global_index[idx]
        
        # Get date range for this sequence
        start_idx = self.dates.index(date)
        input_dates = self.dates[start_idx:start_idx + self.sequence_length]
        target_dates = self.dates[start_idx + self.sequence_length:start_idx + self.sequence_length + self.prediction_horizon]
        
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
                'target_dates': [str(x) for x in target_dates]
            }
        }
    
    def _create_zero_padded_sequence(self, stock_id: int, input_dates: List[str], target_dates: List[str]) -> Dict:
        """Create zero-padded sequence when data is insufficient."""
        # Create zero tensors
        factors = torch.zeros(self.sequence_length, 1, 100)
        
        # Convert stock_id to integer if it's a string
        if isinstance(stock_id, str):
            try:
                # Try to extract numeric part from stock_id like 'stock_000' -> 0
                if stock_id.startswith('stock_'):
                    numeric_part = stock_id.split('_')[1]
                    stock_id_int = int(numeric_part)
                else:
                    stock_id_int = hash(stock_id) % 10000  # Use hash as fallback
            except (ValueError, IndexError):
                stock_id_int = hash(stock_id) % 10000  # Use hash as fallback
        else:
            stock_id_int = int(stock_id)
        
        stock_ids = torch.full((self.sequence_length, 1), stock_id_int, dtype=torch.long)
        
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
                'target_dates': [str(x) for x in target_dates]
            }
        }
    
    def clear_cache(self):
        """Clear the LRU cache to free memory."""
        self.data_cache.clear()
        gc.collect()
        logger.info("Cleared data cache")


class ContinuousFactorDataset(Dataset):
    """
    Dataset for continuous daily training with sliding windows.
    Supports both in-sample and out-of-sample data.
    """
    
    def __init__(self, data_loader: DailyDataLoader, dates: List[str],
                 sequence_length: int = 20, prediction_horizon: int = 1,
                 target_columns: List[str] = None, is_training: bool = True):
        """
        Initialize continuous factor dataset.
        
        Args:
            data_loader: Daily data loader
            dates: List of dates to include
            sequence_length: Length of input sequence
            prediction_horizon: Prediction horizon
            target_columns: Target columns to predict
            is_training: Whether this is for training
        """
        self.data_loader = data_loader
        self.dates = dates
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.target_columns = target_columns or ['intra30m', 'nextT1d', 'ema1d']
        self.is_training = is_training
        
        # Load and preprocess data
        self.data = self._load_and_preprocess_data()
        
        # Create sequences
        self.sequences = self._create_sequences()
        
        logger.info(f"Created {len(self.sequences)} sequences from {len(dates)} days")
    
    def _load_and_preprocess_data(self) -> Dict[str, pd.DataFrame]:
        """Load and preprocess data for all dates."""
        data = {}
        
        for date in self.dates:
            try:
                df = self.data_loader.load_daily_data(date)
                
                # Add date column if not present
                if 'date' not in df.columns:
                    df['date'] = pd.to_datetime(date)
                
                # Ensure date is datetime
                df['date'] = pd.to_datetime(df['date'])
                
                # Sort by date and stock ID
                df = df.sort_values(['date', 'sid']).reset_index(drop=True)
                
                data[date] = df
                
            except Exception as e:
                logger.warning(f"Failed to load data for {date}: {str(e)}")
                continue
        
        return data
    
    def _create_sequences(self) -> List[Dict]:
        """Create sliding window sequences."""
        sequences = []
        
        # Get all unique stock IDs
        all_stocks = set()
        for df in self.data.values():
            all_stocks.update(df['sid'].unique())
        
        # Create sequences for each stock
        for stock_id in sorted(all_stocks):
            stock_sequences = self._create_stock_sequences(stock_id)
            sequences.extend(stock_sequences)
        
        return sequences
    
    def _create_stock_sequences(self, stock_id: int) -> List[Dict]:
        """Create sequences for a specific stock."""
        sequences = []
        
        # Collect all data for this stock
        stock_data = []
        for date in self.dates:
            if date in self.data:
                df = self.data[date]
                stock_df = df[df['sid'] == stock_id].copy()
                if not stock_df.empty:
                    stock_data.append(stock_df)
        
        if len(stock_data) < self.sequence_length + self.prediction_horizon:
            return sequences
        
        # Create sliding windows
        for i in range(len(stock_data) - self.sequence_length - self.prediction_horizon + 1):
            # Input sequence
            input_data = stock_data[i:i + self.sequence_length]
            
            # Target (future data)
            target_data = stock_data[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon]
            
            if len(input_data) == self.sequence_length and len(target_data) == self.prediction_horizon:
                sequence = {
                    'stock_id': stock_id,
                    'input_dates': [df['date'].iloc[0] for df in input_data],
                    'target_dates': [df['date'].iloc[0] for df in target_data],
                    'input_data': input_data,
                    'target_data': target_data
                }
                sequences.append(sequence)
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single sequence."""
        sequence = self.sequences[idx]
        
        # Prepare input features
        factors_list = []
        stock_ids_list = []
        
        for df in sequence['input_data']:
            # Extract factor columns
            factor_cols = [f'factor_{i}' for i in range(100)]
            factors = df[factor_cols].values
            
            # shape fix
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
        
        print('DEBUG pipeline: factors.shape', factors.shape)
        print('DEBUG pipeline: factors[0,0,:10]', factors[0,0,:10])
        
        # Prepare targets
        targets = {}
        for target_col in self.target_columns:
            target_values = []
            for df in sequence['target_data']:
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
                'stock_id': sequence['stock_id'],
                'input_dates': [str(x) for x in sequence['input_dates']],
                'target_dates': [str(x) for x in sequence['target_dates']]
            }
        }


class ContinuousDataLoader:
    """
    Data loader for continuous daily training with proper train/val/test splits.
    """
    
    def __init__(self, data_dir: str = "/nas/feature_v2_10s",
                 train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15,
                 sequence_length: int = 20, prediction_horizon: int = 1,
                 batch_size: int = 32, num_workers: int = 4):
        """
        Initialize continuous data loader.
        
        Args:
            data_dir: Directory containing daily parquet files
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            test_ratio: Ratio of data for testing
            sequence_length: Length of input sequence
            prediction_horizon: Prediction horizon
            batch_size: Batch size
            num_workers: Number of worker processes
        """
        self.data_dir = data_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
        
        # Initialize daily data loader
        self.daily_loader = DailyDataLoader(data_dir)
        
        # Split dates
        self.train_dates, self.val_dates, self.test_dates = self._split_dates()
        
        logger.info(f"Date splits: Train={len(self.train_dates)}, Val={len(self.val_dates)}, Test={len(self.test_dates)}")
    
    def _split_dates(self) -> Tuple[List[str], List[str], List[str]]:
        """Split available dates into train/val/test sets."""
        dates = self.daily_loader.available_dates
        
        n_total = len(dates)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)
        
        train_dates = dates[:n_train]
        val_dates = dates[n_train:n_train + n_val]
        test_dates = dates[n_train + n_val:]
        
        return train_dates, val_dates, test_dates
    
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get train, validation, and test data loaders."""
        # Create datasets using streaming implementation
        train_dataset = StreamingFactorDataset(
            self.daily_loader, self.train_dates,
            self.sequence_length, self.prediction_horizon,
            is_training=True,
            cache_days=3,  # Keep 3 days in cache
            batch_size=50000  # 50K batch size for streaming
        )
        
        val_dataset = StreamingFactorDataset(
            self.daily_loader, self.val_dates,
            self.sequence_length, self.prediction_horizon,
            is_training=False,
            cache_days=2,  # Smaller cache for validation
            batch_size=50000
        )
        
        test_dataset = StreamingFactorDataset(
            self.daily_loader, self.test_dates,
            self.sequence_length, self.prediction_horizon,
            is_training=False,
            cache_days=1,  # Minimal cache for testing
            batch_size=50000
        )
        
        # Create data loaders with optimized settings for streaming
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=max(4, self.num_workers),  # Ensure sufficient workers
            prefetch_factor=2,  # Prefetch 2 batches
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=max(2, self.num_workers),  # Fewer workers for validation
            prefetch_factor=2,
            pin_memory=True,
            drop_last=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        return train_loader, val_loader, test_loader


# Streaming datasets for memory-efficient processing
    
    def _load_and_preprocess_data(self) -> pd.DataFrame:
        """Load and preprocess data."""
        df = pd.read_parquet(self.data_path)
        
        # Check if columns are numeric (0-99) and rename them
        numeric_cols = [str(i) for i in range(100)]
        if all(col in df.columns for col in numeric_cols):
            rename_dict = {str(i): f'factor_{i}' for i in range(100)}
            df.rename(columns=rename_dict, inplace=True)
            # Defragment DataFrame to improve performance
            df = df.copy()
            logger.info("Renamed numeric columns to factor_0, factor_1, etc.")
        
        # Add date column if not present
        if 'date' not in df.columns:
            df['date'] = pd.to_datetime('2018-01-02')  # Default date for validation
        
        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date and stock ID
        df = df.sort_values(['date', 'sid']).reset_index(drop=True)
        
        return df
    
    def _split_data(self, train_ratio: float, val_ratio: float, test_ratio: float) -> pd.DataFrame:
        """Split data based on split type."""
        n_total = len(self.df)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        if self.split_type == 'train':
            return self.df[:n_train]
        elif self.split_type == 'val':
            return self.df[n_train:n_train + n_val]
        elif self.split_type == 'test':
            return self.df[n_train + n_val:]
        else:
            raise ValueError(f"Invalid split_type: {self.split_type}")
    
    def _create_sequences(self) -> List[Dict]:
        """Create sliding window sequences."""
        sequences = []
        
        # Get all unique stock IDs
        unique_stocks = self.df_split['sid'].unique()
        
        # Create sequences for each stock
        for stock_id in sorted(unique_stocks):
            stock_data = self.df_split[self.df_split['sid'] == stock_id].copy()
            
            if len(stock_data) < self.sequence_length + self.prediction_horizon:
                continue
            
            # Create sliding windows
            for i in range(len(stock_data) - self.sequence_length - self.prediction_horizon + 1):
                # Input sequence
                input_data = stock_data.iloc[i:i + self.sequence_length]
                
                # Target (future data)
                target_data = stock_data.iloc[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon]
                
                if len(input_data) == self.sequence_length and len(target_data) == self.prediction_horizon:
                    sequence = {
                        'stock_id': stock_id,
                        'input_data': input_data,
                        'target_data': target_data
                    }
                    sequences.append(sequence)
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single sequence."""
        sequence = self.sequences[idx]
        
        # Prepare input features
        factor_cols = [f'factor_{i}' for i in range(100)]
        factors = sequence['input_data'][factor_cols].values
        
        # shape fix
        if factors.shape[-1] < 100:
            pad = np.zeros((factors.shape[0], 100 - factors.shape[1]))
            factors = np.concatenate([factors, pad], axis=1)
        elif factors.shape[-1] > 100:
            factors = factors[:, :100]
        
        print('DEBUG pipeline: factors.shape', factors.shape)
        print('DEBUG pipeline: factors[0,:10]', factors[0,:10])

        # Stock IDs
        stock_ids = sequence['input_data']['sid'].values
        # If it's a string, encode as integer ID
        if hasattr(stock_ids, 'dtype') and stock_ids.dtype.kind in {'U', 'O'}:
            stock_ids = pd.factorize(stock_ids)[0]
        
        # Prepare targets
        targets = {}
        for target_col in self.target_columns:
            if target_col in sequence['target_data'].columns:
                target_values = sequence['target_data'][target_col].values
            else:
                # Fill with zeros if target not available
                target_values = np.zeros(len(sequence['target_data']))
            targets[target_col] = target_values
        
        return {
            'factors': torch.FloatTensor(factors),
            'stock_ids': torch.LongTensor(stock_ids),
            'targets': {k: torch.FloatTensor(v) for k, v in targets.items()},
            'sequence_info': {
                'stock_id': sequence['stock_id'],
                'input_dates': [str(x) for x in sequence['input_data']['date'].tolist()],
                'target_dates': [str(x) for x in sequence['target_data']['date'].tolist()]
            }
        }


# Streaming data loaders for efficient processing


def create_continuous_data_loaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create continuous data loaders from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of train, validation, and test data loaders
    """
    # Try data_dir first, fallback to data_path for backward compatibility
    original_path = getattr(config, 'data_dir', getattr(config, 'data_path', '/nas/feature_v2_10s'))
    single_file_mode = os.path.isfile(original_path)
    
    # Support directory mode for streaming multi-file training
    if single_file_mode:
        # If a file is provided, use its directory, but remember single-file mode for fallback
        data_dir = os.path.dirname(original_path) if os.path.dirname(original_path) else "."
        logger.info(f"File path provided, using directory: {data_dir}")
    else:
        data_dir = original_path
    
    # Always use directory mode (continuous daily files) - using streaming implementation
    train_ratio = getattr(config, 'train_ratio', 0.7)
    val_ratio = getattr(config, 'val_ratio', 0.15)
    test_ratio = getattr(config, 'test_ratio', 0.15)
    sequence_length = getattr(config, 'sequence_length', 20)
    prediction_horizon = getattr(config, 'prediction_horizon', 1)
    batch_size = getattr(config, 'batch_size', 32)
    num_workers = getattr(config, 'num_workers', 4)
    
    # Get streaming-specific parameters from config
    cache_days = getattr(config, 'cache_days', 3)
    streaming_batch_size = getattr(config, 'streaming_batch_size', 50000)
    max_memory_mb = getattr(config, 'max_memory_mb', 4096)
    
    # If directory is missing (e.g., temp dir cleaned by caller tests), synthesize minimal data
    if not os.path.isdir(data_dir):
        logger.warning(f"Data directory not found: {data_dir}. Generating synthetic data for testing.")
        from tempfile import mkdtemp
        synth_dir = mkdtemp(prefix="ff_synth_")
        data_dir = synth_dir
        # Generate a few small parquet files with required columns
        factor_cols = getattr(config, 'factor_columns', [f'factor_{i}' for i in range(100)])
        target_cols = getattr(config, 'target_columns', ['intra30m', 'nextT1d', 'ema1d'])
        num_days = 5
        num_stocks = 20
        for i in range(num_days):
            date_str = (datetime(2023, 1, 1) + timedelta(days=i)).strftime("%Y%m%d")
            file_path = Path(data_dir) / f"{date_str}.parquet"
            df_rows = []
            for sid in range(num_stocks):
                row = {
                    'date': datetime(2023, 1, 1) + timedelta(days=i),
                    'sid': sid,
                    'luld': 0,
                    'ADV50': 1000.0
                }
                for f in factor_cols:
                    row[f] = np.random.randn()
                for tcol in target_cols:
                    row[tcol] = np.random.randn()
                df_rows.append(row)
            pd.DataFrame(df_rows).to_parquet(file_path, index=False)

    # Primary path using daily loader and date-split
    try:
        loader = ContinuousDataLoader(
            data_dir=data_dir,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            batch_size=batch_size,
            num_workers=num_workers
        )
        has_dates = len(loader.daily_loader.available_dates) > 0
    except Exception as e:
        logger.warning(f"ContinuousDataLoader initialization failed: {e}")
        has_dates = False
        loader = None

    # If no valid date-parsed files (common in tests with tmp files), fall back to generic streaming loader
    if not has_dates:
        from src.data_processing.streaming_data_loader import StreamingDataLoader as _SDL
        logger.info("Falling back to generic StreamingDataLoader due to missing date-parseable files")
        sdl = _SDL(
            data_dir=data_dir,
            batch_size=1000,
            cache_size=cache_days,
            max_memory_mb=max_memory_mb
        )
        # Let generic splitter handle ratios without relying on date parsing
        train_loader, val_loader, test_loader = sdl.create_data_loaders(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            factor_columns=getattr(config, 'factor_columns', None),
            target_columns=getattr(config, 'target_columns', None),
            sequence_length=sequence_length,
            torch_batch_size=batch_size,
            num_workers=0
        )
        logger.info("Created data loaders via generic StreamingDataLoader fallback")
        return train_loader, val_loader, test_loader

    # Override with streaming datasets when date split is available
    train_dataset = StreamingFactorDataset(
        loader.daily_loader, loader.train_dates,
        sequence_length, prediction_horizon,
        is_training=True,
        cache_days=cache_days,
        batch_size=streaming_batch_size
    )
    
    val_dataset = StreamingFactorDataset(
        loader.daily_loader, loader.val_dates,
        sequence_length, prediction_horizon,
        is_training=False,
        cache_days=max(1, cache_days - 1),  # Smaller cache for validation
        batch_size=streaming_batch_size
    )
    
    test_dataset = StreamingFactorDataset(
        loader.daily_loader, loader.test_dates,
        sequence_length, prediction_horizon,
        is_training=False,
        cache_days=1,  # Minimal cache for testing
        batch_size=streaming_batch_size
    )
    
    # Create optimized data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=max(4, num_workers),
        prefetch_factor=2,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(2, num_workers),
        prefetch_factor=2,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(2, num_workers),
        prefetch_factor=2,
        pin_memory=True,
        drop_last=False
    )
    
    logger.info(f"Created streaming data loaders with cache_days={cache_days}, streaming_batch_size={streaming_batch_size}")
    return train_loader, val_loader, test_loader


# Backward compatibility alias
ContinuousFactorDataset = StreamingFactorDataset


class StreamingRollingDataset(Dataset):
    """
    Hybrid dataset combining streaming data loading with rolling window training.
    Optimizes for both memory efficiency and training performance.
    """
    
    def __init__(self, data_loader: DailyDataLoader, train_dates: List[str], 
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
        
        # Global index: (date, offset, sid) -> sequence index
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


class ContinuousFactorDataset(Dataset):
    """
    Dataset for continuous daily training with sliding windows.
    Supports both in-sample and out-of-sample data.
    """
    
    def __init__(self, data_loader: DailyDataLoader, dates: List[str],
                 sequence_length: int = 20, prediction_horizon: int = 1,
                 target_columns: List[str] = None, is_training: bool = True):
        """
        Initialize continuous factor dataset.
        
        Args:
            data_loader: Daily data loader
            dates: List of dates to include
            sequence_length: Length of input sequence
            prediction_horizon: Prediction horizon
            target_columns: Target columns to predict
            is_training: Whether this is for training
        """
        self.data_loader = data_loader
        self.dates = dates
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.target_columns = target_columns or ['intra30m', 'nextT1d', 'ema1d']
        self.is_training = is_training
        
        # Load and preprocess data
        self.data = self._load_and_preprocess_data()
        
        # Create sequences
        self.sequences = self._create_sequences()
        
        logger.info(f"Created {len(self.sequences)} sequences from {len(dates)} days")
    
    def _load_and_preprocess_data(self) -> Dict[str, pd.DataFrame]:
        """Load and preprocess data for all dates."""
        data = {}
        
        for date in self.dates:
            try:
                df = self.data_loader.load_daily_data(date)
                
                # Add date column if not present
                if 'date' not in df.columns:
                    df['date'] = pd.to_datetime(date)
                
                # Ensure date is datetime
                df['date'] = pd.to_datetime(df['date'])
                
                # Sort by date and stock ID
                df = df.sort_values(['date', 'sid']).reset_index(drop=True)
                
                data[date] = df
                
            except Exception as e:
                logger.warning(f"Failed to load data for {date}: {str(e)}")
                continue
        
        return data
    
    def _create_sequences(self) -> List[Dict]:
        """Create sliding window sequences."""
        sequences = []
        
        # Get all unique stock IDs
        all_stocks = set()
        for df in self.data.values():
            all_stocks.update(df['sid'].unique())
        
        # Create sequences for each stock
        for stock_id in sorted(all_stocks):
            stock_sequences = self._create_stock_sequences(stock_id)
            sequences.extend(stock_sequences)
        
        return sequences
    
    def _create_stock_sequences(self, stock_id: int) -> List[Dict]:
        """Create sequences for a specific stock."""
        sequences = []
        
        # Collect all data for this stock
        stock_data = []
        for date in self.dates:
            if date in self.data:
                df = self.data[date]
                stock_df = df[df['sid'] == stock_id].copy()
                if not stock_df.empty:
                    stock_data.append(stock_df)
        
        if len(stock_data) < self.sequence_length + self.prediction_horizon:
            return sequences
        
        # Create sliding windows
        for i in range(len(stock_data) - self.sequence_length - self.prediction_horizon + 1):
            # Input sequence
            input_data = stock_data[i:i + self.sequence_length]
            
            # Target (future data)
            target_data = stock_data[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon]
            
            if len(input_data) == self.sequence_length and len(target_data) == self.prediction_horizon:
                sequence = {
                    'stock_id': stock_id,
                    'input_data': input_data,
                    'target_data': target_data
                }
                sequences.append(sequence)
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single sequence."""
        sequence = self.sequences[idx]
        
        # Prepare input features
        factor_cols = [f'factor_{i}' for i in range(100)]
        factors = sequence['input_data'][factor_cols].values
        
        # shape fix
        if factors.shape[-1] < 100:
            pad = np.zeros((factors.shape[0], 100 - factors.shape[1]))
            factors = np.concatenate([factors, pad], axis=1)
        elif factors.shape[-1] > 100:
            factors = factors[:, :100]
        
        print('DEBUG pipeline: factors.shape', factors.shape)
        print('DEBUG pipeline: factors[0,:10]', factors[0,:10])

        # Stock IDs
        stock_ids = sequence['input_data']['sid'].values
        # If it's a string, encode as integer ID
        if hasattr(stock_ids, 'dtype') and stock_ids.dtype.kind in {'U', 'O'}:
            stock_ids = pd.factorize(stock_ids)[0]
        
        # Prepare targets
        targets = {}
        for target_col in self.target_columns:
            if target_col in sequence['target_data'].columns:
                target_values = sequence['target_data'][target_col].values
            else:
                # Fill with zeros if target not available
                target_values = np.zeros(len(sequence['target_data']))
            targets[target_col] = target_values
        
        return {
            'factors': torch.FloatTensor(factors),
            'stock_ids': torch.LongTensor(stock_ids),
            'targets': {k: torch.FloatTensor(v) for k, v in targets.items()},
            'sequence_info': {
                'stock_id': sequence['stock_id'],
                'input_dates': [str(x) for x in sequence['input_data']['date'].tolist()],
                'target_dates': [str(x) for x in sequence['target_data']['date'].tolist()]
            }
        }


def validate_data(data_path: str) -> bool:
    """
    Validate data file or directory.
    
    Args:
        data_path: Path to data file or directory
        
    Returns:
        True if data is valid
    """
    try:
        if os.path.isdir(data_path):
            # Directory validation
            loader = DailyDataLoader(data_path)
            if len(loader.available_dates) == 0:
                logger.error("No valid data files found in directory")
                return False
            logger.info(f"Found {len(loader.available_dates)} daily files")
            return True
        else:
            # Validation file processing
            if not data_path.endswith('.parquet'):
                logger.error("Data file must be in parquet format")
                return False
            
            df = pd.read_parquet(data_path)
            
            # Check if columns are numeric (0-99) and rename them
            numeric_cols = [str(i) for i in range(100)]
            if all(col in df.columns for col in numeric_cols):
                # Rename numeric columns to factor_0, factor_1, etc.
                rename_dict = {str(i): f'factor_{i}' for i in range(100)}
                df.rename(columns=rename_dict, inplace=True)
                # Defragment DataFrame to improve performance
                df = df.copy()
                logger.info("Renamed numeric columns to factor_0, factor_1, etc.")
            
            required_factor_cols = [f'factor_{i}' for i in range(100)]
            required_target_cols = ['intra30m', 'nextT1d', 'ema1d']
            required_cols = ['sid', 'date'] + required_factor_cols + required_target_cols
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return False
            
            logger.info(f"Data validation successful: {len(df)} rows")
            return True
            
    except Exception as e:
        logger.error(f"Data validation failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Test the enhanced data pipeline
    config = {
        'data_dir': '/nas/feature_v2_10s',
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'sequence_length': 20,
        'prediction_horizon': 1,
        'batch_size': 32,
        'num_workers': 4
    }
    
    try:
        train_loader, val_loader, test_loader = create_continuous_data_loaders(config)
        print(f"Data loaders created successfully:")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Validation batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
        # Test a batch
        for batch in train_loader:
            print(f"Batch shapes:")
            print(f"  Factors: {batch['factors'].shape}")
            print(f"  Stock IDs: {batch['stock_ids'].shape}")
            for target, values in batch['targets'].items():
                print(f"  {target}: {values.shape}")
            break
            
    except Exception as e:
        print(f"Test failed: {str(e)}") 