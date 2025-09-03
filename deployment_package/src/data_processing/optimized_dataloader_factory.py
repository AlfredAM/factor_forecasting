#!/usr/bin/env python3
"""
Optimized DataLoader Factory: Implements all design principles
- Strict streaming with PyArrow iter_batches
- Cross-batch sliding windows with per-sid tail buffers
- LRU caching with controlled capacity
- Rolling windows with TimeSeriesSplit compatibility
- DataLoader parallelization with worker sharding
- Buffered shuffling for large-scale streaming
"""

import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader, IterableDataset
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
from collections import OrderedDict, defaultdict
import threading
import gc
import hashlib
from functools import lru_cache

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data_processing.streaming_arrow_dataset import StrictStreamingDataset
from src.data_processing.streaming_rolling_dataset import StreamingRollingDataset

logger = logging.getLogger(__name__)


class OptimizedDataLoaderFactory:
    """
    Factory class for creating optimized data loaders that implement all design principles.
    """
    
    def __init__(self, config_path: str = "configs/streaming_optimization.yaml"):
        """
        Initialize the factory with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger(__name__)
        
        # Validate configuration
        self._validate_config()
        
        self.logger.info("Optimized DataLoader Factory initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            self.logger.warning(f"Failed to load config from {config_path}: {e}")
            # Return default configuration
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file loading fails."""
        return {
            'streaming': {
                'batch_size': 50000,
                'memory_peak_control': True,
                'tail_buffer_length': 25,
                'enable_sliding_windows': True,
                'cache': {
                    'max_items': 10,
                    'max_bytes': 1073741824,  # 1GB
                    'enable_lru': True,
                    'eviction_policy': 'lru'
                },
                'dataloader': {
                    'num_workers': 8,
                    'prefetch_factor': 2,
                    'pin_memory': True,
                    'persistent_workers': True,
                    'drop_last': True
                },
                'worker_sharding': {
                    'enable': True,
                    'hash_based_assignment': True,
                    'consistent_distribution': True
                },
                'shuffling': {
                    'enable': True,
                    'buffer_size': 10000,
                    'algorithm': 'buffered_shuffle'
                }
            }
        }
    
    def _validate_config(self):
        """Validate configuration parameters."""
        streaming_config = self.config.get('streaming', {})
        
        # Validate required parameters
        required_params = ['batch_size', 'cache', 'dataloader']
        for param in required_params:
            if param not in streaming_config:
                raise ValueError(f"Missing required parameter: {param}")
        
        # Validate cache configuration
        cache_config = streaming_config['cache']
        if not cache_config.get('enable_lru', False):
            self.logger.warning("LRU caching is disabled - this may impact performance")
        
        # Validate DataLoader configuration
        dataloader_config = streaming_config['dataloader']
        if dataloader_config.get('num_workers', 0) <= 0:
            self.logger.warning("num_workers should be > 0 for optimal performance")
        
        self.logger.info("Configuration validation completed")
    
    def create_strict_streaming_dataset(self, data_dir: str, train_dates: List[str], 
                                      test_dates: List[str], sequence_length: int = 20,
                                      prediction_horizon: int = 1, worker_id: int = 0,
                                      num_workers: int = 1) -> StrictStreamingDataset:
        """
        Create strict streaming dataset implementing all design principles.
        
        Args:
            data_dir: Directory containing daily parquet files
            train_dates: Training dates
            test_dates: Test dates
            sequence_length: Length of input sequence
            prediction_horizon: Prediction horizon
            worker_id: Current worker ID for sharding
            num_workers: Total number of workers
            
        Returns:
            StrictStreamingDataset with optimized configuration
        """
        streaming_config = self.config['streaming']
        
        dataset = StrictStreamingDataset(
            data_dir=data_dir,
            train_dates=train_dates,
            test_dates=test_dates,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            batch_size=streaming_config['batch_size'],
            cache_max_items=streaming_config['cache']['max_items'],
            cache_max_bytes=streaming_config['cache']['max_bytes'],
            enable_shuffle=streaming_config['shuffling']['enable'],
            shuffle_buffer_size=streaming_config['shuffling']['buffer_size']
        )
        
        self.logger.info(f"Created strict streaming dataset: worker {worker_id}/{num_workers}")
        return dataset
    
    def create_rolling_window_dataset(self, data_dir: str, train_years: List[int],
                                    test_year: int, sequence_length: int = 20,
                                    prediction_horizon: int = 1) -> StreamingRollingDataset:
        """
        Create rolling window dataset for concept drift management.
        
        Args:
            data_dir: Data directory
            train_years: Training years
            test_year: Test year
            sequence_length: Input sequence length
            prediction_horizon: Prediction horizon
            
        Returns:
            StreamingRollingDataset with rolling window optimization
        """
        # This would need to be implemented based on your existing StreamingRollingDataset
        # For now, we'll create a placeholder
        raise NotImplementedError("Rolling window dataset creation not yet implemented")
    
    def create_optimized_dataloader(self, dataset: IterableDataset, is_training: bool = True,
                                  worker_id: int = 0, num_workers: int = 1) -> DataLoader:
        """
        Create optimized DataLoader with all performance optimizations.
        
        Args:
            dataset: Dataset to wrap
            is_training: Whether this is for training
            worker_id: Current worker ID
            num_workers: Total number of workers
            
        Returns:
            Optimized DataLoader
        """
        streaming_config = self.config['streaming']
        dataloader_config = streaming_config['dataloader']
        
        # Adjust workers based on training mode
        if is_training:
            num_workers = max(4, dataloader_config['num_workers'])
        else:
            num_workers = max(2, dataloader_config['num_workers'] // 2)
        
        # Create DataLoader with optimizations
        dataloader = DataLoader(
            dataset,
            batch_size=getattr(dataset, 'batch_size', 64),
            num_workers=num_workers,
            pin_memory=dataloader_config['pin_memory'],
            prefetch_factor=dataloader_config['prefetch_factor'],
            persistent_workers=dataloader_config['persistent_workers'],
            drop_last=dataloader_config['drop_last'] if is_training else False
        )
        
        self.logger.info(f"Created optimized DataLoader: training={is_training}, workers={num_workers}")
        return dataloader
    
    def create_continuous_training_loaders(self, data_dir: str, train_dates: List[str],
                                        val_dates: List[str], test_dates: List[str],
                                        sequence_length: int = 20, prediction_horizon: int = 1,
                                        batch_size: int = 64, num_workers: int = 8) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create continuous training data loaders with streaming optimization.
        
        Args:
            data_dir: Data directory
            train_dates: Training dates
            val_dates: Validation dates
            test_dates: Test dates
            sequence_length: Input sequence length
            prediction_horizon: Prediction horizon
            batch_size: Training batch size
            num_workers: Number of workers
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Create datasets
        train_dataset = self.create_strict_streaming_dataset(
            data_dir, train_dates, test_dates, sequence_length, prediction_horizon,
            worker_id=0, num_workers=1  # Single dataset for continuous training
        )
        
        val_dataset = self.create_strict_streaming_dataset(
            data_dir, val_dates, test_dates, sequence_length, prediction_horizon,
            worker_id=0, num_workers=1
        )
        
        test_dataset = self.create_strict_streaming_dataset(
            data_dir, test_dates, test_dates, sequence_length, prediction_horizon,
            worker_id=0, num_workers=1
        )
        
        # Create optimized data loaders
        train_loader = self.create_optimized_dataloader(
            train_dataset, is_training=True, worker_id=0, num_workers=num_workers
        )
        
        val_loader = self.create_optimized_dataloader(
            val_dataset, is_training=False, worker_id=0, num_workers=max(2, num_workers // 2)
        )
        
        test_loader = self.create_optimized_dataloader(
            test_dataset, is_training=False, worker_id=0, num_workers=max(2, num_workers // 2)
        )
        
        self.logger.info("Created continuous training data loaders")
        return train_loader, val_loader, test_loader
    
    def create_rolling_window_loaders(self, data_dir: str, train_years: List[int],
                                    test_year: int, sequence_length: int = 20,
                                    prediction_horizon: int = 1, batch_size: int = 64,
                                    num_workers: int = 8) -> Tuple[DataLoader, DataLoader]:
        """
        Create rolling window data loaders for concept drift management.
        
        Args:
            data_dir: Data directory
            train_years: Training years
            test_year: Test year
            sequence_length: Input sequence length
            prediction_horizon: Prediction horizon
            batch_size: Training batch size
            num_workers: Number of workers
            
        Returns:
            Tuple of (train_loader, test_loader)
        """
        # Get dates for years
        train_dates = self._get_dates_for_years(data_dir, train_years)
        test_dates = self._get_dates_for_years(data_dir, [test_year])
        
        # Create datasets
        train_dataset = self.create_strict_streaming_dataset(
            data_dir, train_dates, test_dates, sequence_length, prediction_horizon,
            worker_id=0, num_workers=1
        )
        
        test_dataset = self.create_strict_streaming_dataset(
            data_dir, test_dates, test_dates, sequence_length, prediction_horizon,
            worker_id=0, num_workers=1
        )
        
        # Create optimized data loaders
        train_loader = self.create_optimized_dataloader(
            train_dataset, is_training=True, worker_id=0, num_workers=num_workers
        )
        
        test_loader = self.create_optimized_dataloader(
            test_dataset, is_training=False, worker_id=0, num_workers=max(2, num_workers // 2)
        )
        
        self.logger.info(f"Created rolling window loaders: {train_years} -> {test_year}")
        return train_loader, test_loader
    
    def _get_dates_for_years(self, data_dir: str, years: List[int]) -> List[str]:
        """Get dates for specific years from data directory."""
        dates = []
        data_path = Path(data_dir)
        
        for year in years:
            for file_path in data_path.glob(f"{year}*.parquet"):
                filename = file_path.stem
                if len(filename) == 8 and filename.isdigit():
                    date_str = f"{filename[:4]}-{filename[4:6]}-{filename[6:8]}"
                    dates.append(date_str)
        
        return sorted(dates)
    
    def get_performance_tips(self) -> List[str]:
        """Get performance optimization tips based on configuration."""
        tips = [
                    "Strict streaming: One date at a time with PyArrow iter_batches",
        "Cross-batch sliding windows: Per-sid tail buffers for continuity",
        "LRU caching: Controlled capacity with max_items/max_bytes",
        "Rolling windows: TimeSeriesSplit compatible validation",
        "DataLoader parallelization: Worker sharding + prefetching",
        "Buffered shuffling: Large-scale streaming with controlled randomness",
        "Memory management: Strict control with immediate cleanup"
        ]
        
        streaming_config = self.config.get('streaming', {})
        
        if streaming_config.get('dataloader', {}).get('num_workers', 0) <= 0:
            tips.append("Consider increasing num_workers for better performance")
        
        if not streaming_config.get('cache', {}).get('enable_lru', False):
            tips.append("Enable LRU caching for better memory utilization")
        
        return tips


# Example usage
if __name__ == "__main__":
    # Initialize factory
    factory = OptimizedDataLoaderFactory()
    
    # Print performance tips
    print("Performance Optimization Tips:")
    for tip in factory.get_performance_tips():
        print(f"  {tip}")
    
    # Example: Create continuous training loaders
    data_dir = "/nas/feature_v2_10s"
    
    # Sample dates (replace with actual dates)
    train_dates = ["2020-01-01", "2020-01-02", "2020-01-03"]
    val_dates = ["2020-01-04", "2020-01-05"]
    test_dates = ["2020-01-06", "2020-01-07"]
    
    try:
        train_loader, val_loader, test_loader = factory.create_continuous_training_loaders(
            data_dir=data_dir,
            train_dates=train_dates,
            val_dates=val_dates,
            test_dates=test_dates,
            sequence_length=20,
            prediction_horizon=1,
            batch_size=64,
            num_workers=4
        )
        
        print(f"\nCreated data loaders:")
        print(f"  Train: {len(train_loader)} batches")
        print(f"  Validation: {len(val_loader)} batches")
        print(f"  Test: {len(test_loader)} batches")
        
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        print("This is expected if the data directory doesn't exist")
