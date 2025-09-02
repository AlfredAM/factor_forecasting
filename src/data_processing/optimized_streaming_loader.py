"""
Optimized Streaming Data Loader: Quantitative finance data processing system with adaptive memory management
Solves 300MB+ large file and memory overflow issues, providing industrial-grade data loading solutions
"""
import os
import gc
import logging
import threading
from collections import OrderedDict, deque
from pathlib import Path
from typing import Iterator, List, Dict, Optional, Tuple, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from datetime import datetime, timedelta, date
import psutil
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, DataLoader

from .adaptive_memory_manager import AdaptiveMemoryManager, create_memory_manager

logger = logging.getLogger(__name__)


class OptimizedStreamingCache:
    """Optimized LRU cache with integrated adaptive memory management"""
    
    def __init__(self, memory_manager: AdaptiveMemoryManager):
        self.memory_manager = memory_manager
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.cache_lock = threading.Lock()
        
        # Dynamic cache configuration
        self.max_items = 5
        self.max_bytes = int(memory_manager.memory_budget.max_cache_gb * 1024**3)
        
    def get(self, key: str) -> Optional[pd.DataFrame]:
        """Get cached data"""
        with self.cache_lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                return value.copy()
            return None
    
    def put(self, key: str, value: pd.DataFrame):
        """Add data to cache"""
        data_size = value.memory_usage(deep=True).sum()
        
        with self.cache_lock:
            # Check if exceeds single file size limit
            if data_size > self.max_bytes * 0.5:  # Single file cannot exceed half of cache
                logger.warning(f"File {key} too large ({data_size/1024**2:.1f}MB), skipping cache")
                return
            
            # Clear space
            while (len(self.cache) >= self.max_items or 
                   self.cache_bytes + data_size > self.max_bytes):
                if not self.cache:
                    break
                
                oldest_key, oldest_data = self.cache.popitem(last=False)
                removed_size = oldest_data.memory_usage(deep=True).sum()
                self.cache_bytes -= removed_size
                logger.debug(f"Cache eviction: {oldest_key}, freed {removed_size/1024**2:.1f}MB")
                del oldest_data
            
            # Add new data
            self.cache[key] = value.copy()
            self.cache_bytes += data_size
            logger.debug(f"Cache added: {key}, size {data_size/1024**2:.1f}MB, total {self.cache_bytes/1024**2:.1f}MB")
    
    def update_limits(self, avg_file_size_mb: float):
        """Update cache limits based on file size"""
        new_max_items = self.memory_manager.calculate_optimal_cache_size(avg_file_size_mb)
        new_max_bytes = int(self.memory_manager.memory_budget.max_cache_gb * 1024**3)
        
        with self.cache_lock:
            if new_max_items != self.max_items or new_max_bytes != self.max_bytes:
                logger.info(f"Cache limits updated: items {self.max_items}->{new_max_items}, "
                           f"size {self.max_bytes/1024**2:.1f}->{new_max_bytes/1024**2:.1f}MB")
                self.max_items = new_max_items
                self.max_bytes = new_max_bytes
                
                # If current cache exceeds new limits, need to clean up
                while (len(self.cache) > self.max_items or 
                       self.cache_bytes > self.max_bytes):
                    if not self.cache:
                        break
                    oldest_key, oldest_data = self.cache.popitem(last=False)
                    removed_size = oldest_data.memory_usage(deep=True).sum()
                    self.cache_bytes -= removed_size
                    del oldest_data
    
    def clear(self):
        """Clear cache"""
        with self.cache_lock:
            self.cache.clear()
            self.cache_bytes = 0
            gc.collect()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.cache_lock:
            return {
                'items': len(self.cache),
                'max_items': self.max_items,
                'bytes': self.cache_bytes,
                'max_bytes': self.max_bytes,
                'usage_ratio': self.cache_bytes / max(self.max_bytes, 1),
                'keys': list(self.cache.keys())
            }


class ChunkedFileLoader:
    """Chunked file loader supporting memory-safe loading of large files"""
    
    def __init__(self, memory_manager: AdaptiveMemoryManager, default_batch_size: int = 50000):
        self.memory_manager = memory_manager
        # Use fixed batch size to avoid dynamic adjustments
        try:
            env_bs = int(os.environ.get('ARROW_BATCH_SIZE', '0'))
        except Exception:
            env_bs = 0
        self.default_batch_size = env_bs if env_bs > 0 else int(default_batch_size)
        self.chunk_stats = {
            'total_files_loaded': 0,
            'total_chunks_processed': 0,
            'memory_cleanups_triggered': 0
        }
    
    def load_file_chunked(self, file_path: Path, 
                         required_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load large files in chunks to prevent memory overflow
        
        Args:
            file_path: File path
            required_columns: Required columns
            
        Returns:
            Loaded DataFrame
        """
        try:
            # Use fixed, pre-determined parquet read batch size
            optimal_batch_size = self.default_batch_size
            
            # Use PyArrow streaming read
            parquet_file = pq.ParquetFile(file_path)
            
            # Filter columns
            if required_columns:
                available_columns = [col for col in required_columns 
                                   if col in parquet_file.schema.names]
            else:
                available_columns = None
            
            # Read in chunks
            chunks = []
            total_rows = 0
            chunk_count = 0
            
            for batch in parquet_file.iter_batches(columns=available_columns, batch_size=optimal_batch_size):
                # Check memory status
                if self.memory_manager.should_trigger_cleanup():
                    logger.info(f"File loading process triggered memory cleanup: {file_path.name}")
                    self.memory_manager.trigger_memory_cleanup()
                    self.chunk_stats['memory_cleanups_triggered'] += 1
                
                # Convert to DataFrame
                chunk_df = batch.to_pandas()
                chunks.append(chunk_df)
                total_rows += len(chunk_df)
                chunk_count += 1
                
                # If memory pressure is high, process and release chunk immediately
                if self.memory_manager.should_trigger_aggressive_cleanup():
                    # If there are already multiple chunks, merge and clean up first
                    if len(chunks) > 3:
                        partial_df = pd.concat(chunks[:-1], ignore_index=True)
                        chunks = [partial_df, chunks[-1]]
                        gc.collect()
            
            # Merge all chunks
            if chunks:
                df = pd.concat(chunks, ignore_index=True)
                logger.debug(f"File loading completed: {file_path.name}, "
                           f"{total_rows:,} rows, {chunk_count} chunks")
            else:
                df = pd.DataFrame()
            
            # Clean up chunks to release memory
            del chunks
            gc.collect()
            
            self.chunk_stats['total_files_loaded'] += 1
            self.chunk_stats['total_chunks_processed'] += chunk_count
            
            return df
            
        except Exception as e:
            logger.error(f"Chunked file loading failed {file_path}: {e}")
            raise

    def iter_file_batches(self, file_path: Path,
                          required_columns: Optional[List[str]] = None) -> Iterator[pd.DataFrame]:
        """Yield DataFrame per parquet batch to minimize first-batch latency"""
        try:
            optimal_batch_size = self.default_batch_size
            parquet_file = pq.ParquetFile(file_path)

            if required_columns:
                available_columns = [col for col in required_columns if col in parquet_file.schema.names]
            else:
                available_columns = None

            for batch in parquet_file.iter_batches(columns=available_columns, batch_size=optimal_batch_size):
                if self.memory_manager.should_trigger_cleanup():
                    self.memory_manager.trigger_memory_cleanup()
                    self.chunk_stats['memory_cleanups_triggered'] += 1
                yield batch.to_pandas()
        except Exception as e:
            logger.error(f"Iter batch loading failed {file_path}: {e}")
            return
    
    def get_stats(self) -> Dict[str, Any]:
        """Get loading statistics"""
        return self.chunk_stats.copy()


class OptimizedStreamingDataLoader:
    """Optimized streaming data loader with integrated adaptive memory management"""
    
    def __init__(self,
                 data_dir: str,
                 memory_manager: Optional[AdaptiveMemoryManager] = None,
                 max_workers: int = 4,
                 enable_async_loading: bool = True):
        """
        Initialize optimized streaming data loader
        
        Args:
            data_dir: Data directory
            memory_manager: Memory manager
            max_workers: Number of async loading worker threads
            enable_async_loading: Whether to enable async loading
        """
        self.data_dir = Path(data_dir)
        self.memory_manager = memory_manager or create_memory_manager()
        self.max_workers = max_workers
        self.enable_async_loading = enable_async_loading
        
        # Core components
        self.cache = OptimizedStreamingCache(self.memory_manager)
        self.chunked_loader = ChunkedFileLoader(self.memory_manager)
        
        # Async loading
        self.executor = ThreadPoolExecutor(max_workers=max_workers) if enable_async_loading else None
        self.prefetch_queue = deque(maxlen=max_workers * 2)
        
        # Get data files
        self.data_files = self._get_data_files()
        
        # Estimate average file size and update cache configuration
        if self.data_files:
            avg_file_size_mb = self._estimate_average_file_size()
            self.cache.update_limits(avg_file_size_mb)
        
        logger.info(f"Optimized streaming data loader initialized:")
        logger.info(f"  Data directory: {data_dir}")
        logger.info(f"  Data files: {len(self.data_files)} files")
        logger.info(f"  Async loading: {enable_async_loading}")
        logger.info(f"  Worker threads: {max_workers}")
    
    def _get_data_files(self) -> List[Path]:
        """Get all data files sorted by date"""
        files = list(self.data_dir.glob("*.parquet"))
        
        def extract_date(file_path):
            try:
                date_str = file_path.stem
                if len(date_str) == 8 and date_str.isdigit():  # YYYYMMDD
                    return datetime.strptime(date_str, "%Y%m%d")
                elif len(date_str) == 10:  # YYYY-MM-DD
                    return datetime.strptime(date_str, "%Y-%m-%d")
                else:
                    return datetime.min
            except:
                return datetime.min
        
        files.sort(key=extract_date)
        return files
    
    def _estimate_average_file_size(self) -> float:
        """Estimate average file size"""
        if not self.data_files:
            return 300.0  # Default estimation
        
        # Sample a few files for estimation
        sample_size = min(5, len(self.data_files))
        sample_files = self.data_files[:sample_size]
        
        total_size_mb = 0
        for file_path in sample_files:
            size_mb = self.memory_manager.estimate_data_memory_usage(file_path)
            total_size_mb += size_mb
        
        avg_size_mb = total_size_mb / sample_size
        logger.info(f"Estimated average file size: {avg_size_mb:.1f}MB")
        return avg_size_mb
    
    def load_file_with_cache(self, file_path: Path,
                           required_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load file with cache
        
        Args:
            file_path: File path
            required_columns: Required columns
            
        Returns:
            Loaded DataFrame
        """
        cache_key = file_path.name
        
        # Attempt to retrieve from cache
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            logger.debug(f"Cache hit: {cache_key}")
            return cached_data
        
        # Load file
        logger.debug(f"Loading file: {file_path.name}")
        df = self.chunked_loader.load_file_chunked(file_path, required_columns)
        
        # Add to cache (if memory allows)
        if not self.memory_manager.should_trigger_cleanup():
            self.cache.put(cache_key, df)
        
        return df
    
    def stream_by_date_range(self, 
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           required_columns: Optional[List[str]] = None) -> Iterator[pd.DataFrame]:
        """
        Stream data by date range
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            required_columns: Required columns
            
        Yields:
            Daily DataFrame
        """
        # Filter by date range
        filtered_files = self._filter_files_by_date(start_date, end_date)
        
        for i, file_path in enumerate(filtered_files):
            try:
                # Prefetch next batch of files (if async loading enabled)
                if self.enable_async_loading and i % 3 == 0:
                    self._start_prefetch(filtered_files[i:i+3], required_columns)
                
                # Stream current file by parquet batches to reduce latency
                yielded = False
                for batch_df in self.chunked_loader.iter_file_batches(file_path, required_columns):
                    if batch_df is not None and not batch_df.empty:
                        yield batch_df
                        yielded = True
                # Fallback to full file if no batches yielded
                if not yielded:
                    df = self.load_file_with_cache(file_path, required_columns)
                    if not df.empty:
                        yield df
                
                # Periodically check memory and trigger cleanup
                if i % 5 == 0:
                    if self.memory_manager.should_trigger_cleanup():
                        logger.info("Streaming load process triggered memory cleanup")
                        self.memory_manager.trigger_memory_cleanup()
                
            except Exception as e:
                logger.error(f"Loading file failed {file_path}: {e}")
                continue
    
    def _filter_files_by_date(self, start_date: Optional[Union[str, date]],
                            end_date: Optional[Union[str, date]]) -> List[Path]:
        """Filter files by date"""
        if not start_date and not end_date:
            return self.data_files

        filtered_files = []
        for file_path in self.data_files:
            file_date = self._extract_date_from_filename(file_path)
            if file_date:
                try:
                    # Convert to datetime objects for proper date comparison
                    file_date_obj = datetime.strptime(file_date, "%Y-%m-%d")

                    # Normalize start_date/end_date which might be str or datetime.date
                    if start_date:
                        if isinstance(start_date, (datetime, date)):
                            start_date_obj = datetime.combine(start_date, datetime.min.time()) if isinstance(start_date, date) and not isinstance(start_date, datetime) else start_date
                        else:
                            start_date_obj = datetime.strptime(str(start_date), "%Y-%m-%d")
                        if file_date_obj < start_date_obj:
                            continue

                    if end_date:
                        if isinstance(end_date, (datetime, date)):
                            end_date_obj = datetime.combine(end_date, datetime.min.time()) if isinstance(end_date, date) and not isinstance(end_date, datetime) else end_date
                        else:
                            end_date_obj = datetime.strptime(str(end_date), "%Y-%m-%d")
                        if file_date_obj > end_date_obj:
                            continue

                    filtered_files.append(file_path)
                except ValueError:
                    # Skip files with invalid date format
                    continue

        return filtered_files
    
    def _extract_date_from_filename(self, file_path: Path) -> Optional[str]:
        """Extract date from filename"""
        try:
            filename = file_path.stem
            if len(filename) == 8 and filename.isdigit():  # YYYYMMDD
                date_obj = datetime.strptime(filename, "%Y%m%d")
                return date_obj.strftime("%Y-%m-%d")
            elif len(filename) == 10:  # YYYY-MM-DD
                return filename
        except:
            pass
        return None
    
    def _start_prefetch(self, file_paths: List[Path], 
                       required_columns: Optional[List[str]] = None):
        """Start async prefetch"""
        if not self.enable_async_loading:
            return
        
        # Clear prefetch queue
        self.prefetch_queue.clear()
        
        # Submit async loading tasks
        futures = []
        for file_path in file_paths:
            if not self.cache.get(file_path.name):  # Prefetch only uncached files
                future = self.executor.submit(
                    self.chunked_loader.load_file_chunked, 
                    file_path, 
                    required_columns
                )
                futures.append((file_path, future))
        
        # Collect results
        for file_path, future in futures:
            try:
                df = future.result(timeout=120)  # 120file
                self.prefetch_queue.append((file_path.name, df))
            except Exception as e:
                logger.error(f"Prefetch file failed {file_path}: {e}")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        return {
            'data_files': len(self.data_files),
            'memory_manager': self.memory_manager.get_stats(),
            'cache': self.cache.get_stats(),
            'chunked_loader': self.chunked_loader.get_stats(),
            'prefetch_queue_size': len(self.prefetch_queue),
            'async_loading_enabled': self.enable_async_loading
        }
    
    def cleanup(self):
        """Clean up resources"""
        # Stop memory monitoring
        self.memory_manager.stop_monitoring()
        
        # Clear cache
        self.cache.clear()
        
        # Close thread pool
        if self.executor:
            self.executor.shutdown(wait=True)
        
        logger.info("Streaming data loader cleaned up")
    
    def __enter__(self):
        self.memory_manager.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


class OptimizedStreamingDataset(IterableDataset):
    """Optimized streaming dataset for PyTorch training"""
    
    def __init__(self,
                 data_loader: OptimizedStreamingDataLoader,
                 factor_columns: List[str],
                 target_columns: List[str],
                 sequence_length: int = 20,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 enable_sequence_shuffle: bool = True,
                 shuffle_buffer_size: int = 10000):
        """
        Initialize optimized streaming dataset
        
        Args:
            data_loader: Optimized streaming data loader
            factor_columns: Factor column names
            target_columns: Target column names
            sequence_length: Sequence length
            start_date: Start date
            end_date: End date
            enable_sequence_shuffle: Whether to enable sequence-level shuffle
            shuffle_buffer_size: Shuffle buffer size
        """
        self.data_loader = data_loader
        self.factor_columns = factor_columns
        self.target_columns = target_columns
        self.sequence_length = sequence_length
        self.start_date = start_date
        self.end_date = end_date
        self.enable_sequence_shuffle = enable_sequence_shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        
        # Prepare required columns
        self.required_columns = ['sid'] + factor_columns + target_columns
    
    def __iter__(self):
        """samplerankoutputGPU"""
        # distributed
        rank = 0
        world_size = 1
        try:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
                world_size = dist.get_world_size()
            else:
                rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', 0)))
                world_size = int(os.environ.get('WORLD_SIZE', 1))
        except Exception:
            rank, world_size = 0, 1

        # stockqueue
        per_stock_queue: Dict[Any, deque] = {}
        ddp_counter = 0
        # sample
        buffer: List[Dict[str, torch.Tensor]] = []

        # 
        factor_cols = self.factor_columns
        target_cols = self.target_columns
        L = self.sequence_length

        for df in self.data_loader.stream_by_date_range(
            self.start_date, self.end_date, self.required_columns
        ):
            if df is None or df.empty:
                continue

            # sidsortqueue
            cols = ['sid']
            if 'time' in df.columns:
                df = df.sort_values(['sid', 'time'])
            else:
                df = df.sort_values('sid')

            for sid, g in df.groupby('sid'):
                # initializesidqueue
                if sid not in per_stock_queue:
                    per_stock_queue[sid] = deque(maxlen=L)

                # 
                g_f = g[factor_cols].values if set(factor_cols).issubset(g.columns) else None
                if g_f is None:
                    continue
                # targetnext-steptarget
                g_targets = g[target_cols] if set(target_cols).issubset(g.columns) else None
                if g_targets is None:
                    continue

                for idx in range(len(g)):
                    row_f = g_f[idx]
                    if np.isnan(row_f).any():
                        # packageNaNfeatures
                        continue
                    per_stock_queue[sid].append(row_f)

                    # queuelengthLusagenext-steptarget
                    next_idx = idx + 1
                    if len(per_stock_queue[sid]) == L and next_idx < len(g_targets):
                        target_row = g_targets.iloc[next_idx]
                        # target
                        if any(col not in target_row.index or pd.isna(target_row[col]) for col in target_cols):
                            continue
                        target_vals = [float(target_row[col]) for col in target_cols]
                        # target1Dnum_targetsDataLoader(B, T)
                        targets_tensor = torch.tensor(target_vals, dtype=torch.float32)

                        features_np = np.stack(per_stock_queue[sid], axis=0)
                        if features_np.shape[0] != L or np.isnan(features_np).any():
                            continue

                        # Map stock id to bounded vocabulary to avoid embedding OOB
                        try:
                            raw_sid = int(sid) if str(sid).isdigit() else (abs(hash(str(sid))))
                        except Exception:
                            raw_sid = abs(hash(str(sid)))
                        num_stocks_bucket = int(os.environ.get('NUM_STOCKS', '100000'))
                        safe_sid = int(abs(raw_sid) % max(1, num_stocks_bucket))
                        sample = {
                            'features': torch.from_numpy(features_np).float(),
                            'targets': targets_tensor,
                            'stock_id': torch.LongTensor([safe_sid]),
                            'sequence_length': L,
                        }
                        # DDPdataoptimizebuffersample
                        buffer.append(sample)
                        
                        # buffersizerankbatchoutput
                        if len(buffer) >= max(64, world_size * 16):  # rank16sample
                            start_idx = rank * len(buffer) // world_size
                            end_idx = (rank + 1) * len(buffer) // world_size
                            for i in range(start_idx, end_idx):
                                yield buffer[i]
                            buffer.clear()
                        
                        ddp_counter += 1

        # processbufferdata
        if buffer:
            start_idx = rank * len(buffer) // world_size
            end_idx = (rank + 1) * len(buffer) // world_size
            for i in range(start_idx, min(end_idx, len(buffer))):
                yield buffer[i]
            buffer.clear()
    
    def _create_sequences_from_daily_data(self, df: pd.DataFrame) -> Iterator[Dict[str, torch.Tensor]]:
        """Create sequences from daily data"""
        if len(df) < self.sequence_length:
            return
        
        # Ensure data is sorted by time and stock ID
        if 'time' in df.columns:
            df = df.sort_values(['sid', 'time'])
        else:
            df = df.sort_values('sid')
        
        # Group by stock
        for stock_id, stock_group in df.groupby('sid'):
            if len(stock_group) < self.sequence_length:
                continue
            
            # Create sliding window sequences with proper future prediction
            # Need at least sequence_length + 1 points to predict future
            for i in range(len(stock_group) - self.sequence_length):
                # Feature sequences (historical data only)
                feature_data = stock_group[self.factor_columns].iloc[i:i+self.sequence_length].values
                # Replace NaNs in features to avoid dropping batches; targets remain strict
                if np.isnan(feature_data).any():
                    feature_data = np.nan_to_num(feature_data, nan=0.0)
                
                # Target values (FUTURE data - next time point after sequence)
                target_row = stock_group[self.target_columns].iloc[i + self.sequence_length]
                
                # Check data integrity
                # validationfilterNaNbatch
                if feature_data.size == 0:
                    continue
                if any(col not in target_row.index or np.isnan(target_row[col]) for col in self.target_columns):
                    continue
                
                # target1Dnum_targetsDataLoader(B, T)
                target_vals = [float(target_row[col]) for col in self.target_columns]
                targets_tensor = torch.tensor(target_vals, dtype=torch.float32)

                # Map stock id to bounded vocabulary to avoid embedding OOB
                try:
                    raw_sid = int(stock_id) if str(stock_id).isdigit() else (abs(hash(str(stock_id))))
                except Exception:
                    raw_sid = abs(hash(str(stock_id)))
                num_stocks_bucket = int(os.environ.get('NUM_STOCKS', '100000'))
                safe_sid = int(abs(raw_sid) % max(1, num_stocks_bucket))
                yield {
                    'features': torch.FloatTensor(feature_data),
                    'targets': targets_tensor,
                    'stock_id': torch.LongTensor([safe_sid]),
                    'sequence_length': self.sequence_length
                }


def create_optimized_dataloaders(
    data_dir: str,
    factor_columns: List[str],
    target_columns: List[str],
    train_dates: Tuple[str, str],
    val_dates: Tuple[str, str],
    test_dates: Tuple[str, str],
    sequence_length: int = 20,
    batch_size: int = 64,
    num_workers: int = 4,
    memory_config: Optional[Dict[str, Any]] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create optimized data loader
    
    Args:
        data_dir: Data directory
        factor_columns: Factor column names
        target_columns: Target column names
        train_dates: Training date range
        val_dates: Validation date range
        test_dates: Test date range
        sequence_length: Sequence length
        batch_size: Batch size
        num_workers: Number of worker threads
        memory_config: Memory configuration
        
    Returns:
        Training, validation, and test data loaders
    """
    # Create memory manager
    memory_manager = create_memory_manager(memory_config)
    
    # Create optimized streaming data loader
    streaming_loader = OptimizedStreamingDataLoader(
        data_dir=data_dir,
        memory_manager=memory_manager,
        max_workers=max(2, num_workers//2),
        enable_async_loading=True
    )
    
    # Create dataset
    train_dataset = OptimizedStreamingDataset(
        streaming_loader, factor_columns, target_columns,
        sequence_length, train_dates[0], train_dates[1],
        enable_sequence_shuffle=True
    )
    
    val_dataset = OptimizedStreamingDataset(
        streaming_loader, factor_columns, target_columns,
        sequence_length, val_dates[0], val_dates[1],
        enable_sequence_shuffle=False
    )
    
    test_dataset = OptimizedStreamingDataset(
        streaming_loader, factor_columns, target_columns,
        sequence_length, test_dates[0], test_dates[1],
        enable_sequence_shuffle=False
    )
    
    # Create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,  # IterableDataset does not support multiprocessing
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test optimized streaming data loader
    data_dir = "/Users/scratch/Documents/My Code/Projects/factor_forecasting/data"
    
    # Create memory manager
    memory_manager = create_memory_manager()
    
    with OptimizedStreamingDataLoader(data_dir, memory_manager) as loader:
        print("=== Optimized streaming data loader test ===")
        
        # Get statistics
        stats = loader.get_comprehensive_stats()
        print(f"Number of data files: {stats['data_files']}")
        print(f"Memory status: {stats['memory_manager']['current_status']['system']['usage_ratio']:.1%}")
        
        # Test data stream
        count = 0
        for df in loader.stream_by_date_range():
            print(f"Loaded file {count+1}: {len(df)} rows")
            count += 1
            if count >= 1:  # Test only 1 file
                break
        
        print(f"Final statistics: {loader.get_comprehensive_stats()}")
