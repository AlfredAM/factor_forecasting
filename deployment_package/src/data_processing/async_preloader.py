"""
Asynchronous data preloading system for efficient pipeline processing.
Provides background data loading with queue management and error handling.
"""
import asyncio
import threading
import queue
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Callable, Iterator
from pathlib import Path
import pandas as pd
import numpy as np
import time
from datetime import datetime
import gc

logger = logging.getLogger(__name__)


class AsyncDataPreloader:
    """Asynchronous data preloader with queue management.

    Supports two modes:
    1) File mode: prefetch items from a file list by calling data_loader_func(file_path).
    2) Batch mode: prefetch batches from a DataLoader returned by data_loader_func().
    """
    
    def __init__(self,
                 data_loader_func: Callable,
                 queue_size: int = 10,
                 num_workers: int = 4,
                 prefetch_size: int = 3,
                 timeout: float = 30.0):
        """
        Args:
            data_loader_func: Function to load data files
            queue_size: Maximum size of the preload queue
            num_workers: Number of worker threads
            prefetch_size: Number of items to prefetch ahead
            timeout: Timeout for queue operations
        """
        self.data_loader_func = data_loader_func
        self.queue_size = queue_size
        self.num_workers = num_workers
        self.prefetch_size = prefetch_size
        self.timeout = timeout
        
        # Threading components
        self.data_queue = queue.Queue(maxsize=queue_size)
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.preload_thread = None
        self.stop_event = threading.Event()
        
        # Statistics
        self.stats = {
            'items_loaded': 0,
            'items_served': 0,
            'load_errors': 0,
            'queue_full_events': 0,
            'total_load_time': 0.0
        }
        
        self.is_running = False
        self.mode = 'file'  # 'file' or 'batch'
    
    def start_preloading(self, file_list: Optional[List[str]] = None):
        """Start the preloading process.

        Args:
            file_list: Optional list of file paths. If None, switch to batch mode and
                       consume batches from a DataLoader returned by data_loader_func().
        """
        if self.is_running:
            logger.warning("Preloader already running")
            return
        
        self.stop_event.clear()
        self.file_list = file_list.copy() if file_list is not None else []
        self.file_index = 0
        
        if file_list is None:
            self.mode = 'batch'
            target_fn = self._batch_worker
        else:
            self.mode = 'file'
            target_fn = self._preload_worker

        self.preload_thread = threading.Thread(target=target_fn, daemon=True)
        self.preload_thread.start()
        self.is_running = True
        
        if self.mode == 'file':
            logger.info(f"Started preloading with {len(self.file_list)} files")
        else:
            logger.info("Started preloading from DataLoader (batch mode)")
    
    def stop_preloading(self):
        """Stop the preloading process."""
        if not self.is_running:
            return
        
        self.stop_event.set()
        
        # Clear the queue
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except queue.Empty:
                break
        
        # Wait for thread to finish
        if self.preload_thread and self.preload_thread.is_alive():
            self.preload_thread.join(timeout=5.0)
        
        self.executor.shutdown(wait=True)
        self.is_running = False
        
        logger.info("Preloader stopped")
    
    def get_next_batch(self) -> Optional[Dict[str, Any]]:
        """Get the next preloaded batch."""
        try:
            item = self.data_queue.get(timeout=self.timeout)
            self.stats['items_served'] += 1
            
            if isinstance(item, Exception):
                raise item
            
            return item
            
        except queue.Empty:
            logger.warning("Queue timeout - no data available")
            return None
    
    def _preload_worker(self):
        """Background worker for preloading data."""
        futures = set()
        
        while not self.stop_event.is_set() and self.file_index < len(self.file_list):
            # Submit new tasks up to prefetch_size
            while (len(futures) < self.prefetch_size and 
                   self.file_index < len(self.file_list) and
                   not self.stop_event.is_set()):
                
                file_path = self.file_list[self.file_index]
                future = self.executor.submit(self._load_individual_file, file_path)
                futures.add(future)
                self.file_index += 1
            
            # Process completed futures
            if futures:
                # Wait for at least one future to complete
                completed_futures = []
                for future in as_completed(futures, timeout=1.0):
                    completed_futures.append(future)
                    break  # Process one at a time
                
                for future in completed_futures:
                    futures.remove(future)
                    
                    try:
                        result = future.result()
                        self._put_in_queue(result)
                    except Exception as e:
                        logger.error(f"Error loading data: {e}")
                        self.stats['load_errors'] += 1
                        self._put_in_queue(e)
            
            # Brief pause to prevent busy waiting
            time.sleep(0.01)
        
        # Cancel remaining futures
        for future in futures:
            future.cancel()
        
        logger.info("Preload worker finished")

    def _batch_worker(self):
        """Background worker that pulls batches from a DataLoader."""
        try:
            dl = self.data_loader_func()
        except Exception as e:
            self._put_in_queue(e)
            logger.error(f"Failed to get DataLoader: {e}")
            return

        try:
            for batch in dl:
                if self.stop_event.is_set():
                    break
                self._put_in_queue(batch)
        except Exception as e:
            logger.error(f"Error iterating DataLoader: {e}")
            self._put_in_queue(e)
        finally:
            # Signal completion by putting a sentinel None if queue empty, non-blocking
            try:
                self.data_queue.put_nowait(None)
            except queue.Full:
                pass
            logger.info("Batch worker finished")
    
    def _load_individual_file(self, file_path: str) -> Dict[str, Any]:
        """Load a single data file."""
        start_time = time.time()
        
        try:
            data = self.data_loader_func(file_path)
            
            load_time = time.time() - start_time
            self.stats['total_load_time'] += load_time
            self.stats['items_loaded'] += 1
            
            return {
                'data': data,
                'file_path': file_path,
                'load_time': load_time,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            raise
    
    def _put_in_queue(self, item):
        """Put item in queue with timeout handling."""
        try:
            self.data_queue.put(item, timeout=self.timeout)
        except queue.Full:
            logger.warning("Queue full, dropping item")
            self.stats['queue_full_events'] += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get preloader statistics."""
        avg_load_time = (self.stats['total_load_time'] / 
                        max(1, self.stats['items_loaded']))
        
        return {
            **self.stats,
            'queue_size': self.data_queue.qsize(),
            'is_running': self.is_running,
            'files_remaining': len(self.file_list) - self.file_index if hasattr(self, 'file_list') else 0,
            'avg_load_time': avg_load_time
        }

    def __iter__(self):
        """Yield preloaded items. In file mode yields loaded data; in batch mode yields batches."""
        while True:
            item = self.get_next_batch()
            if item is None:
                break
            if self.mode == 'file':
                if isinstance(item, dict) and 'data' in item:
                    yield item['data']
                else:
                    yield item
            else:
                yield item
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_preloading()
        
    def __del__(self):
        """Ensure cleanup on deletion."""
        try:
            self.stop_preloading()
        except Exception:
            pass  # Ignore errors during cleanup


class StreamingDataPipeline:
    """High-level streaming data pipeline with preloading."""
    
    def __init__(self,
                 data_dir: str,
                 batch_processor: Callable,
                 file_pattern: str = "*.parquet",
                 preload_queue_size: int = 10,
                 num_workers: int = 4):
        """
        Args:
            data_dir: Directory containing data files
            batch_processor: Function to process loaded batches
            file_pattern: Pattern for finding data files
            preload_queue_size: Size of preload queue
            num_workers: Number of worker threads
        """
        self.data_dir = Path(data_dir)
        self.batch_processor = batch_processor
        self.file_pattern = file_pattern
        
        # Create data loader function
        def load_file(file_path: str) -> pd.DataFrame:
            return pd.read_parquet(file_path)
        
        self.preloader = AsyncDataPreloader(
            data_loader_func=load_file,
            queue_size=preload_queue_size,
            num_workers=num_workers
        )
        
        self.pipeline_stats = {
            'batches_processed': 0,
            'total_processing_time': 0.0,
            'errors': 0
        }
    
    def process_files(self, 
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> Iterator[Dict[str, Any]]:
        """Process files with asynchronous preloading."""
        
        # Get file list
        file_list = self._get_file_list(start_date, end_date)
        if not file_list:
            logger.warning("No files found to process")
            return
        
        logger.info(f"Starting pipeline with {len(file_list)} files")
        
        # Start preloading
        self.preloader.start_preloading(file_list)
        
        try:
            processed_count = 0
            
            while processed_count < len(file_list):
                # Get preloaded batch
                batch_data = self.preloader.get_next_batch()
                
                if batch_data is None:
                    logger.warning("No data received, pipeline may be stalled")
                    break
                
                # Process batch
                start_time = time.time()
                try:
                    processed_batch = self.batch_processor(batch_data['data'])
                    
                    processing_time = time.time() - start_time
                    self.pipeline_stats['total_processing_time'] += processing_time
                    self.pipeline_stats['batches_processed'] += 1
                    
                    yield {
                        'processed_data': processed_batch,
                        'file_path': batch_data['file_path'],
                        'load_time': batch_data['load_time'],
                        'processing_time': processing_time,
                        'timestamp': batch_data['timestamp']
                    }
                    
                    processed_count += 1
                    
                except Exception as e:
                    logger.error(f"Error processing batch from {batch_data['file_path']}: {e}")
                    self.pipeline_stats['errors'] += 1
                    processed_count += 1
                    
                # Memory cleanup
                if processed_count % 10 == 0:
                    gc.collect()
                    
        finally:
            self.preloader.stop_preloading()
    
    def _get_file_list(self, start_date: Optional[str], end_date: Optional[str]) -> List[str]:
        """Get filtered list of data files."""
        files = list(self.data_dir.glob(self.file_pattern))
        
        # Filter by date if specified
        if start_date or end_date:
            filtered_files = []
            for file_path in files:
                file_date = file_path.stem
                
                # Convert to standard date format if needed
                try:
                    if len(file_date) == 8:  # YYYYMMDD
                        file_date = datetime.strptime(file_date, "%Y%m%d").strftime("%Y-%m-%d")
                    
                    if start_date and file_date < start_date:
                        continue
                    if end_date and file_date > end_date:
                        continue
                    
                    filtered_files.append(str(file_path))
                except ValueError:
                    logger.warning(f"Invalid date format in filename: {file_path.name}")
                    continue
            
            files = filtered_files
        else:
            files = [str(f) for f in files]
        
        # Sort by filename (date)
        files.sort()
        return files
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        preloader_stats = self.preloader.get_statistics()
        
        avg_processing_time = (self.pipeline_stats['total_processing_time'] / 
                             max(1, self.pipeline_stats['batches_processed']))
        
        return {
            'pipeline': {
                **self.pipeline_stats,
                'avg_processing_time': avg_processing_time
            },
            'preloader': preloader_stats,
            'efficiency': {
                'throughput': self.pipeline_stats['batches_processed'] / max(1, self.pipeline_stats['total_processing_time']),
                'error_rate': self.pipeline_stats['errors'] / max(1, self.pipeline_stats['batches_processed'])
            }
        }


class BatchedAsyncPreloader:
    """Preloader optimized for batch processing."""
    
    def __init__(self,
                 data_sources: List[str],
                 batch_size: int = 5,
                 max_batches_in_memory: int = 3,
                 preprocessing_func: Optional[Callable] = None):
        """
        Args:
            data_sources: List of data source paths
            batch_size: Number of files per batch
            max_batches_in_memory: Maximum batches to keep in memory
            preprocessing_func: Optional preprocessing function
        """
        self.data_sources = data_sources
        self.batch_size = batch_size
        self.max_batches_in_memory = max_batches_in_memory
        self.preprocessing_func = preprocessing_func
        
        self.batch_queue = asyncio.Queue(maxsize=max_batches_in_memory)
        self.current_index = 0
        self.is_loading = False
    
    async def start_async_loading(self):
        """Start asynchronous batch loading."""
        if self.is_loading:
            return
        
        self.is_loading = True
        asyncio.create_task(self._load_batches())
        logger.info("Started async batch loading")
    
    async def get_next_batch(self) -> Optional[List[pd.DataFrame]]:
        """Get next preloaded batch."""
        try:
            batch = await asyncio.wait_for(self.batch_queue.get(), timeout=30.0)
            return batch
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for batch")
            return None
    
    async def _load_batches(self):
        """Background task for loading batches."""
        while self.current_index < len(self.data_sources):
            # Create batch
            batch_sources = self.data_sources[
                self.current_index:self.current_index + self.batch_size
            ]
            
            # Load batch asynchronously
            batch_data = await self._load_batch_async(batch_sources)
            
            # Add to queue
            await self.batch_queue.put(batch_data)
            
            self.current_index += self.batch_size
        
        self.is_loading = False
        logger.info("Async batch loading completed")
    
    async def _load_batch_async(self, file_paths: List[str]) -> List[pd.DataFrame]:
        """Load a batch of files asynchronously."""
        loop = asyncio.get_event_loop()
        
        # Load files in parallel
        tasks = []
        for file_path in file_paths:
            task = loop.run_in_executor(None, self._load_individual_file, file_path)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error loading file: {result}")
            else:
                valid_results.append(result)
        
        return valid_results
    
    def _load_individual_file(self, file_path: str) -> pd.DataFrame:
        """Load individual file with optional preprocessing."""
        df = pd.read_parquet(file_path)
        
        if self.preprocessing_func:
            df = self.preprocessing_func(df)
        
        return df
