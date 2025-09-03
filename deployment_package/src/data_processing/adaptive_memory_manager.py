"""
Adaptive Memory Management System: Optimized memory management for quantitative financial time series forecasting
Solves memory explosion issues caused by large data files (300MB+)
"""
import os
import gc
import logging
import psutil
import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import threading
import time

logger = logging.getLogger(__name__)


@dataclass
class MemoryBudget:
    """Memory budget configuration"""
    total_system_gb: float
    reserved_system_gb: float = 2.0  # Reserved memory for system
    max_cache_ratio: float = 0.3     # Maximum cache usage ratio
    max_batch_ratio: float = 0.2     # Maximum single batch usage ratio
    gpu_reserved_gb: float = 1.0     # GPU reserved memory
    
    @property
    def available_system_gb(self) -> float:
        return self.total_system_gb - self.reserved_system_gb
    
    @property
    def max_cache_gb(self) -> float:
        return self.available_system_gb * self.max_cache_ratio
    
    @property
    def max_batch_gb(self) -> float:
        return self.available_system_gb * self.max_batch_ratio


class AdaptiveMemoryManager:
    """
    Adaptive Memory Manager:
    1. Dynamic monitoring of system and GPU memory usage
    2. Adaptive adjustment of batch size and caching strategies
    3. Prevention of memory overflow and GPU memory shortage
    4. Optimization of data loading and processing pipeline
    """
    
    def __init__(self, 
                 memory_budget: Optional[MemoryBudget] = None,
                 monitoring_interval: float = 1.0,
                 critical_threshold: float = 0.98,
                 warning_threshold: float = 0.95):
        """
        Initialize adaptive memory manager
        
        Args:
            memory_budget: Memory budget configuration
            monitoring_interval: Monitoring interval (seconds)
            critical_threshold: Critical threshold
            warning_threshold: Warning threshold
        """
        # Auto-detect system memory
        system_memory_gb = psutil.virtual_memory().total / (1024**3)
        self.memory_budget = memory_budget or MemoryBudget(
            total_system_gb=system_memory_gb
        )
        
        self.monitoring_interval = monitoring_interval
        self.critical_threshold = critical_threshold
        self.warning_threshold = warning_threshold
        
        # Current state
        # fixedbatchrunadjustment
        self.current_batch_size = 50000
        self.adaptive_cache_size = 5     # Adaptive cache size
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread = None
        self.stats = {
            'memory_warnings': 0,
            'memory_cleanups': 0,
            'batch_size_adjustments': 0,
            'cache_evictions': 0
        }
        
        # Thread lock
        self.lock = threading.Lock()
        
        logger.info(f"Adaptive memory manager initialization completed:")
        logger.info(f"  Total system memory: {system_memory_gb:.1f}GB")
        logger.info(f"  Available memory: {self.memory_budget.available_system_gb:.1f}GB") 
        logger.info(f"  Max cache: {self.memory_budget.max_cache_gb:.1f}GB")
        logger.info(f"  Max batch: {self.memory_budget.max_batch_gb:.1f}GB")
        
    def start_monitoring(self):
        """Start memory monitoring"""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_memory,
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        logger.info("Memory monitoring stopped")
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Get current memory status"""
        # System memory
        memory = psutil.virtual_memory()
        system_usage_ratio = memory.percent / 100.0
        system_available_gb = memory.available / (1024**3)
        
        # GPU memory
        gpu_info = {}
        if torch.cuda.is_available():
            for device_id in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(device_id)
                reserved = torch.cuda.memory_reserved(device_id)
                total = torch.cuda.get_device_properties(device_id).total_memory
                
                gpu_info[f'gpu_{device_id}'] = {
                    'allocated_gb': allocated / (1024**3),
                    'reserved_gb': reserved / (1024**3),
                    'total_gb': total / (1024**3),
                    'usage_ratio': (allocated + reserved) / total
                }
        
        return {
            'system': {
                'total_gb': memory.total / (1024**3),
                'available_gb': system_available_gb,
                'used_gb': memory.used / (1024**3),
                'usage_ratio': system_usage_ratio,
                'is_critical': system_usage_ratio > self.critical_threshold,
                'is_warning': system_usage_ratio > self.warning_threshold
            },
            'gpu': gpu_info,
            'budget': {
                'max_cache_gb': self.memory_budget.max_cache_gb,
                'max_batch_gb': self.memory_budget.max_batch_gb,
                'available_system_gb': self.memory_budget.available_system_gb
            },
            'adaptive': {
                'current_batch_size': self.current_batch_size,
                'adaptive_cache_size': self.adaptive_cache_size
            }
        }
    
    def calculate_optimal_batch_size(self, 
                                   data_sample_mb: float,
                                   sequence_length: int = 20,
                                   feature_dim: int = 100) -> int:
        """Deprecated: return fixed current_batch_size for compatibility."""
        return int(self.current_batch_size)
    
    def calculate_optimal_cache_size(self, avg_file_size_mb: float) -> int:
        """
        Calculate optimal cache size based on memory estimation
        
        Args:
            avg_file_size_mb: Average file size (MB)
            
        Returns:
            Optimal cache item count
        """
        max_cache_mb = self.memory_budget.max_cache_gb * 1024
        
        if avg_file_size_mb > 0:
            optimal_cache_size = int(max_cache_mb / avg_file_size_mb)
        else:
            optimal_cache_size = 5
        
        # Set reasonable range
        optimal_cache_size = max(1, min(optimal_cache_size, 20))
        
        if optimal_cache_size != self.adaptive_cache_size:
            logger.info(f"Cache size adjustment: {self.adaptive_cache_size} -> {optimal_cache_size}")
            self.adaptive_cache_size = optimal_cache_size
        
        return optimal_cache_size
    
    def should_trigger_cleanup(self) -> bool:
        """Determine whether to trigger memory cleanup"""
        status = self.get_memory_status()
        
        # System memory warning
        if status['system']['is_warning']:
            return True
        
        # GPU memory warning
        for gpu_info in status['gpu'].values():
            if gpu_info['usage_ratio'] > self.warning_threshold:
                return True
        
        return False
    
    def should_trigger_aggressive_cleanup(self) -> bool:
        """Determine whether to trigger aggressive cleanup"""
        status = self.get_memory_status()
        
        # System memory critical
        if status['system']['is_critical']:
            return True
        
        # GPU memory critical
        for gpu_info in status['gpu'].values():
            if gpu_info['usage_ratio'] > self.critical_threshold:
                return True
        
        return False
    
    def trigger_memory_cleanup(self, aggressive: bool = False):
        """
        Trigger memory cleanup
        
        Args:
            aggressive: Whether to perform aggressive cleanup
        """
        with self.lock:
            logger.info(f"Triggering memory cleanup (aggressive mode: {aggressive})")
            
            # Basic cleanup
            gc.collect()
            
            # GPU memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Aggressive cleanup
            if aggressive:
                # Force multiple garbage collections
                for _ in range(3):
                    gc.collect()
                
                # Clear all GPU caches
                if torch.cuda.is_available():
                    for device_id in range(torch.cuda.device_count()):
                        with torch.cuda.device(device_id):
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
            
            self.stats['memory_cleanups'] += 1
            
            # Record post-cleanup status
            status = self.get_memory_status()
            logger.info(f"Post-cleanup memory status: system={status['system']['usage_ratio']:.1%}")
    
    def _monitor_memory(self):
        """Background memory monitoring thread"""
        logger.info("Memory monitoring thread started")
        
        while self.monitoring_active:
            try:
                if self.should_trigger_aggressive_cleanup():
                    self.trigger_memory_cleanup(aggressive=True)
                elif self.should_trigger_cleanup():
                    self.trigger_memory_cleanup(aggressive=False)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(5.0)  # Wait longer after an error
        
        logger.info("Memory monitoring thread ended")
    
    def estimate_data_memory_usage(self, file_path: Path) -> float:
        """
        Estimate memory usage of data file
        
        Args:
            file_path: Data file path
            
        Returns:
            Estimated memory usage (MB)
        """
        try:
            if file_path.suffix == '.parquet':
                import pyarrow.parquet as pq
                
                # Use PyArrow to get file statistics
                parquet_file = pq.ParquetFile(file_path)
                
                # Estimate memory usage after decompression
                metadata = parquet_file.metadata
                file_size_mb = file_path.stat().st_size / (1024**2)
                
                # General parquet compression ratio is 3-5 times
                estimated_memory_mb = file_size_mb * 4
                
                return estimated_memory_mb
            else:
                # For other file types, use file size estimation
                return file_path.stat().st_size / (1024**2)
                
        except Exception as e:
            logger.warning(f"Unable to estimate file memory usage {file_path}: {e}")
            return 300.0  # Default estimated value
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        return {
            **self.stats,
            'current_status': self.get_memory_status(),
            'memory_budget': {
                'total_system_gb': self.memory_budget.total_system_gb,
                'available_system_gb': self.memory_budget.available_system_gb,
                'max_cache_gb': self.memory_budget.max_cache_gb,
                'max_batch_gb': self.memory_budget.max_batch_gb
            }
        }
    
    def __enter__(self):
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_monitoring()
    
    def __del__(self):
        try:
            self.stop_monitoring()
        except Exception:
            pass  # Ignore cleanup errors


def create_memory_manager(config: Optional[Dict[str, Any]] = None) -> AdaptiveMemoryManager:
    """
    Factory function to create a memory manager
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured memory manager
    """
    config = config or {}
    
    # Auto-detect system configuration
    system_memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Adjust default configuration based on system memory
    if system_memory_gb >= 64:
        # Large memory system
        memory_budget = MemoryBudget(
            total_system_gb=system_memory_gb,
            reserved_system_gb=4.0,
            max_cache_ratio=0.4,
            max_batch_ratio=0.3
        )
    elif system_memory_gb >= 32:
        # Medium memory system
        memory_budget = MemoryBudget(
            total_system_gb=system_memory_gb,
            reserved_system_gb=3.0,
            max_cache_ratio=0.3,
            max_batch_ratio=0.2
        )
    else:
        # Small memory system
        memory_budget = MemoryBudget(
            total_system_gb=system_memory_gb,
            reserved_system_gb=2.0,
            max_cache_ratio=0.2,
            max_batch_ratio=0.15
        )
    
    return AdaptiveMemoryManager(
        memory_budget=memory_budget,
        **config
    )


# Example usage
if __name__ == "__main__":
    # Create memory manager
    memory_manager = create_memory_manager()
    
    # Start monitoring
    memory_manager.start_monitoring()
    
    try:
        # Simulate data processing
        print("Current memory status:")
        status = memory_manager.get_memory_status()
        print(f"System memory usage: {status['system']['usage_ratio']:.1%}")
        
        # Calculate optimal batch size
        batch_size = memory_manager.calculate_optimal_batch_size(
            data_sample_mb=0.5,
            sequence_length=20,
            feature_dim=100
        )
        print(f"Recommended batch size: {batch_size}")
        
        # Calculate optimal cache size
        cache_size = memory_manager.calculate_optimal_cache_size(avg_file_size_mb=300)
        print(f"Recommended cache size: {cache_size}")
        
        # Wait for a while to see the monitoring
        time.sleep(5)
        
    finally:
        memory_manager.stop_monitoring()
        print("Memory manager statistics:", memory_manager.get_stats())
