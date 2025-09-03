#!/usr/bin/env python3
"""
Unified Complete Training System V2 - Fixed Version
Complete implementation of all 8 core features:
1. Strict prevention of data leakage in time series prediction
2. Streaming and rolling data loading
3. Full GPU utilization optimization
4. Complete checkpoint management
5. TCN with Attention model
6. Scheduled IC correlation reporting
7. Annual rolling training inference
8. Multi-objective correlation optimization

Key fixes:
- Simplified configuration loading (flat YAML structure)
- Adaptable yearly rolling (disabled for limited data)
- Fixed field mapping between YAML and code
- Proper error handling and validation
"""

import os
import sys
import argparse
import logging
import json
import yaml
import signal
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import gc
import socket

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Import components
from src.data_processing.optimized_streaming_loader import OptimizedStreamingDataLoader, OptimizedStreamingDataset
from src.data_processing.adaptive_memory_manager import create_memory_manager
from src.models.advanced_tcn_attention import create_advanced_model
from src.training.quantitative_loss import QuantitativeCorrelationLoss
from src.monitoring.ic_reporter import ICCorrelationReporter

# Global variables
training_should_stop = False

def signal_handler(signum, frame):
    """Handle interrupt signals"""
    global training_should_stop
    print(f"\nReceived signal {signum}, gracefully shutting down...")
    training_should_stop = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

class UnifiedCompleteTrainer:
    """Unified Complete Training System with all 8 features"""
    
    def __init__(self, config: Dict[str, Any], rank: int = 0, world_size: int = 1):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        
        # Auto-enable features based on runtime conditions
        self._auto_enable_features()
        
        # Setup output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(config.get('output_dir', '/tmp/factor_forecasting/outputs')) / f"unified_complete_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.loss_function = None
        self.memory_manager = None
        self.streaming_loader = None
        
        # Monitoring components (Feature 6)
        self.ic_reporter = None
        if config.get('enable_ic_reporting', True) and rank == 0:
            self.ic_reporter = ICCorrelationReporter(
                output_dir=str(self.output_dir / "ic_reports"),
                target_columns=config.get('target_columns', ['intra30m', 'nextT1d', 'ema1d']),
                report_interval=config.get('ic_report_interval', 7200)
            )
            # 启动自动报告线程
            self.ic_reporter.start_automatic_reporting()
        
        # Training state (Feature 4)
        self.start_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_ic = -float('inf')
        self.train_losses = []
        self.val_losses = []
        self.ic_scores = []
        
        if self.rank == 0:
            self.logger.info(f"Unified Complete Training System initialized - Rank {rank}/{world_size}")
            self.logger.info(f"Output directory: {self.output_dir}")
            self.logger.info(f"Device: {self.device}")
            self.logger.info("Integrated 8 core features:")
            self.logger.info("   1. Strict data leakage prevention")
            self.logger.info("   2. Streaming + rolling data loading")
            self.logger.info("   3. Full GPU utilization optimization")
            self.logger.info("   4. Complete checkpoint management")
            self.logger.info("   5. TCN+Attention model")
            self.logger.info("   6. Scheduled IC correlation reporting")
            self.logger.info("   7. Annual rolling training inference")
            self.logger.info("   8. Multi-objective correlation optimization")

    def _auto_enable_features(self):
        """Auto-enable features based on actual runtime conditions"""
        auto = self.config.get('auto_enable_features', True)
        if not auto:
            return
        
        # Auto AMP if CUDA available
        if torch.cuda.is_available():
            self.config['use_mixed_precision'] = True
        
        # Auto distributed if multiple GPUs and not explicitly disabled
        num_gpus = torch.cuda.device_count()
        if num_gpus and num_gpus > 1 and self.config.get('use_distributed', False):
            # Keep distributed training as configured
            pass
        elif num_gpus and num_gpus > 1 and not self.config.get('use_distributed', False):
            # Force disable distributed training if explicitly set to false
            self.config['use_distributed'] = False
        
        # Ensure async streaming enabled implicitly by loader
        # Ensure at least 1 worker
        if self.config.get('num_workers', 0) <= 0:
            self.config['num_workers'] = 1
        
        # Enable IC reporter by default on rank 0
        self.config['enable_ic_reporting'] = True
        
        # Auto enable yearly rolling only when not explicitly set and allowed by auto flag
        try:
            if self.config.get('enable_yearly_rolling', None) is None and self.config.get('auto_enable_yearly_rolling', False):
                data_dir = Path(self.config.get('data_dir', '/nas/feature_v2_10s'))
                files = list(data_dir.glob('*.parquet'))
                years = set()
                for fp in files:
                    stem = fp.stem
                    if len(stem) == 8 and stem.isdigit():
                        years.add(int(stem[:4]))
                    elif len(stem) >= 10 and stem[4] == '-' and stem[7] == '-':
                        years.add(int(stem[:4]))
                if years:
                    min_train_years = self.config.get('min_train_years', 1)
                    if len(years) >= min_train_years + 1:
                        self.config['enable_yearly_rolling'] = True
        except Exception:
            pass

        # configurefilesetup
        # configurefile enable_yearly_rolling dataenable
        explicit_rolling = self.config.get('enable_yearly_rolling', None)
        if explicit_rolling is False:
            # closeenabledisable
            self.config['auto_enable_yearly_rolling'] = False
        elif explicit_rolling is None and self.config.get('auto_enable_yearly_rolling', False):
            try:
                data_dir = Path(self.config.get('data_dir', '/nas/feature_v2_10s'))
                files = list(data_dir.glob('*.parquet'))
                years = set()
                for fp in files:
                    stem = Path(fp).stem
                    if len(stem) == 8 and stem.isdigit():
                        years.add(int(stem[:4]))
                    elif len(stem) >= 10 and stem[4] == '-' and stem[7] == '-':
                        years.add(int(stem[:4]))
                if years:
                    min_train_years = self.config.get('min_train_years', 1)
                    if len(years) >= min_train_years + 1:
                        self.config['enable_yearly_rolling'] = True
            except Exception:
                pass

    def setup_logging(self):
        """Setup logging system"""
        log_file = self.output_dir / f"training_rank_{self.rank}.log"
        
        log_level = self.config.get('log_level', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout) if self.rank == 0 else logging.NullHandler()
            ]
        )
        
        self.logger = logging.getLogger(f'UnifiedComplete_Rank_{self.rank}')

    def setup_distributed(self):
        """Setup distributed training environment (Feature 3)"""
        if self.world_size > 1:
            if self.rank == 0:
                self.logger.info("Initializing distributed training...")
            
            # setupNCCLdebug
            os.environ.setdefault('NCCL_TIMEOUT', '3600')
            os.environ.setdefault('NCCL_DEBUG', 'INFO')
            os.environ.setdefault('NCCL_IB_DISABLE', '1')  # disableInfiniBandnetwork
            os.environ.setdefault('NCCL_P2P_DISABLE', '1')  # disableP2P
            
            # Initialize process group with timeout
            try:
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=self.world_size,
                    rank=self.rank,
                    timeout=timedelta(minutes=30)
            )
            
            torch.cuda.set_device(self.rank)
            
            if self.rank == 0:
                self.logger.info(f"Distributed training initialized - {self.world_size} GPUs")
            except Exception as e:
                self.logger.error(f"Failed to initialize distributed training: {e}")
                # Fallback to single GPU
                self.world_size = 1
                self.rank = 0
                torch.cuda.set_device(0)

    def setup_data_loaders(self):
        """Setup data loaders with streaming and leakage prevention (Features 1&2)"""
        if self.rank == 0:
            self.logger.info("Initializing data loading system...")
        
        # Create memory manager (Feature 2)
        # memoryvaluecleanupdata
        self.memory_manager = create_memory_manager({
            'monitoring_interval': 5.0,
            'critical_threshold': 0.95,
            'warning_threshold': 0.90
        })
        
        # Create streaming data loader (Feature 2)
        max_workers = self.config.get('num_workers', 8)
        if max_workers <= 0:
            max_workers = 1  # Ensure at least 1 worker
        
        # 使用优化的多worker数据加载
        self.streaming_loader = OptimizedStreamingDataLoader(
            data_dir=self.config.get('data_dir', '/nas/feature_v2_10s'),
            memory_manager=self.memory_manager,
            max_workers=max_workers,  # 使用配置的worker数量
            enable_async_loading=True  # 启用异步加载提高效率
        )
        
        # Define columns
        factor_columns = [str(i) for i in range(self.config.get('input_dim', 100))]
        target_columns = self.config.get('target_columns', ['intra30m', 'nextT1d', 'ema1d'])
        
        # Get data files with strict time ordering (Feature 1)
        data_dir = Path(self.config.get('data_dir', '/nas/feature_v2_10s'))
        all_files = sorted(list(data_dir.glob("*.parquet")))
        
        # Check for yearly rolling mode (Feature 7)
        # Auto-enable yearly rolling if years are sufficient and config allows
        enable_yearly_rolling = bool(self.config.get('enable_yearly_rolling', False))
        
        if enable_yearly_rolling:
            self.logger.info("Annual rolling training enabled")
            return self._setup_yearly_rolling_data(all_files, factor_columns, target_columns)
        else:
            self.logger.info("Standard training mode (yearly rolling disabled)")
            return self._setup_standard_data(all_files, factor_columns, target_columns)

    def _setup_standard_data(self, all_files: List[Path], factor_columns: List[str], target_columns: List[str],
                             overrides: Optional[Dict[str, str]] = None):
        """Setup data for standard training mode"""
        #  all_files  streaming_loader  data_dir file
        if not all_files:
            try:
                files_attr = getattr(self.streaming_loader, 'data_files', [])
                all_files = list(files_attr) if files_attr else []
            except Exception:
                all_files = []
            if not all_files:
                data_dir = Path(self.config.get('data_dir', '/nas/feature_v2_10s'))
                all_files = sorted(list(data_dir.glob('*.parquet')))
        # Parse date range from config
        train_start = (overrides or {}).get('train_start_date', self.config.get('train_start_date', '2018-01-02'))
        train_end = (overrides or {}).get('train_end_date', self.config.get('train_end_date', '2018-10-31'))
        val_start = (overrides or {}).get('val_start_date', self.config.get('val_start_date', '2018-11-01'))
        val_end = (overrides or {}).get('val_end_date', self.config.get('val_end_date', '2018-12-31'))
        test_start = (overrides or {}).get('test_start_date', self.config.get('test_start_date', '2019-01-01'))
        test_end = (overrides or {}).get('test_end_date', self.config.get('test_end_date', '2019-03-27'))

        # Runtime validation to ensure splits are disjoint and ordered (Feature 1)
        try:
            ts = datetime.strptime(test_start, '%Y-%m-%d')
            te = datetime.strptime(test_end, '%Y-%m-%d')
            vs = datetime.strptime(val_start, '%Y-%m-%d')
            ve = datetime.strptime(val_end, '%Y-%m-%d')
            tr_s = datetime.strptime(train_start, '%Y-%m-%d')
            tr_e = datetime.strptime(train_end, '%Y-%m-%d')
            # If validation overlaps or equals test range, auto-correct to previous year Nov-Dec
            overlap = not (ve < ts or vs > te)
            if overlap:
                prev_year = ts.year - 1
                new_vs = datetime.strptime(f"{prev_year}-11-01", '%Y-%m-%d')
                new_ve = datetime.strptime(f"{prev_year}-12-31", '%Y-%m-%d')
                if self.rank == 0:
                    self.logger.warning(
                        f"Validation window overlaps test window (val {val_start}~{val_end}, test {test_start}~{test_end}). "
                        f"Auto-correct to {new_vs.date()}~{new_ve.date()}"
                    )
                val_start = new_vs.strftime('%Y-%m-%d')
                val_end = new_ve.strftime('%Y-%m-%d')
            # Ensure train and val disjoint: train_end < val_start
            if tr_e >= vs:
                # shift val_start to train_end + 1 day
                corrected_vs = (tr_e + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                if self.rank == 0:
                    self.logger.warning(
                        f"Train/Val overlap detected (train_end {train_end} >= val_start {val_start}). "
                        f"Auto-correct val_start -> {corrected_vs}"
                    )
                val_start = corrected_vs

            # Enforce year boundary: predict next year (no overlapping year between val and test)
            if self.config.get('enforce_next_year_prediction', True):
                # Ensure test starts on Jan 1 of the year after validation end
                next_year = ve.year + 1
                enforced_test_start = datetime.strptime(f"{next_year}-01-01", '%Y-%m-%d')
                if ts < enforced_test_start:
                    if self.rank == 0:
                        self.logger.warning(
                            f"Adjusting test_start from {test_start} to {enforced_test_start.date()} to enforce next-year prediction"
                        )
                    ts = enforced_test_start
                    test_start = ts.strftime('%Y-%m-%d')

                # Cap test_end to actual data end or Dec 31 next_year
                actual = self._get_actual_date_range()
                cap_date_str = f"{next_year}-12-31"
                try:
                    cap_date = datetime.strptime(cap_date_str, '%Y-%m-%d')
                except Exception:
                    cap_date = te
                if actual and actual.get('end'):
                    try:
                        actual_end = datetime.strptime(actual['end'], '%Y-%m-%d')
                        cap_date = min(cap_date, actual_end)
                    except Exception:
                        pass
                if te > cap_date:
                    if self.rank == 0:
                        self.logger.info(
                            f"Capping test_end from {test_end} to {cap_date.date()} based on data availability/year boundary"
                        )
                    te = cap_date
                    test_end = te.strftime('%Y-%m-%d')
        except Exception as e:
            if self.rank == 0:
                self.logger.warning(f"Date validation failed: {e}, using original dates")
            # If parsing fails, keep original and proceed
            pass
        
        # Filter files by date ranges (Feature 1: strict data leakage prevention)
        train_files = []
        val_files = []
        test_files = []
        
        for file_path in all_files:
            # Extract date from filename (assuming format like YYYYMMDD.parquet)
            try:
                date_str = file_path.stem
                file_date = datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d')

                # Ensure all dates are strings for comparison
                train_start_str = str(train_start)
                train_end_str = str(train_end)
                val_start_str = str(val_start)
                val_end_str = str(val_end)
                test_start_str = str(test_start)
                test_end_str = str(test_end)

                if train_start_str <= file_date <= train_end_str:
                    train_files.append(file_path)
                elif val_start_str <= file_date <= val_end_str:
                    val_files.append(file_path)
                elif test_start_str <= file_date <= test_end_str:
                    test_files.append(file_path)
            except (ValueError, TypeError) as e:
                if self.rank == 0:
                    self.logger.warning(f"Date parsing error for {file_path}: {e}")
                continue
        
        if self.rank == 0:
            self.logger.info(
                f"Planned windows -> Train: {train_start}~{train_end}, Val: {val_start}~{val_end}, Test: {test_start}~{test_end}"
            )
            self.logger.info(f"Data split: Train({len(train_files)}) Val({len(val_files)}) Test({len(test_files)})")
            if len(train_files) == 0:
                self.logger.warning("No training files found! Check date ranges and file naming.")
        
        # Create datasets with date filtering
        sequence_length = self.config.get('sequence_length', 20)
        
        train_dataset = OptimizedStreamingDataset(
            self.streaming_loader, factor_columns, target_columns,
            sequence_length=sequence_length,
            start_date=train_start,
            end_date=train_end,
            enable_sequence_shuffle=True,
            shuffle_buffer_size=int(self.config.get('shuffle_buffer_size', 256))
        )
        
        val_dataset = OptimizedStreamingDataset(
            self.streaming_loader, factor_columns, target_columns,
            sequence_length=sequence_length,
            start_date=val_start,
            end_date=val_end,
            enable_sequence_shuffle=False
        )
        
        test_dataset = OptimizedStreamingDataset(
            self.streaming_loader, factor_columns, target_columns,
            sequence_length=sequence_length,
            start_date=test_start,
            end_date=test_end,
            enable_sequence_shuffle=False
        )
        
        # Create DataLoaders
        # usagefixedbatchstarthardwaredata fixed_batch_sizeDivide evenly by world_size
        fixed_bs = self.config.get('fixed_batch_size') or self.config.get('batch_size', 256)
        # disablebatch
        self.config['use_adaptive_batch_size'] = False
        self.config['adaptive_batch_size'] = False
        per_rank_bs = max(1, int(fixed_bs) // max(self.world_size, 1))
        
        # 计算每个DataLoader的worker数量
        dataloader_workers = max(1, self.config.get('num_workers', 16) // 4)  # 为每个DataLoader分配workers
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=per_rank_bs,
            shuffle=False,  # Streaming dataset handles shuffle internally
            num_workers=dataloader_workers,  # 使用配置的worker数量
            pin_memory=self.config.get('pin_memory', True),
            prefetch_factor=self.config.get('prefetch_factor', 4),
            persistent_workers=True  # 保持worker进程避免重复创建
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=per_rank_bs,
            shuffle=False,
            num_workers=max(1, dataloader_workers // 2),  # 验证用较少workers
            pin_memory=self.config.get('pin_memory', True),
            prefetch_factor=self.config.get('prefetch_factor', 4),
            persistent_workers=True
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=per_rank_bs,
            shuffle=False,
            num_workers=max(1, dataloader_workers // 2),  # 测试用较少workers
            pin_memory=self.config.get('pin_memory', True),
            prefetch_factor=self.config.get('prefetch_factor', 4),
            persistent_workers=True
        )
        
        # Note: OptimizedStreamingDataLoader automatically loads files from directory
        # File filtering is handled by the OptimizedStreamingDataset based on date ranges
        
        if self.rank == 0:
            self.logger.info("Data loading system initialized successfully")

    def _get_available_years(self) -> List[int]:
        """Extract available years from data files"""
        years = set()
        try:
            for fp in getattr(self.streaming_loader, 'data_files', []):
                stem = Path(fp).stem
                if len(stem) == 8 and stem.isdigit():
                    years.add(int(stem[:4]))
                elif len(stem) >= 10 and stem[4] == '-' and stem[7] == '-':
                    years.add(int(stem[:4]))
        except Exception:
            pass
        return sorted(list(years))

    def _get_actual_date_range(self) -> Optional[Dict[str, str]]:
        """Get actual date range from data files"""
        try:
            dates = []
            for fp in getattr(self.streaming_loader, 'data_files', []):
                stem = Path(fp).stem
                if len(stem) == 8 and stem.isdigit():
                    # Convert YYYYMMDD to YYYY-MM-DD
                    date_str = f"{stem[:4]}-{stem[4:6]}-{stem[6:]}"
                    dates.append(date_str)
                elif len(stem) >= 10 and stem[4] == '-' and stem[7] == '-':
                    dates.append(stem)

            if dates:
                dates.sort()
                if self.rank == 0:
                    self.logger.info(f"Found {len(dates)} data files, date range: {dates[0]} to {dates[-1]}")
                return {
                    'start': dates[0],
                    'end': dates[-1]
                }
        except Exception as e:
            if self.rank == 0:
                self.logger.error(f"Error getting date range: {e}")
        return None

    def _setup_yearly_rolling_data(self, all_files: List[Path], factor_columns: List[str], target_columns: List[str]):
        """Setup initial state for yearly rolling; actual loop in train()"""
        if self.rank == 0:
            self.logger.info("Setting up yearly rolling training data...")
        # We only need loader and file inventory; per-year splits are done dynamically
        return self._setup_standard_data(all_files, factor_columns, target_columns)

    def create_model(self):
        """Create TCN+Attention model (Feature 5)"""
        if self.rank == 0:
            self.logger.info("Creating TCN+Attention model...")
        
        # Prepare model configuration
        model_config = {
            'model_type': self.config.get('model_type', 'AdvancedFactorForecastingTCNAttention'),
            'input_dim': self.config.get('input_dim', 100),
            'hidden_dim': self.config.get('hidden_dim', 512),
            'num_layers': self.config.get('num_layers', 8),
            'num_heads': self.config.get('num_heads', 8),
            'dropout_rate': self.config.get('dropout_rate', 0.2),
            'attention_dropout': self.config.get('attention_dropout', 0.15),
            'sequence_length': self.config.get('sequence_length', 20),
            'target_columns': self.config.get('target_columns', ['intra30m', 'nextT1d', 'ema1d']),
            'tcn_kernel_size': self.config.get('tcn_kernel_size', 3),
            'tcn_dilation_factor': self.config.get('tcn_dilation_factor', 2)
        }
        
        # Ensure bounded vocabulary size for stock embeddings
        model_config['num_stocks'] = int(os.environ.get('NUM_STOCKS', self.config.get('num_stocks', 100000)))
        self.model = create_advanced_model(model_config)
        self.model = self.model.to(self.device)
        
        # Distributed wrapper (Feature 3)
        if self.world_size > 1:
            self.model = DDP(
                self.model, 
                device_ids=[self.rank],
                find_unused_parameters=True,  # usageparametergradientsynchronizationerror
                broadcast_buffers=True,  # buffersynchronization
                gradient_as_bucket_view=True  # optimizememoryusage
            )
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 0.001),
            weight_decay=self.config.get('weight_decay', 1e-5)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=20,
            T_mult=2,
            eta_min=self.config.get('learning_rate', 0.001) * 0.01
        )
        
        # Mixed precision training (Feature 3)
        if self.config.get('use_mixed_precision', True):
            # Use new torch.amp API to avoid deprecation warnings
            try:
                self.scaler = torch.amp.GradScaler('cuda')
            except Exception:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Multi-objective loss function (Feature 8)
        try:
            # Try with target_columns parameter first
            self.loss_function = QuantitativeCorrelationLoss(
                target_columns=self.config.get('target_columns', ['intra30m', 'nextT1d', 'ema1d']),
                correlation_weight=0.5,
                mse_weight=0.5
            )
        except TypeError:
            # Fallback to basic initialization if target_columns not supported
            self.loss_function = QuantitativeCorrelationLoss()
        
        if self.rank == 0:
            if hasattr(self.model, 'module'):
                model_info = self.model.module.get_model_info()
            else:
                model_info = self.model.get_model_info()
            
            self.logger.info(f"Model type: {model_info['model_type']}")
            self.logger.info(f"Total parameters: {model_info['total_parameters']:,}")
            self.logger.info(f"Trainable parameters: {model_info['trainable_parameters']:,}")

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load checkpoint (Feature 4)"""
        if not os.path.exists(checkpoint_path):
            return False
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            if hasattr(self.model, 'module'):
                self.model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'scheduler_state_dict' in checkpoint and self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if 'scaler_state_dict' in checkpoint and self.scaler:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            self.start_epoch = checkpoint.get('epoch', 0) + 1
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.best_ic = checkpoint.get('best_ic', -float('inf'))
            
            if self.rank == 0:
                self.logger.info(f"Successfully loaded checkpoint: {checkpoint_path}")
                self.logger.info(f"Resuming from epoch {self.start_epoch}")
            
            return True
            
        except Exception as e:
            if self.rank == 0:
                self.logger.error(f"Failed to load checkpoint: {e}")
            return False

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save checkpoint (Feature 4)"""
        if self.rank != 0:  # Only rank 0 saves
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_val_loss': self.best_val_loss,
            'best_ic': self.best_ic,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model: {best_path}")
        
        # Clean old checkpoints if save_all_checkpoints is False
        if not self.config.get('save_all_checkpoints', True):
            checkpoints = list(self.output_dir.glob("checkpoint_epoch_*.pth"))
            if len(checkpoints) > 3:  # Keep only last 3 checkpoints
                old_checkpoints = sorted(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))[:-3]
                for old_cp in old_checkpoints:
                    old_cp.unlink()
        
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")

    def train_epoch(self, epoch: int) -> float:
        """Train one epoch"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} Training") if self.rank == 0 else self.train_loader
        
        for batch_idx, batch in enumerate(pbar):
            if training_should_stop:
                break
            
            try:
                # Move data to device
                features = batch['features'].to(self.device, non_blocking=True)
                # Normalize targets: expect batch['targets'] as (B, T_targets) with T_targets=len(target_columns)
                raw_targets = batch['targets']
                if raw_targets.dim() == 1:
                    raw_targets = raw_targets.unsqueeze(0)
                if raw_targets.dim() == 3:
                    # If mistakenly (B, S, T), collapse time by taking last step
                    raw_targets = raw_targets[:, -1, :]
                assert raw_targets.dim() == 2, "targets tensor must be (B, T_targets)"
                target_cols = self.config.get('target_columns', ['intra30m', 'nextT1d', 'ema1d'])
                assert raw_targets.size(1) == len(target_cols), "targets second dim must equal number of target_columns"
                targets = {col: raw_targets[:, i].to(self.device, non_blocking=True)
                          for i, col in enumerate(target_cols)}
                stock_ids = batch['stock_id']
                # Debug stock_ids shape handling
                if self.rank == 0 and batch_idx == 0:
                    self.logger.info(f"Original stock_ids shape: {stock_ids.shape}")
                    self.logger.info(f"Features shape: {features.shape}")
                
                # Ensure stock_ids matches batch and sequence dimensions
                if stock_ids.dim() == 1:
                    stock_ids = stock_ids.unsqueeze(1)
                # Expand to match sequence length only if needed
                if stock_ids.size(1) == 1 and features.size(1) > 1:
                    stock_ids = stock_ids.expand(-1, features.size(1))
                stock_ids = stock_ids.to(self.device, non_blocking=True)
                
                self.optimizer.zero_grad()
                
                if self.scaler:
                    # Mixed precision training (Feature 3)
                    with torch.amp.autocast('cuda'):
                        predictions = self.model(features, stock_ids)
                        # securityprocesspredicttensordimensionusagetimesteps (B, seq)->(B,)
                        pred_last = {}
                        for col in predictions:
                            pred = predictions[col]
                            if pred.dim() >= 2:
                                pred_last[col] = pred[:, -1]
                            else:
                                pred_last[col] = pred
                        loss = self.loss_function(pred_last, targets)
                        if isinstance(loss, dict):
                            loss = loss.get('total_loss', None) or next((v for k, v in loss.items() if isinstance(v, torch.Tensor)), None)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                 max_norm=self.config.get('gradient_clip_norm', 1.0))
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Standard training
                    predictions = self.model(features, stock_ids)
                    # securityprocesspredicttensordimension
                    pred_last = {}
                    for col in predictions:
                        pred = predictions[col]
                        if pred.dim() >= 2:
                            pred_last[col] = pred[:, -1]
                        else:
                            pred_last[col] = pred
                    loss = self.loss_function(pred_last, targets)
                    if isinstance(loss, dict):
                        loss = loss.get('total_loss', None) or next((v for k, v in loss.items() if isinstance(v, torch.Tensor)), None)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                 max_norm=self.config.get('gradient_clip_norm', 1.0))
                    self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                self.global_step += 1
                
                if self.rank == 0 and isinstance(pbar, tqdm):
                    pbar.set_postfix({
                        'Loss': f"{loss.item():.6f}",
                        'Avg': f"{epoch_loss/num_batches:.6f}"
                    })
                
                # Report IC periodically (Feature 6)
                if self.ic_reporter and self.global_step % 100 == 0:
                    # Convert predictions and targets to numpy arrays for IC reporter
                    pred_dict = {col: pred.detach().cpu().numpy() for col, pred in pred_last.items()}
                    target_dict = {col: target.detach().cpu().numpy() for col, target in targets.items()}
                    self.ic_reporter.add_in_sample_data(pred_dict, target_dict)
                
            except Exception as e:
                if self.rank == 0:
                    import traceback
                    self.logger.error(f"Training batch error: {e}")
                    self.logger.error(f"Features shape: {features.shape}")
                    self.logger.error(f"Raw targets shape: {batch['targets'].shape}")  
                    self.logger.error(f"Stock IDs shape: {batch['stock_id'].shape}")
                    self.logger.error(f"Full traceback:\n{traceback.format_exc()}")
                
                # Clear any remaining gradients and skip this batch
                self.optimizer.zero_grad()
                if self.world_size > 1:
                    # Sync all processes on error to prevent DDP hangs
                    dist.barrier()
                continue
        
        return epoch_loss / max(num_batches, 1)

    def validate_epoch(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        """Validate one epoch with IC calculation (Feature 6)"""
        self.model.eval()
        epoch_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} Validation") if self.rank == 0 else self.val_loader
        
        with torch.no_grad():
            for batch in pbar:
                try:
                    features = batch['features'].to(self.device, non_blocking=True)
                    # Normalize targets: expect batch['targets'] as (B, T_targets) with T_targets=len(target_columns)
                    raw_targets = batch['targets']
                    if raw_targets.dim() == 1:
                        raw_targets = raw_targets.unsqueeze(0)
                    if raw_targets.dim() == 3:
                        # If mistakenly (B, S, T), collapse time by taking last step
                        raw_targets = raw_targets[:, -1, :]
                    assert raw_targets.dim() == 2, f"targets tensor must be (B, T_targets), got shape {raw_targets.shape}"
                    target_cols = self.config.get('target_columns', ['intra30m', 'nextT1d', 'ema1d'])
                    assert raw_targets.size(1) == len(target_cols), f"targets second dim must equal number of target_columns, got {raw_targets.size(1)} vs {len(target_cols)}"
                    targets = {col: raw_targets[:, i].to(self.device, non_blocking=True)
                              for i, col in enumerate(target_cols)}
                    # stock_idsdimensionprocess - train_epoch
                    stock_ids = batch['stock_id']
                    if stock_ids.dim() == 1:
                        stock_ids = stock_ids.unsqueeze(1)
                    if stock_ids.size(1) == 1 and features.size(1) > 1:
                        stock_ids = stock_ids.expand(-1, features.size(1))
                    stock_ids = stock_ids.to(self.device, non_blocking=True)
                    
                    if self.scaler:
                        with torch.amp.autocast('cuda'):
                            predictions = self.model(features, stock_ids)
                            # securityprocesspredicttensordimension
                            pred_last = {}
                            for col in predictions:
                                pred = predictions[col]
                                if pred.dim() >= 2:
                                    pred_last[col] = pred[:, -1]
                                else:
                                    pred_last[col] = pred
                            loss = self.loss_function(pred_last, targets)
                            if isinstance(loss, dict):
                                loss = loss.get('total_loss', None) or next((v for k, v in loss.items() if isinstance(v, torch.Tensor)), None)
                    else:
                        predictions = self.model(features, stock_ids)
                        # securityprocesspredicttensordimension
                        pred_last = {}
                        for col in predictions:
                            pred = predictions[col]
                            if pred.dim() >= 2:
                                pred_last[col] = pred[:, -1]
                            else:
                                pred_last[col] = pred
                        loss = self.loss_function(pred_last, targets)
                        if isinstance(loss, dict):
                            loss = loss.get('total_loss', None) or next((v for k, v in loss.items() if isinstance(v, torch.Tensor)), None)
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                    
                    # Collect predictions for IC calculation
                    if num_batches <= 10:  # Limit to prevent memory issues
                        for col in self.config.get('target_columns', ['intra30m', 'nextT1d', 'ema1d']):
                            if col in predictions:
                                all_predictions.append(predictions[col].cpu().numpy())
                                all_targets.append(targets[col].cpu().numpy())
                                break  # Just use first available target
                    
                    if self.rank == 0 and isinstance(pbar, tqdm):
                        pbar.set_postfix({
                            'Loss': f"{loss.item():.6f}",
                            'Avg': f"{epoch_loss/num_batches:.6f}"
                        })
                        
                except Exception as e:
                    if self.rank == 0:
                        self.logger.error(f"Validation batch error: {e}")
                    continue
        
        avg_loss = epoch_loss / max(num_batches, 1)
        
        # Calculate IC metrics (Feature 6)
        ic_metrics = {}
        if all_predictions and all_targets and self.rank == 0:
            try:
                pred_array = np.concatenate(all_predictions, axis=0)
                target_array = np.concatenate(all_targets, axis=0)
                
                # Flatten arrays for correlation calculation
                pred_flat = pred_array.flatten()
                target_flat = target_array.flatten()
                
                # Remove NaN values
                mask = ~(np.isnan(pred_flat) | np.isnan(target_flat))
                if mask.sum() > 100:  # At least 100 valid samples
                    pred_clean = pred_flat[mask]
                    target_clean = target_flat[mask]
                    
                    # Calculate IC metrics
                    ic_metrics = {
                        'ic': np.corrcoef(pred_clean, target_clean)[0, 1],
                        'rank_ic': np.corrcoef(np.argsort(pred_clean), np.argsort(target_clean))[0, 1],
                        'mse': np.mean((pred_clean - target_clean) ** 2),
                        'mae': np.mean(np.abs(pred_clean - target_clean))
                    }
                    
                    # Report validation metrics to IC reporter
                    if self.ic_reporter:
                        # buildverificationpredicttargetdictionary
                        val_pred_dict = {}
                        val_target_dict = {}
                        
                        # targetdata
                        target_cols = self.config.get('target_columns', ['intra30m', 'nextT1d', 'ema1d'])
                        
                        if len(target_cols) <= pred_array.shape[1]:
                            for i, col in enumerate(target_cols):
                                if i < pred_array.shape[1]:
                                    val_pred_dict[col] = pred_array[:, i]
                                    val_target_dict[col] = target_array[:, i]
                        
                        # addverificationdataout-of-sample
                        if val_pred_dict and val_target_dict:
                            self.ic_reporter.add_out_sample_data(val_pred_dict, val_target_dict)
                    
            except Exception as e:
                self.logger.error(f"IC calculation error: {e}")
        
        return avg_loss, ic_metrics

    def evaluate_test(self) -> Dict[str, float]:
        """Evaluate on test set with IC metrics (out-of-sample)"""
        if self.test_loader is None:
            return {}
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                try:
                    features = batch['features'].to(self.device, non_blocking=True)
                    targets = {col: batch['targets'][:, :, i].to(self.device, non_blocking=True) 
                              for i, col in enumerate(self.config.get('target_columns', ['intra30m', 'nextT1d', 'ema1d']))}
                    # stock_idsdimensionprocess
                    stock_ids = batch['stock_id']
                    if stock_ids.dim() == 1:
                        stock_ids = stock_ids.unsqueeze(1)
                    if stock_ids.size(1) == 1 and features.size(1) > 1:
                        stock_ids = stock_ids.expand(-1, features.size(1))
                    stock_ids = stock_ids.to(self.device, non_blocking=True)
                    
                    predictions = self.model(features, stock_ids)
                    
                    # recordtargetIC
                    for col in self.config.get('target_columns', ['intra30m', 'nextT1d', 'ema1d']):
                        if col in predictions:
                            all_predictions.append(predictions[col].cpu().numpy())
                            all_targets.append(targets[col].cpu().numpy())
                            break
                except Exception:
                    continue
        
        ic_metrics = {}
        if all_predictions and all_targets and self.rank == 0:
            try:
                pred_array = np.concatenate(all_predictions, axis=0)
                target_array = np.concatenate(all_targets, axis=0)
                
                pred_flat = pred_array.flatten()
                target_flat = target_array.flatten()
                mask = ~(np.isnan(pred_flat) | np.isnan(target_flat))
                if mask.sum() > 100:
                    pred_clean = pred_flat[mask]
                    target_clean = target_flat[mask]
                    ic_metrics = {
                        'test_ic': np.corrcoef(pred_clean, target_clean)[0, 1],
                        'test_rank_ic': np.corrcoef(np.argsort(pred_clean), np.argsort(target_clean))[0, 1]
                    }
            except Exception:
                pass
        return ic_metrics

    def train(self):
        """Main training loop with all 8 features integrated"""
        if self.rank == 0:
            self.logger.info("Starting unified complete training...")
            self.logger.info(f"Training configuration: {json.dumps(self.config, indent=2, default=str)}")
        
        # Find latest checkpoint for auto-resume (Feature 4)
        if self.config.get('auto_resume', True):
            checkpoints = list(self.output_dir.glob("checkpoint_epoch_*.pth"))
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
                self.load_checkpoint(str(latest_checkpoint))
        
        patience_counter = 0
        max_patience = self.config.get('early_stopping_patience', 30)
        
        try:
            # If yearly rolling enabled and data sufficient, run per-year loop
            if self.config.get('enable_yearly_rolling', False):
                available_years = self._get_available_years()
                min_train_years = self.config.get('min_train_years', 1)
                if len(available_years) >= min_train_years + 1:
                    # Define prediction years list
                    prediction_years = self.config.get('prediction_years')
                    if not prediction_years:
                        prediction_years = available_years[min_train_years:]
                    
                    for y in prediction_years:
                        # Determine train/val/test windows
                        prev_year = y - 1

                        # getdatafile
                        actual_dates = self._get_actual_date_range()
                        if actual_dates:
                            actual_start = actual_dates['start']
                            # trainingdatabeginyears10months
                            train_start = actual_start
                            train_end = f"{prev_year}-10-31"
                        else:
                            # 
                            train_start = f"{available_years[0]}-01-01"
                            train_end = f"{prev_year}-10-31"
                        # verificationyears11-12monthstraining
                        val_start = f"{prev_year}-11-01"
                        val_end = f"{prev_year}-12-31"
                        test_start = f"{y}-01-01"
                        test_end = f"{y}-12-31"
                        
                        if self.rank == 0:
                            self.logger.info(f"Rolling year plan -> Train: {train_start}~{train_end}, Val: {val_start}~{val_end}, Test: {test_start}~{test_end}")
                        
                        # Rebuild loaders with per-year overrides all_files 
                        self._setup_standard_data(
                            all_files=[],
                            factor_columns=[str(i) for i in range(self.config.get('input_dim', 100))],
                            target_columns=self.config.get('target_columns', ['intra30m','nextT1d','ema1d']),
                            overrides={
                                'train_start_date': train_start,
                                'train_end_date': train_end,
                                'val_start_date': val_start,
                                'val_end_date': val_end,
                                'test_start_date': test_start,
                                'test_end_date': test_end,
                            }
                        )
                        
                        # Reset state for each year
                        self.start_epoch = 0
                        patience_counter = 0
                        self.best_val_loss = float('inf')
                        
                        local_epochs = max(1, min(self.config.get('epochs', 50), 10))
                        for epoch in range(local_epochs):
                            if training_should_stop:
                                break
                            train_loss = self.train_epoch(epoch)
                            self.train_losses.append(train_loss)
                            val_loss, ic_metrics = self.validate_epoch(epoch)
                            self.val_losses.append(val_loss)
                            if self.scheduler:
                                self.scheduler.step()
                            current_ic = ic_metrics.get('ic', -float('inf'))
                            self.ic_scores.append(current_ic)
                            is_best = val_loss < self.best_val_loss
                            if is_best:
                                self.best_val_loss = val_loss
                                patience_counter = 0
                            else:
                                patience_counter += 1
                            if self.rank == 0:
                                lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.config.get('learning_rate', 0.001)
                                self.logger.info(
                                    f"[Year {y}] Epoch {epoch}: Train {train_loss:.6f}, Val {val_loss:.6f}, IC {current_ic:.4f}, LR {lr:.2e}"
                                )
                            if self.rank == 0 and (epoch + 1) % self.config.get('checkpoint_frequency', 10) == 0:
                                self.save_checkpoint(epoch, is_best)
                                test_ic = self.evaluate_test()
                                if test_ic:
                                    self.logger.info(f"[Year {y}] Out-of-sample IC: {test_ic}")
                                    if self.ic_reporter:
                                        self.ic_reporter.log_ic_metrics(test_ic)
                            if patience_counter >= self.config.get('early_stopping_patience', 30):
                                if self.rank == 0:
                                    self.logger.info(f"[Year {y}] Early stopping")
                                break
                            if self.world_size > 1:
                                dist.barrier()
                    
                    # Done rolling loop
                    return
            
            # Fallback: standard single-window training
            for epoch in range(self.start_epoch, self.config.get('epochs', 50)):
                if training_should_stop:
                    if self.rank == 0:
                        self.logger.info("Stop signal received, saving model...")
                    break
                
                # Training
                train_loss = self.train_epoch(epoch)
                self.train_losses.append(train_loss)
                
                # Validation
                val_loss, ic_metrics = self.validate_epoch(epoch)
                self.val_losses.append(val_loss)
                
                # Update learning rate
                if self.scheduler:
                    self.scheduler.step()
                
                # Check for best model
                current_ic = ic_metrics.get('ic', -float('inf'))
                self.ic_scores.append(current_ic)
                
                is_best = False
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    patience_counter = 0
                    is_best = True
                else:
                    patience_counter += 1
                
                if current_ic > self.best_ic:
                    self.best_ic = current_ic
                
                # Log training progress
                if self.rank == 0:
                    lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.config.get('learning_rate', 0.001)
                    self.logger.info(
                        f"Epoch {epoch}: Train Loss: {train_loss:.6f}, "
                        f"Val Loss: {val_loss:.6f}, IC: {current_ic:.4f}, "
                        f"LR: {lr:.2e}"
                    )
                    
                    if ic_metrics:
                        for metric, value in ic_metrics.items():
                            self.logger.info(f"  {metric}: {value:.4f}")
                
                # Save checkpoint (Feature 4)
                if self.rank == 0 and (epoch + 1) % self.config.get('checkpoint_frequency', 10) == 0:
                    self.save_checkpoint(epoch, is_best)
                
                # Periodically evaluate out-of-sample IC (Feature 6)
                if self.rank == 0 and self.test_loader and (epoch + 1) % self.config.get('checkpoint_frequency', 10) == 0:
                    test_ic = self.evaluate_test()
                    if test_ic:
                        self.logger.info(f"Out-of-sample IC: {test_ic}")
                        if self.ic_reporter:
                            self.ic_reporter.log_ic_metrics(test_ic)
                
                # Early stopping
                if patience_counter >= max_patience:
                    if self.rank == 0:
                        self.logger.info(f"Early stopping triggered - {patience_counter} epochs without improvement")
                    break
                
                # Synchronize processes
                if self.world_size > 1:
                    dist.barrier()
        
        except Exception as e:
            if self.rank == 0:
                self.logger.error(f"Training error: {e}")
            raise
        
        finally:
            # Save final results
            if self.rank == 0:
                self.save_final_results()
            
            # Cleanup
            self.cleanup()

    def save_final_results(self):
        """Save final training results"""
        results = {
            'training_config': self.config,
            'training_results': {
                'epochs_trained': len(self.train_losses),
                'best_val_loss': self.best_val_loss,
                'best_ic': self.best_ic,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'ic_scores': self.ic_scores
            },
            'final_stats': {
                'final_train_loss': self.train_losses[-1] if self.train_losses else None,
                'final_val_loss': self.val_losses[-1] if self.val_losses else None,
                'final_ic': self.ic_scores[-1] if self.ic_scores else None
            }
        }
        
        results_file = self.output_dir / "training_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info("="*80)
        self.logger.info("Unified complete training finished!")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
        self.logger.info(f"Best IC: {self.best_ic:.4f}")
        self.logger.info(f"Epochs trained: {len(self.train_losses)}")
        self.logger.info(f"Results saved to: {self.output_dir}")
        self.logger.info("="*80)

    def cleanup(self):
        """Cleanup resources"""
        if self.streaming_loader:
            self.streaming_loader.cleanup()
        
        if self.ic_reporter:
            self.ic_reporter.cleanup()
        
        if self.world_size > 1:
            dist.destroy_process_group()

def run_worker(rank: int, world_size: int, config: Dict[str, Any]):
    """Distributed training worker process - only used with mp.spawn"""
    # Set environment variables only if not already set by torchrun
    if 'RANK' not in os.environ:
    os.environ['RANK'] = str(rank)
    if 'WORLD_SIZE' not in os.environ:
    os.environ['WORLD_SIZE'] = str(world_size)
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(rank)
        
    # Set master address/port (respect pre-set values)
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', os.environ.get('MASTER_PORT', '12355'))
    
    try:
        trainer = UnifiedCompleteTrainer(config, rank, world_size)
        trainer.setup_distributed()
        trainer.setup_data_loaders()
        trainer.create_model()
        trainer.train()
        
    except Exception as e:
        print(f"Worker {rank} error: {e}")
        raise

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file with proper validation"""
    # Define defaults first to avoid UnboundLocalError
        defaults = {
            'model_type': 'AdvancedFactorForecastingTCNAttention',
            'input_dim': 100,
            'hidden_dim': 512,
            'num_layers': 8,
            'num_heads': 8,
            'dropout_rate': 0.2,
            'attention_dropout': 0.15,
            'sequence_length': 20,
            'epochs': 50,
            'batch_size': 256,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'gradient_clip_norm': 1.0,
            'use_mixed_precision': True,
            'enable_yearly_rolling': False,  # Disabled by default for limited data
            'min_train_years': 1,
            'rolling_window_years': 1,
            'num_workers': 8,
            'pin_memory': True,
            'use_distributed': True,
            'output_dir': '/nas/factor_forecasting/outputs',
            'log_level': 'INFO',
            'checkpoint_frequency': 10,
            'save_all_checkpoints': False,
            'auto_resume': True,
            'ic_report_interval': 7200,
            'enable_ic_reporting': True,
        'target_columns': ['intra30m', 'nextT1d', 'ema1d'],
        'data_dir': '/nas/feature_v2_10s'
    }
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required fields
        required_fields = ['data_dir', 'target_columns']
        for field in required_fields:
            if field not in config:
                print(f"Warning: Required field '{field}' not found in config")
        
        # Merge with defaults
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
        
        print(f"Configuration loaded successfully from {config_path}")
        print(f"Key settings: enable_yearly_rolling={config['enable_yearly_rolling']}, "
              f"min_train_years={config['min_train_years']}")
        
        return config
        
    except Exception as e:
        print(f"Error loading configuration: {e}")
        print("Using default configuration...")
        return defaults

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Unified Complete Training System')
    parser.add_argument('--config', type=str, default='server_optimized_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--distributed', action='store_true',
                       help='Enable distributed training')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Check if running under torchrun (environment variables set by torchrun)
    is_torchrun = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ and 'LOCAL_RANK' in os.environ
    
    if is_torchrun:
        # Running under torchrun: use environment variables directly
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        print(f"Running under torchrun - Rank {rank}/{world_size}, Local rank: {local_rank}")
        
        # Direct worker execution (no mp.spawn)
        try:
            trainer = UnifiedCompleteTrainer(config, rank, world_size)
            trainer.setup_distributed()
            trainer.setup_data_loaders()
            trainer.create_model()
            
            if args.resume:
                trainer.load_checkpoint(args.resume)
                
            trainer.train()
        except Exception as e:
            print(f"Worker {rank} error: {e}")
            raise
    else:
        # Traditional launch: check GPU count and use mp.spawn if needed
    num_gpus = torch.cuda.device_count()
    use_distributed = config.get('use_distributed', False) and num_gpus > 1
    
    if use_distributed:
            # Set multiprocessing start method
            mp.set_start_method('spawn', force=True)
            
        # Choose a free port if not provided
        if 'MASTER_PORT' not in os.environ:
            def _find_free_port(default_port: int = 12355) -> str:
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.bind(('', 0))
                        return str(s.getsockname()[1])
                except Exception:
                    return str(default_port)
            os.environ['MASTER_PORT'] = _find_free_port()
        print(f"Starting distributed training - {num_gpus} GPUs")
        mp.spawn(run_worker, args=(num_gpus, config), nprocs=num_gpus, join=True)
    else:
        print("Starting single GPU training")
        trainer = UnifiedCompleteTrainer(config, 0, 1)
        trainer.setup_data_loaders()
        trainer.create_model()
        
        if args.resume:
            trainer.load_checkpoint(args.resume)
        
        trainer.train()

if __name__ == "__main__":
    main()
