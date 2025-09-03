#!/usr/bin/env python3
"""
创建一个完全干净的训练脚本
从根本上解决所有问题
"""

def create_clean_training_script():
    """创建一个完全干净、可工作的训练脚本"""
    
    script_content = '''#!/usr/bin/env python3
"""
Unified Complete Training System V2 - Clean Fixed Version
完全修复版本，解决所有CUDA多进程和语法问题
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

# Fix CUDA multiprocessing compatibility
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# Add project path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Import components
from src.models.advanced_tcn_attention import create_advanced_model
from src.data_processing.optimized_streaming_loader import OptimizedStreamingDataLoader, OptimizedStreamingDataset
from src.training.quantitative_loss import QuantitativeCorrelationLoss, AdaptiveQuantitativeLoss, create_quantitative_loss_function
from src.monitoring.ic_reporter import ICCorrelationReporter
from src.data_processing.adaptive_memory_manager import create_memory_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)


class UnifiedCompleteTrainer:
    """Unified Complete Training System with all 8 features"""
    
    def __init__(self, config: Dict[str, Any], rank: int = 0, world_size: int = 1):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        
        # Setup device
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{rank}')
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device('cpu')
        
        # Setup logger
        self.logger = logging.getLogger(f'Trainer_Rank_{rank}')
        
        # Initialize components
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scaler = None
        self.ic_reporter = None
        
        # Memory manager
        self.memory_manager = create_memory_manager()
        
        if self.rank == 0:
            self.logger.info(f"Trainer initialized - Rank {rank}/{world_size}, Device: {self.device}")

    def setup_data_loaders(self):
        """Setup data loaders with streaming and leakage prevention"""
        self.logger.info("Initializing data loading system...")
        
        # Create optimized streaming data loader
        self.streaming_loader = OptimizedStreamingDataLoader(
            data_dir=self.config.get('data_dir', '/nas/feature_v2_10s'),
            memory_manager=self.memory_manager,
            max_workers=1,  # Single worker to avoid CUDA issues
            enable_async_loading=False  # Disabled for CUDA compatibility
        )
        
        # Define columns
        factor_columns = [str(i) for i in range(self.config.get('input_dim', 100))]
        target_columns = self.config.get('target_columns', ['intra30m', 'nextT1d', 'ema1d'])
        
        # Get data files with strict time ordering
        data_dir = Path(self.config.get('data_dir', '/nas/feature_v2_10s'))
        all_files = sorted(list(data_dir.glob("*.parquet")))
        
        # Split data by date ranges
        train_start = self.config.get('train_start_date', '2018-01-02')
        train_end = self.config.get('train_end_date', '2018-10-31')
        val_start = self.config.get('val_start_date', '2018-11-01')
        val_end = self.config.get('val_end_date', '2018-12-31')
        test_start = self.config.get('test_start_date', '2019-01-01')
        test_end = self.config.get('test_end_date', '2019-12-31')
        
        # Create datasets
        train_dataset = OptimizedStreamingDataset(
            self.streaming_loader, factor_columns, target_columns,
            self.config.get('sequence_length', 60), train_start, train_end,
            enable_sequence_shuffle=True
        )
        
        val_dataset = OptimizedStreamingDataset(
            self.streaming_loader, factor_columns, target_columns,
            self.config.get('sequence_length', 60), val_start, val_end,
            enable_sequence_shuffle=False
        )
        
        test_dataset = OptimizedStreamingDataset(
            self.streaming_loader, factor_columns, target_columns,
            self.config.get('sequence_length', 60), test_start, test_end,
            enable_sequence_shuffle=False
        )
        
        # Calculate batch size per rank
        total_batch_size = self.config.get('batch_size', 4096)
        per_rank_bs = total_batch_size // self.world_size
        
        # Create DataLoaders with CUDA-safe parameters
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=per_rank_bs,
            num_workers=0,  # No multiprocessing
            pin_memory=torch.cuda.is_available(),
            persistent_workers=False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=per_rank_bs,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=False
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=per_rank_bs,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=False
        )
        
        if self.rank == 0:
            self.logger.info(f"Data loaders created with batch_size={per_rank_bs} per rank")

    def create_model(self):
        """Create model and related components"""
        self.logger.info("Creating model...")
        
        # Model configuration
        model_config = {
            'input_dim': self.config.get('input_dim', 100),
            'hidden_dim': self.config.get('hidden_dim', 768),
            'num_layers': self.config.get('num_layers', 12),
            'num_heads': self.config.get('num_heads', 16),
            'tcn_kernel_size': self.config.get('tcn_kernel_size', 7),
            'tcn_dilation_factor': self.config.get('tcn_dilation_factor', 2),
            'dropout_rate': self.config.get('dropout_rate', 0.15),
            'attention_dropout': self.config.get('attention_dropout', 0.1),
            'target_columns': self.config.get('target_columns', ['nextT1d']),
            'sequence_length': self.config.get('sequence_length', 60)
        }
        
        # Create model
        self.model = create_advanced_model(model_config)
        self.model.to(self.device)
        
        # Wrap with DDP if distributed
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.rank])
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 0.0001),
            weight_decay=self.config.get('weight_decay', 0.01)
        )
        
        # Mixed precision scaler
        if self.config.get('use_mixed_precision', True):
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Loss function
        self.criterion = create_quantitative_loss_function(self.config)
        self.criterion.to(self.device)
        
        # IC Reporter
        if self.config.get('enable_ic_reporting', True) and self.rank == 0:
            self.ic_reporter = ICCorrelationReporter(
                output_dir=self.config.get('output_dir', 'outputs'),
                target_columns=self.config.get('target_columns', ['intra30m', 'nextT1d', 'ema1d']),
                report_interval=self.config.get('ic_report_interval', 7200)
            )
        
        if self.rank == 0:
            total_params = sum(p.numel() for p in self.model.parameters())
            self.logger.info(f"Model created with {total_params:,} parameters")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch} Training', disable=self.rank != 0)
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            features = batch['features'].to(self.device, non_blocking=True)
            targets = {k: v.to(self.device, non_blocking=True) for k, v in batch['targets'].items()}
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                predictions = self.model(features)
                loss_dict = self.criterion(predictions, targets)
                loss = loss_dict['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress
            if self.rank == 0:
                progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_loss = total_loss / max(num_batches, 1)
        return {'train_loss': avg_loss}

    def validate(self) -> Dict[str, float]:
        """Validation"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                features = batch['features'].to(self.device, non_blocking=True)
                targets = {k: v.to(self.device, non_blocking=True) for k, v in batch['targets'].items()}
                
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    predictions = self.model(features)
                    loss_dict = self.criterion(predictions, targets)
                    loss = loss_dict['total_loss']
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        return {'val_loss': avg_loss}

    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        
        epochs = self.config.get('epochs', 100)
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            epoch_time = time.time() - epoch_start_time
            
            if self.rank == 0:
                self.logger.info(f"Epoch {epoch}: Train Loss={train_metrics['train_loss']:.6f}, "
                               f"Val Loss={val_metrics['val_loss']:.6f}, Time={epoch_time:.2f}s")
                
                # IC reporting
                if self.ic_reporter and self.ic_reporter.should_generate_report():
                    report = self.ic_reporter.generate_report()
                    self.logger.info(f"IC Report generated: {report}")
            
            # Save checkpoint
            if self.rank == 0 and (epoch + 1) % self.config.get('checkpoint_frequency', 10) == 0:
                self.save_checkpoint(epoch)

    def save_checkpoint(self, epoch: int):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict() if self.world_size == 1 else self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        
        checkpoint_dir = Path(self.config.get('output_dir', 'outputs')) / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration"""
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return {}
    
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            return yaml.safe_load(f)
        else:
            return json.load(f)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Unified Complete Training System V2')
    parser.add_argument('--config', type=str, default='optimized_server_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    print(f"Configuration loaded from {args.config}")
    print("Starting single GPU training")
    
    # Create and run trainer
    trainer = UnifiedCompleteTrainer(config, 0, 1)
    trainer.setup_data_loaders()
    trainer.create_model()
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    trainer.train()


if __name__ == "__main__":
    main()
'''
    
    # Write the clean script
    with open("/nas/factor_forecasting/unified_complete_training_v2_clean.py", 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print("创建了完全干净的训练脚本: unified_complete_training_v2_clean.py")

if __name__ == "__main__":
    create_clean_training_script()
