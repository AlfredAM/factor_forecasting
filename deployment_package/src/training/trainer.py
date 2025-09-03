"""
Enhanced trainer module for factor forecasting models
Includes DDP distributed training, gradient accumulation, mixed precision, and advanced optimizations
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
# Optional tensorboard import
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
import logging
import os
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
from scipy import stats as scipy_stats
# Only suppress specific warnings that are known to be harmless
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, message='.*NumPy.*')

# Standard attention mechanism (no flash_attn dependency)
def standard_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False):
    """Standard attention function as fallback"""
    attn_weights = torch.matmul(q, k.transpose(-2, -1))
    if softmax_scale is not None:
        attn_weights = attn_weights * softmax_scale
    attn_weights = torch.softmax(attn_weights, dim=-1)
    if dropout_p > 0:
        attn_weights = torch.dropout(attn_weights, p=dropout_p, train=True)
    return torch.matmul(attn_weights, v)

# Use standard attention
flash_attn_func = standard_attn_func
FLASH_ATTENTION_AVAILABLE = False


class CorrelationLoss(nn.Module):
    """Enhanced loss function that optimizes for correlation between predictions and targets"""
    
    def __init__(self, correlation_weight: float = 1.0, mse_weight: float = 0.1, 
                 rank_weight: float = 0.1, target_correlations: List[float] = None):
        super().__init__()
        self.correlation_weight = correlation_weight
        self.mse_weight = mse_weight
        self.rank_weight = rank_weight
        self.mse_loss = nn.MSELoss()
    
        # Target correlations for each target (intra30m, nextT1d, ema1d)
        self.target_correlations = target_correlations or [0.1, 0.05, 0.08]
    
    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute enhanced correlation loss for three targets
        Args:
            predictions: Dictionary with predictions for each target
            targets: Dictionary with targets for each target
        """
        total_loss = 0.0
        target_names = ['intra30m', 'nextT1d', 'ema1d']
        
        # MSE loss
        mse_loss = 0.0
        for target_name in target_names:
            if target_name in predictions and target_name in targets:
                mse_loss += self.mse_loss(predictions[target_name], targets[target_name])
        
        # Correlation loss with target-specific optimization
        correlation_loss = 0.0
        rank_loss = 0.0
        
        for i, target_name in enumerate(target_names):
            if target_name in predictions and target_name in targets:
                pred = predictions[target_name]
                target = targets[target_name]
                
                # Ensure pred and target have the same shape before creating mask
                if pred.shape != target.shape:
                    if pred.dim() == 0:
                        pred = pred.expand_as(target)
                    elif target.dim() == 0:
                        target = target.expand_as(pred)
                    elif pred.dim() < target.dim():
                        pred = pred.unsqueeze(-1)
                    elif target.dim() < pred.dim():
                        target = target.unsqueeze(-1)

                # Remove NaN values (guard inside the presence branch)
                mask = ~(torch.isnan(pred) | torch.isnan(target))
                
                if mask.sum() > 1:
                    pred_clean = pred[mask]
                    target_clean = target[mask]
                    
                    # Compute correlation
                    pred_mean = pred_clean.mean()
                    target_mean = target_clean.mean()
                    pred_std = pred_clean.std()
                    target_std = target_clean.std()
                    
                    if pred_std > 1e-8 and target_std > 1e-8:
                        correlation = ((pred_clean - pred_mean) * (target_clean - target_mean)).mean() / (pred_std * target_std)
                        
                        # Target-specific correlation optimization
                        target_corr = self.target_correlations[i] if i < len(self.target_correlations) else 0.05
                        
                        # Penalize deviation from target correlation
                        correlation_loss += torch.abs(correlation - target_corr)
                        
                        # Rank correlation loss (Spearman's rank correlation)
                        pred_ranks = torch.argsort(torch.argsort(pred_clean)).float()
                        target_ranks = torch.argsort(torch.argsort(target_clean)).float()
                        rank_correlation = ((pred_ranks - pred_ranks.mean()) * (target_ranks - target_ranks.mean())).mean() / (pred_ranks.std() * target_ranks.std())
                        rank_loss += -rank_correlation  # Negative because we want to maximize
        
        # Combine losses
        total_loss = (self.mse_weight * mse_loss + 
                     self.correlation_weight * correlation_loss + 
                     self.rank_weight * rank_loss)
        
        return total_loss

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Enhanced early stopping mechanism with checkpointing.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, 
                 restore_best_weights: bool = True, checkpoint_dir: str = "checkpoints"):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change in validation loss to be considered as improvement
            restore_best_weights: Whether to restore best weights when stopping
            checkpoint_dir: Directory to save checkpoints
        """
        self.patience = patience
        self.min_delta = min_delta
        # Use a flag name to avoid clashing with method name
        self.restore_best_weights_flag = restore_best_weights
        self.checkpoint_dir = checkpoint_dir
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        self.best_epoch = 0
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def __call__(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Model to save weights from
            epoch: Current epoch number
            
        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            if self.restore_best_weights_flag:
                self.best_weights = model.state_dict().copy()
            
            # Save best checkpoint
            self.save_checkpoint(model, epoch, val_loss, is_best=True)
        else:
            self.counter += 1
            
        return self.counter >= self.patience
    
    def save_checkpoint(self, model: nn.Module, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
            'best_loss': self.best_loss,
            'counter': self.counter
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved at epoch {epoch} with loss {val_loss:.6f}")
    
    def restore_best_weights(self, model: nn.Module):
        """Restore best weights to the model."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            logger.info(f"Restored best weights from epoch {self.best_epoch}")


class LearningRateScheduler:
    """
    Enhanced learning rate scheduler with warmup and multiple strategies.
    """
    
    def __init__(self, optimizer: optim.Optimizer, scheduler_type: str = 'cosine',
                 warmup_steps: int = 1000, total_steps: int = 100000,
                 initial_lr: float = 1e-4, min_lr: float = 1e-6,
                 warmup_strategy: str = 'linear'):
        """
        Initialize learning rate scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            scheduler_type: Type of scheduler ('cosine', 'step', 'exponential', 'plateau')
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
            initial_lr: Initial learning rate
            min_lr: Minimum learning rate
            warmup_strategy: Warmup strategy ('linear', 'exponential')
        """
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.warmup_strategy = warmup_strategy
        self.current_step = 0
        
        # Initialize schedulers
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps - warmup_steps, eta_min=min_lr)
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=(total_steps - warmup_steps) // 10, gamma=0.5)
        elif scheduler_type == 'exponential':
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=0.95)
        elif scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, min_lr=min_lr)
        else:
            self.scheduler = None
    
    def step(self, val_loss: float = None):
        """Update learning rate."""
        if self.current_step < self.warmup_steps:
            # Warmup phase
            if self.warmup_strategy == 'linear':
                lr = self.initial_lr * (self.current_step / self.warmup_steps)
            elif self.warmup_strategy == 'exponential':
                lr = self.initial_lr * (0.1 ** (1 - self.current_step / self.warmup_steps))
            else:
                lr = self.initial_lr * (self.current_step / self.warmup_steps)
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        elif self.scheduler is not None:
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                if val_loss is not None:
                    self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
        
        self.current_step += 1
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']


class MetricsTracker:
    """
    Enhanced metrics tracker with IC calculations and backtesting metrics.
    """
    
    def __init__(self, target_columns: List[str]):
        """
        Initialize metrics tracker.
        
        Args:
            target_columns: List of target column names
        """
        self.target_columns = target_columns
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = {col: [] for col in self.target_columns}
        self.targets = {col: [] for col in self.target_columns}
        self.dates = []
        self.stock_ids = []
        self.sequence_info = []
    
    def update(self, predictions: Dict[str, torch.Tensor], 
               targets: Dict[str, torch.Tensor], sequence_info: List[Dict] = None):
        """
        Update metrics with new batch.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            sequence_info: Additional sequence information
        """
        for col in self.target_columns:
            if col in predictions and col in targets:
                self.predictions[col].extend(predictions[col].detach().cpu().numpy())
                self.targets[col].extend(targets[col].detach().cpu().numpy())
        
        if sequence_info:
            self.sequence_info.extend(sequence_info)
    
    def compute_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute comprehensive metrics including IC and backtesting metrics.
        
        Returns:
            Dictionary containing metrics for each target
        """
        metrics = {}
        
        for col in self.target_columns:
            if len(self.predictions[col]) > 0:
                pred = np.array(self.predictions[col])
                true = np.array(self.targets[col])
                
                # Basic metrics
                metrics[col] = {
                    'mse': mean_squared_error(true, pred),
                    'rmse': np.sqrt(mean_squared_error(true, pred)),
                    'mae': mean_absolute_error(true, pred),
                    'r2': r2_score(true, pred),
                    'mape': np.mean(np.abs((true - pred) / (true + 1e-8))) * 100
                }
                
                # IC metrics
                ic_metrics = self._compute_ic_metrics(pred, true, col)
                metrics[col].update(ic_metrics)
                
                # Backtesting metrics
                backtest_metrics = self._compute_backtest_metrics(pred, true, col)
                metrics[col].update(backtest_metrics)
        
        return metrics
    
    def _compute_ic_metrics(self, pred: np.ndarray, true: np.ndarray, target: str) -> Dict[str, float]:
        """Compute IC-related metrics."""
        # Ensure pred and true have the same shape
        if pred.shape != true.shape:
            if pred.size == true.size:
                # Reshape to match
                pred = pred.reshape(true.shape)
            else:
                # If sizes don't match, return default values
                return {
                    'correlation': 0.0,
                    'rank_ic': 0.0,
                    'ic_t_stat': 0.0,
                    'ic_p_value': 1.0,
                    'ic_significant': False
                }
        
        # Flatten arrays for correlation calculation
        pred_flat = pred.flatten()
        true_flat = true.flatten()
        
        # Remove NaN values
        mask = ~(np.isnan(pred_flat) | np.isnan(true_flat))
        if mask.sum() < 2:
            return {
                'correlation': 0.0,
                'rank_ic': 0.0,
                'ic_t_stat': 0.0,
                'ic_p_value': 1.0,
                'ic_significant': False
            }
        
        pred_clean = pred_flat[mask]
        true_clean = true_flat[mask]
        
        # Calculate correlation
        try:
            pred_std = np.std(pred_clean)
            true_std = np.std(true_clean)
            if pred_std < 1e-12 or true_std < 1e-12:
                correlation = 0.0
            else:
                correlation = np.corrcoef(pred_clean, true_clean)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
        except Exception:
            correlation = 0.0
        
        # Calculate Rank IC
        try:
            pred_rank = np.argsort(np.argsort(pred_clean))
            true_rank = np.argsort(np.argsort(true_clean))
            rank_ic = np.corrcoef(pred_rank, true_rank)[0, 1]
            if np.isnan(rank_ic):
                rank_ic = 0.0
        except Exception:
            rank_ic = 0.0
        
        # Calculate IC significance (assuming normal distribution)
        n = len(pred_clean)
        if n > 3:
            # Use scipy t-test for robustness
            try:
                # Fisher z-transform approximate test can be unstable at |r|~1; use scipy as primary
                ic_t_stat, ic_p_value = scipy_stats.ttest_1samp([correlation], 0.0)
                # ttest_1samp on a single value is degenerate; fall back to analytical if needed
                if np.isnan(ic_t_stat) or np.isnan(ic_p_value):
                    ic_se = np.sqrt(max(1e-12, (1 - correlation**2)) / (n - 2))
                    ic_t_stat = correlation / ic_se if ic_se > 0 else 0.0
                    # Two-tailed p-value from t-distribution
                    ic_p_value = float(2 * (1 - scipy_stats.t.cdf(abs(ic_t_stat), df=n - 2)))
            except Exception:
                ic_se = np.sqrt(max(1e-12, (1 - correlation**2)) / (n - 2))
                ic_t_stat = correlation / ic_se if ic_se > 0 else 0.0
                ic_p_value = float(2 * (1 - scipy_stats.t.cdf(abs(ic_t_stat), df=n - 2)))
        else:
            ic_t_stat = 0.0
            ic_p_value = 1.0
        
        return {
            'correlation': correlation,
            'rank_ic': rank_ic,
            'ic_t_stat': ic_t_stat,
            'ic_p_value': ic_p_value,
            'ic_significant': ic_p_value < 0.05
        }
    
    def _compute_backtest_metrics(self, pred: np.ndarray, true: np.ndarray, target: str) -> Dict[str, float]:
        """Compute backtesting metrics."""
        # Simple backtesting metrics
        # In a real implementation, you would need more sophisticated backtesting
        
        # Calculate directional accuracy
        pred_direction = np.sign(pred)
        true_direction = np.sign(true)
        directional_accuracy = np.mean(pred_direction == true_direction)
        
        # Calculate hit rate (positive returns)
        hit_rate = np.mean((pred > 0) & (true > 0)) / (np.mean(true > 0) + 1e-8)
        
        # Calculate information ratio (simplified)
        excess_return = pred - true
        information_ratio = np.mean(excess_return) / (np.std(excess_return) + 1e-8)
        
        return {
            'directional_accuracy': directional_accuracy,
            'hit_rate': hit_rate,
            'information_ratio': information_ratio
        }


class FactorForecastingTrainer:
    """
    Enhanced trainer for factor forecasting models with advanced features.
    """
    
    def __init__(self, model: nn.Module, config: Dict, loss_fn: nn.Module = None, 
                 experiment_name: str = None, rank: int = 0, world_size: int = 1):
        """
        Initialize trainer.
        
        Args:
            model: FactorForecastingModel instance
            config: Training configuration
            loss_fn: Custom loss function (optional)
            experiment_name: Name for the experiment
            rank: Process rank for distributed training
            world_size: Total number of processes
        """
        self.model = model
        self.config = config
        self.loss_fn = loss_fn
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.rank = rank
        self.world_size = world_size
        
        # Distributed training setup
        self.is_distributed = world_size > 1
        if self.is_distributed:
            self._setup_distributed()
        
        # Device setup (must be before model.to())
        self.device = torch.device(getattr(config, 'device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Move model to device
        self.model.to(self.device)
        
        # Wrap model with DDP if distributed
        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[rank], output_device=rank)
        
        # Initialize components
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Mixed precision setup
        self.use_mixed_precision = getattr(config, 'use_mixed_precision', True)
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # Gradient accumulation
        self.gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=getattr(config, 'early_stopping_patience', 10),
            min_delta=getattr(config, 'early_stopping_min_delta', 0.001),
            restore_best_weights=getattr(config, 'restore_best_weights', True),
            checkpoint_dir=getattr(config, 'checkpoint_dir', 'checkpoints')
        )
        
        # Metrics tracking
        self.metrics_tracker = MetricsTracker(getattr(config, 'target_columns', ['intra30m', 'nextT1d', 'ema1d']))
        
        # Logging setup
        log_dir = getattr(config, 'log_dir', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=log_dir) if self.rank == 0 and TENSORBOARD_AVAILABLE else None
        
        # Output directory
        self.output_dir = getattr(config, 'output_dir', 'outputs')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Training history
        self.train_history = {
            'train_loss': [], 'val_loss': [], 'learning_rate': [],
            'train_metrics': [], 'val_metrics': []
        }
        
        # Training step counter
        self.current_step = 0
        
        logger.info(f"Initialized trainer with device: {self.device}, distributed: {self.is_distributed}")
    
    def _setup_distributed(self):
        """Setup distributed training."""
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        
        # Set device based on rank
        torch.cuda.set_device(self.rank)
        self.device = torch.device(f'cuda:{self.rank}')
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        optimizer_type = getattr(self.config, 'optimizer', 'adamw')
        lr = getattr(self.config, 'learning_rate', 1e-4)
        weight_decay = getattr(self.config, 'weight_decay', 1e-5)
        
        if optimizer_type == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
    
    def _create_scheduler(self) -> LearningRateScheduler:
        """Create learning rate scheduler."""
        return LearningRateScheduler(
            optimizer=self.optimizer,
            scheduler_type=getattr(self.config, 'scheduler_type', 'cosine'),
            warmup_steps=getattr(self.config, 'warmup_steps', 1000),
            total_steps=getattr(self.config, 'total_steps', 100000),
            initial_lr=getattr(self.config, 'learning_rate', 1e-4),
            min_lr=getattr(self.config, 'min_lr', 1e-6),
            warmup_strategy=getattr(self.config, 'warmup_strategy', 'linear')
        )
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict]:
        """
        Train for one epoch with enhanced features.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss and metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        self.metrics_tracker.reset()
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            # Support both 'features' and 'factors'
            features = batch.get('features', None)
            if features is None:
                features = batch.get('factors')
            # Move to device
            features = features.to(self.device, non_blocking=True)

            # Support both 'stock_ids' and 'stock_id'
            stock_ids = batch.get('stock_ids', None)
            if stock_ids is None:
                stock_ids = batch.get('stock_id')
            stock_ids = stock_ids.to(self.device, non_blocking=True)

            # Normalize shapes: remove singleton extra dims and repeat stock_ids across seq_len if needed
            if features.dim() == 4 and features.size(2) == 1:
                features = features.squeeze(2)
            if stock_ids.dim() == 3 and stock_ids.size(-1) == 1:
                stock_ids = stock_ids.squeeze(-1)
            if stock_ids.dim() == 1:
                # Expand per-time-step IDs to match (batch, seq_len)
                seq_len = features.size(1)
                stock_ids = stock_ids.unsqueeze(1).expand(-1, seq_len)

            targets = {k: v.to(self.device, non_blocking=True) for k, v in batch['targets'].items()}
            # Normalize target shapes to (batch,) when possible
            normalized_targets = {}
            for t_name, t_val in targets.items():
                if t_val.dim() == 3 and t_val.size(-1) == 1:
                    t_val = t_val.squeeze(-1)
                if t_val.dim() == 2:
                    # take last horizon step
                    t_val = t_val[:, -1]
                normalized_targets[t_name] = t_val
            targets = normalized_targets
            sequence_info = batch.get('sequence_info', [])
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with autocast():
                    predictions = self.model(features, stock_ids)
                    if self.loss_fn is not None:
                        # Use custom loss function
                        loss = self.loss_fn(predictions, targets)
                        losses = {'total': loss}  # Create losses dict for consistency
                    else:
                        # Use model's default loss
                        losses = self.model.compute_loss(predictions, targets)
                        loss = losses['total']
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
            else:
                predictions = self.model(features, stock_ids)
                if self.loss_fn is not None:
                    # Use custom loss function
                    loss = self.loss_fn(predictions, targets)
                    losses = {'total': loss}  # Create losses dict for consistency
                else:
                    # Use model's default loss
                    losses = self.model.compute_loss(predictions, targets)
                    loss = losses['total']
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if getattr(self.config, 'gradient_clip', 0) > 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        getattr(self.config, 'gradient_clip')
                    )
                
                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.current_step += 1
            
            total_loss += losses['total'].item()
            
            # Update metrics
            self.metrics_tracker.update(predictions, targets, sequence_info)
            num_batches += 1
            
            # Log progress
            if batch_idx % getattr(self.config, 'log_interval', 100) == 0 and self.rank == 0:
                logger.info(f"Batch {batch_idx}/{len(train_loader)}, "
                           f"Loss: {losses['total'].item():.6f}, "
                           f"LR: {self.scheduler.get_lr():.2e}")
        
        avg_loss = total_loss / num_batches
        metrics = self.metrics_tracker.compute_metrics()
        
        return avg_loss, metrics
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict]:
        """
        Validate the model with enhanced metrics.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average validation loss and metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        self.metrics_tracker.reset()
        
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                features = batch.get('features', None)
                if features is None:
                    features = batch.get('factors')
                features = features.to(self.device, non_blocking=True)
                stock_ids = batch.get('stock_ids', None)
                if stock_ids is None:
                    stock_ids = batch.get('stock_id')
                stock_ids = stock_ids.to(self.device, non_blocking=True)

                if features.dim() == 4 and features.size(2) == 1:
                    features = features.squeeze(2)
                if stock_ids.dim() == 3 and stock_ids.size(-1) == 1:
                    stock_ids = stock_ids.squeeze(-1)
                if stock_ids.dim() == 1:
                    seq_len = features.size(1)
                    stock_ids = stock_ids.unsqueeze(1).expand(-1, seq_len)

                targets = {k: v.to(self.device, non_blocking=True) for k, v in batch['targets'].items()}
                normalized_targets = {}
                for t_name, t_val in targets.items():
                    if t_val.dim() == 3 and t_val.size(-1) == 1:
                        t_val = t_val.squeeze(-1)
                    if t_val.dim() == 2:
                        t_val = t_val[:, -1]
                    normalized_targets[t_name] = t_val
                targets = normalized_targets
                sequence_info = batch.get('sequence_info', [])
                
                # Forward pass
                if self.scaler is not None:
                    with autocast():
                        predictions = self.model(features, stock_ids)
                        losses = self.model.compute_loss(predictions, targets)
                else:
                    predictions = self.model(features, stock_ids)
                    losses = self.model.compute_loss(predictions, targets)
                
                total_loss += losses['total'].item()
                
                # Update metrics
                self.metrics_tracker.update(predictions, targets, sequence_info)
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        metrics = self.metrics_tracker.compute_metrics()
        
        return avg_loss, metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int, resume_from_checkpoint: str = None) -> Dict:
        """
        Train the model with enhanced features.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            resume_from_checkpoint: Path to checkpoint to resume from
            
        Returns:
            Training history
        """
        start_epoch = 0
        # Determine checkpoint to resume from
        chosen_ckpt = None
        if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
            chosen_ckpt = resume_from_checkpoint
        else:
            # Auto resume from latest/best in checkpoint_dir if enabled
            if getattr(self.config, 'auto_resume', True):
                ckpt_dir = getattr(self.config, 'checkpoint_dir', None)
                if ckpt_dir and os.path.isdir(ckpt_dir):
                    best_path = os.path.join(ckpt_dir, 'best_model.pth')
                    if os.path.isfile(best_path):
                        chosen_ckpt = best_path
                    else:
                        candidates = [p for p in os.listdir(ckpt_dir) if p.endswith('.pth')]
                        import re
                        def _epoch_num(name: str) -> int:
                            m = re.findall(r"(\d+)", name)
                            return int(m[-1]) if m else -1
                        candidates.sort(key=_epoch_num)
                        if candidates:
                            chosen_ckpt = os.path.join(ckpt_dir, candidates[-1])

        if chosen_ckpt:
            start_epoch = self.load_checkpoint(chosen_ckpt)
            logger.info(f"Resumed training from epoch {start_epoch} using checkpoint: {chosen_ckpt}")
        
        logger.info(f"Starting training for {num_epochs} epochs from epoch {start_epoch}")
        start_time = time.time()
        
        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_loss, train_metrics = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_metrics = self.validate(val_loader)
            
            # Update learning rate
            current_lr = self.scheduler.get_lr()
            
            # Log epoch results
            epoch_time = time.time() - epoch_start_time
            if self.rank == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                           f"Train Loss: {train_loss:.6f}, "
                           f"Val Loss: {val_loss:.6f}, "
                           f"LR: {current_lr:.2e}, "
                           f"Time: {epoch_time:.2f}s")
                
                # Log detailed metrics
                for target, metrics in val_metrics.items():
                    logger.info(f"  {target}: RMSE={metrics['rmse']:.6f}, "
                               f"R²={metrics['r2']:.4f}, IC={metrics['correlation']:.4f}")
                
                # Log to TensorBoard
                if self.writer:
                    self._log_to_tensorboard(epoch, train_loss, val_loss, current_lr, 
                                           train_metrics, val_metrics)
            
            # Save training history
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['learning_rate'].append(current_lr)
            self.train_history['train_metrics'].append(train_metrics)
            self.train_history['val_metrics'].append(val_metrics)
            
            # Early stopping check
            if self.early_stopping(val_loss, self.model, epoch):
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                self.early_stopping.restore_best_weights(self.model)
                break
        
        # Save final model
        if self.rank == 0:
            self.save_checkpoint("final_model.pth", num_epochs-1, val_loss)
            self.save_training_history()
            self.plot_training_curves()
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time/3600:.2f} hours")
        
        return self.train_history
    
    def _log_to_tensorboard(self, epoch: int, train_loss: float, val_loss: float, 
                           lr: float, train_metrics: Dict, val_metrics: Dict):
        """Log metrics to TensorBoard."""
        # Loss curves
        self.writer.add_scalar('Loss/Train', train_loss, epoch)
        self.writer.add_scalar('Loss/Validation', val_loss, epoch)
        self.writer.add_scalar('Learning_Rate', lr, epoch)
        
        # Metrics for each target
        for target in getattr(self.config, 'target_columns', ['intra30m', 'nextT1d', 'ema1d']):
            if target in val_metrics:
                metrics = val_metrics[target]
                for metric_name, metric_value in metrics.items():
                    self.writer.add_scalar(f'{target}/{metric_name}', metric_value, epoch)
        
        # IC metrics
        for target in getattr(self.config, 'target_columns', ['intra30m', 'nextT1d', 'ema1d']):
            if target in val_metrics:
                ic_metrics = val_metrics[target]
                self.writer.add_scalar(f'{target}/IC_Correlation', ic_metrics.get('correlation', 0), epoch)
                self.writer.add_scalar(f'{target}/Rank_IC', ic_metrics.get('rank_ic', 0), epoch)
                self.writer.add_scalar(f'{target}/IC_Significance', ic_metrics.get('ic_p_value', 1), epoch)
    
    def save_checkpoint(self, filename: str, epoch: int, val_loss: float):
        """
        Save model checkpoint.
        
        Args:
            filename: Checkpoint filename
            epoch: Current epoch
            val_loss: Validation loss
        """
        checkpoint_dir = getattr(self.config, 'checkpoint_dir', 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        
        # Handle scheduler state dict
        try:
            scheduler_state_dict = self.scheduler.state_dict()
        except AttributeError:
            # Custom scheduler doesn't have state_dict method
            scheduler_state_dict = None
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': scheduler_state_dict,
            'val_loss': val_loss,
            'config': self.config,
            'train_history': self.train_history
        }

        # Save scaler state if available
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load checkpoint robustly and return the next epoch to start from."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        state_dict = checkpoint.get('model_state_dict', checkpoint)
        current_keys = list(self.model.state_dict().keys())
        expects_module_prefix = any(k.startswith('module.') for k in current_keys)
        incoming_has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())

        if expects_module_prefix and not incoming_has_module_prefix:
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}
        elif incoming_has_module_prefix and not expects_module_prefix:
            state_dict = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in state_dict.items()}

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            logger.warning(f"Loaded checkpoint with missing={len(missing)} unexpected={len(unexpected)} keys")

        try:
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            logger.warning(f"Optimizer state not loaded: {e}")

        try:
            sched_sd = checkpoint.get('scheduler_state_dict')
            if sched_sd is not None and getattr(self.scheduler, 'scheduler', None) is not None:
                self.scheduler.scheduler.load_state_dict(sched_sd)
        except Exception as e:
            logger.warning(f"Scheduler state not loaded: {e}")

        try:
            scaler_sd = checkpoint.get('scaler_state_dict')
            if scaler_sd is not None and self.scaler is not None:
                self.scaler.load_state_dict(scaler_sd)
        except Exception as e:
            logger.warning(f"Scaler state not loaded: {e}")

        self.train_history = checkpoint.get('train_history', self.train_history)

        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint.get('epoch', -1) + 1
    
    def save_training_history(self):
        """Save training history to JSON file."""
        history_path = os.path.join(self.output_dir, 'logs', 'training_history.json')
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2, default=str)
        
        logger.info(f"Training history saved to {history_path}")
    
    def plot_training_curves(self):
        """Plot training curves and save to TensorBoard."""
        if self.rank != 0 or self.writer is None:  # Only plot on main process
            return
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.train_history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.train_history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Learning rate curve
        axes[0, 1].plot(self.train_history['learning_rate'])
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True)
        
        # IC curves
        target_columns = getattr(self.config, 'target_columns', ['intra30m', 'nextT1d', 'ema1d'])
        for target in target_columns:
            ic_values = [metrics.get(target, {}).get('correlation', 0) for metrics in self.train_history['val_metrics']]
            axes[1, 0].plot(ic_values, label=target)
        
        axes[1, 0].set_title('IC Correlation by Target')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('IC Correlation')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # RMSE curves
        for target in target_columns:
            rmse_values = [metrics.get(target, {}).get('rmse', 0) for metrics in self.train_history['val_metrics']]
            axes[1, 1].plot(rmse_values, label=target)
        
        axes[1, 1].set_title('RMSE by Target')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('RMSE')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save to TensorBoard
        self.writer.add_figure('Training_Curves', fig, 0)
        plt.close()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, 'logs', 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training curves saved to {plot_path}")
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """
        Evaluate the model on test data with comprehensive metrics.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Evaluation metrics
        """
        logger.info("Starting model evaluation")
        
        val_loss, metrics = self.validate(test_loader)
        
        # Log evaluation results
        if self.rank == 0:
            logger.info(f"Test Loss: {val_loss:.6f}")
            for target, target_metrics in metrics.items():
                logger.info(f"{target}:")
                for metric_name, metric_value in target_metrics.items():
                    logger.info(f"  {metric_name}: {metric_value:.6f}")
            
            # Save evaluation results
            eval_path = os.path.join(self.output_dir, 'logs', 'evaluation_results.json')
            os.makedirs(os.path.dirname(eval_path), exist_ok=True)
            
            # Convert numpy types to Python types for JSON serialization
            metrics_serializable = {}
            for target, target_metrics in metrics.items():
                metrics_serializable[target] = {}
                for metric_name, metric_value in target_metrics.items():
                    if isinstance(metric_value, np.bool_):
                        metrics_serializable[target][metric_name] = bool(metric_value)
                    elif isinstance(metric_value, (np.integer, np.int32, np.int64)):
                        metrics_serializable[target][metric_name] = int(metric_value)
                    elif isinstance(metric_value, (np.floating, np.float32, np.float64)):
                        metrics_serializable[target][metric_name] = float(metric_value)
                    else:
                        metrics_serializable[target][metric_name] = metric_value
            
            with open(eval_path, 'w') as f:
                json.dump(metrics_serializable, f, indent=2)
            
            logger.info(f"Evaluation results saved to {eval_path}")
        
        return metrics


def create_trainer(model: nn.Module, config: Dict, rank: int = 0, world_size: int = 1) -> FactorForecastingTrainer:
    """
    Factory function to create a trainer instance with quantitative finance optimization.
    
    Args:
        model: FactorForecastingModel instance
        config: Training configuration
        rank: Process rank for distributed training
        world_size: Total number of processes
        
    Returns:
        FactorForecastingTrainer instance
    """
    # Create loss function based on configuration
    # Handle both dict and object config formats
    if isinstance(config, dict):
        loss_function_type = config.get('loss_function_type', 'quantitative_correlation')
    else:
        loss_function_type = getattr(config, 'loss_function_type', 'quantitative_correlation')
    
    if loss_function_type == 'quantitative_correlation':
        # Use advanced quantitative finance loss function
        try:
            from src.training.quantitative_loss import create_quantitative_loss_function
            loss_fn = create_quantitative_loss_function(config)
            logger.info("Using quantitative correlation loss function")
        except ImportError:
            logger.warning("Quantitative loss function not available, falling back to correlation loss")
            
            # Handle both dict and object config formats
            if isinstance(config, dict):
                correlation_weight = config.get('correlation_weight', 1.0)
                mse_weight = config.get('mse_weight', 0.1)
                rank_weight = config.get('rank_correlation_weight', 0.2)
                target_correlations = config.get('target_correlations', [0.08, 0.05, 0.03])
            else:
                correlation_weight = getattr(config, 'correlation_weight', 1.0)
                mse_weight = getattr(config, 'mse_weight', 0.1)
                rank_weight = getattr(config, 'rank_correlation_weight', 0.2)
                target_correlations = getattr(config, 'target_correlations', [0.08, 0.05, 0.03])
            
            loss_fn = CorrelationLoss(
                correlation_weight=correlation_weight,
                mse_weight=mse_weight,
                rank_weight=rank_weight,
                target_correlations=target_correlations
            )
    elif loss_function_type == 'correlation_loss':
        # Handle both dict and object config formats
        if isinstance(config, dict):
            correlation_weight = config.get('correlation_weight', 1.0)
            mse_weight = config.get('mse_weight', 0.1)
            rank_weight = config.get('rank_correlation_weight', 0.1)
            target_correlations = config.get('target_correlations', [0.1, 0.05, 0.08])
        else:
            correlation_weight = getattr(config, 'correlation_weight', 1.0)
            mse_weight = getattr(config, 'mse_weight', 0.1)
            rank_weight = getattr(config, 'rank_correlation_weight', 0.1)
            target_correlations = getattr(config, 'target_correlations', [0.1, 0.05, 0.08])
        
        loss_fn = CorrelationLoss(
            correlation_weight=correlation_weight,
            mse_weight=mse_weight,
            rank_weight=rank_weight,
            target_correlations=target_correlations
        )
    elif loss_function_type == 'mse':
        loss_fn = nn.MSELoss()
    elif loss_function_type == 'huber':
        loss_fn = nn.HuberLoss()
    else:
        # Default to quantitative correlation loss
        try:
            from src.training.quantitative_loss import create_quantitative_loss_function
            loss_fn = create_quantitative_loss_function(config)
        except ImportError:
            loss_fn = CorrelationLoss()
    
    return FactorForecastingTrainer(model, config, loss_fn, rank=rank, world_size=world_size)


if __name__ == "__main__":
    # Test the enhanced trainer
    from src.models.models import create_model
    
    config = {
        'num_factors': 100,
        'num_stocks': 1000,
        'd_model': 256,
        'num_heads': 8,
        'num_layers': 6,
        'd_ff': 1024,
        'dropout': 0.1,
        'max_seq_len': 50,
        'target_columns': ['intra30m', 'nextT1d', 'ema1d'],
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'optimizer': 'adamw',
        'scheduler_type': 'cosine',
        'warmup_steps': 1000,
        'total_steps': 100000,
        'early_stopping_patience': 10,
        'gradient_clip': 1.0,
        'gradient_accumulation_steps': 4,
        'use_mixed_precision': True,
        'device': 'cpu',
        'output_dir': 'test_outputs',
        'checkpoint_dir': 'test_checkpoints',
        'log_dir': 'test_logs'
    }
    
    model = create_model(config)
    trainer = create_trainer(model, config)
    
    print("Enhanced trainer test successful!")
    print(f"Model parameters: {model.get_model_info()['total_parameters']:,}")
    print(f"FlashAttention available: {FLASH_ATTENTION_AVAILABLE}") 