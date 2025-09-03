"""
Incremental learning framework for online model updates.
Supports continual learning with memory replay and adaptive training.
Integrated with async data preloading for efficient processing.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import copy

logger = logging.getLogger(__name__)

# Import async preloader for integration
try:
    from src.data_processing.async_preloader import AsyncDataPreloader
    ASYNC_PRELOADER_AVAILABLE = True
except ImportError:
    ASYNC_PRELOADER_AVAILABLE = False
    logger.warning("AsyncDataPreloader not available, falling back to standard data loading")


class ExperienceReplay:
    """Experience replay buffer for incremental learning."""
    
    def __init__(self, capacity: int = 10000, diversity_sampling: bool = True):
        """
        Args:
            capacity: Maximum number of samples to store
            diversity_sampling: Whether to use diversity-based sampling
        """
        self.capacity = capacity
        self.diversity_sampling = diversity_sampling
        self.buffer = deque(maxlen=capacity)
        self.importance_weights = deque(maxlen=capacity)
        
    def add(self, experience: Dict[str, torch.Tensor], importance: float = 1.0):
        """Add experience to replay buffer."""
        self.buffer.append(experience)
        self.importance_weights.append(importance)
        
    def sample(self, batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """Sample experiences from buffer."""
        if len(self.buffer) == 0:
            return []
            
        if not self.diversity_sampling or len(self.buffer) <= batch_size:
            # Random sampling
            indices = np.random.choice(len(self.buffer), 
                                     min(batch_size, len(self.buffer)), 
                                     replace=False)
        else:
            # Importance-weighted sampling
            weights = np.array(list(self.importance_weights))
            weights = weights / weights.sum()
            indices = np.random.choice(len(self.buffer), 
                                     batch_size, 
                                     replace=False, 
                                     p=weights)
        
        return [self.buffer[i] for i in indices]
    
    def size(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
        self.importance_weights.clear()


class IncrementalTrainer:
    """Incremental training manager for continuous learning."""
    
    def __init__(self, 
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion: nn.Module,
                 device: torch.device,
                 replay_capacity: int = 10000,
                 replay_ratio: float = 0.3,
                 adaptation_threshold: float = 0.1,
                 performance_window: int = 100,
                 use_async_preloader: bool = True):
        """
        Args:
            model: Neural network model
            optimizer: Optimizer for training
            criterion: Loss function
            device: Training device
            replay_capacity: Experience replay buffer size
            replay_ratio: Ratio of replay samples in each batch
            adaptation_threshold: Performance drop threshold for adaptation
            performance_window: Window size for performance monitoring
            use_async_preloader: Whether to use async data preloading
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        self.replay_buffer = ExperienceReplay(replay_capacity)
        self.replay_ratio = replay_ratio
        self.adaptation_threshold = adaptation_threshold
        
        # Performance monitoring
        self.performance_history = deque(maxlen=performance_window)
        self.baseline_performance = None
        
        # Model checkpoints
        self.best_model_state = None
        self.checkpoint_performance = float('inf')
        
        # Async preloader setup
        self.use_async_preloader = use_async_preloader and ASYNC_PRELOADER_AVAILABLE
        self.async_preloader = None
    
    def __del__(self):
        """Cleanup when trainer is destroyed."""
        if self.async_preloader is not None:
            try:
                self.async_preloader.stop_preloading()
            except Exception:
                pass  # Ignore cleanup errors
        
    def train_step(self, batch: Dict[str, torch.Tensor], 
                   store_experience: bool = True) -> Dict[str, float]:
        """Single training step with optional experience storage."""
        
        self.model.train()
        
        # Move tensor fields to device, keep metadata fields as-is
        batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        
        # Build default stock_ids when not provided (shape: batch, seq_len)
        features = batch['features']
        if features.dim() == 3:
            bsz, seq_len, _ = features.shape
        else:
            # Fallback: treat as (batch, features) with seq_len=1
            bsz, seq_len = features.shape[0], 1
            features = features.view(bsz, seq_len, -1)
            batch['features'] = features

        stock_ids = batch.get('stock_ids')
        if stock_ids is None:
            stock_ids = torch.zeros((bsz, seq_len), dtype=torch.long, device=self.device)

        # Forward pass
        outputs = self.model(features, stock_ids)

        # Align model outputs (dict of targets) to a tensor matching targets
        if isinstance(outputs, dict):
            # Collect predictions in a stable order
            pred_list = []
            for key in sorted(outputs.keys()):
                pred_k = outputs[key]
                if pred_k.dim() >= 2:
                    pred_k = pred_k[:, -1]
                pred_list.append(pred_k)
            preds_tensor = torch.stack(pred_list, dim=1) if pred_list else None
        else:
            preds_tensor = outputs

        targets = batch['targets']
        if preds_tensor is not None:
            # Truncate to smallest common dim
            m = min(preds_tensor.size(1) if preds_tensor.dim() > 1 else 1, targets.size(1) if targets.dim() > 1 else 1)
            if preds_tensor.dim() == 1:
                preds_tensor = preds_tensor.unsqueeze(1)
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            loss = self.criterion(preds_tensor[:, :m], targets[:, :m])
        else:
            loss = self.criterion(outputs, targets)
        
        # Store experience for replay (always store when requested to ensure stability in tests)
        if store_experience:
            # Store only tensor fields; skip heavy or non-tensor metadata
            experience = {k: v.detach().cpu().clone() for k, v in batch.items() if isinstance(v, torch.Tensor)}
            importance = float(loss.item())
            self.replay_buffer.add(experience, importance)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Clear gradients to free memory
        loss_value = loss.item()
        del loss, outputs
        
        return {'loss': loss_value}
    
    def incremental_update(self, new_data_loader, 
                          validation_loader: Optional[DataLoader] = None,
                          epochs: int = 1) -> Dict[str, Any]:
        """Perform incremental update with new data."""
        
        logger.info(f"Starting incremental update with {epochs} epochs")
        
        # Setup async preloader if enabled
        if self.use_async_preloader and self.async_preloader is None:
            try:
                # Initialize AsyncDataPreloader in batch mode
                def _dl_func():
                    return new_data_loader
                self.async_preloader = AsyncDataPreloader(
                    data_loader_func=_dl_func,
                    queue_size=2,
                    num_workers=1,
                    prefetch_size=2
                )
                self.async_preloader.start_preloading()
                logger.info("Async data preloading enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize async preloader: {e}")
                self.use_async_preloader = False
        
        # Check if adaptation is needed
        if self._should_adapt(validation_loader):
            logger.info("Performance degradation detected, enabling replay")
            replay_enabled = True
        else:
            replay_enabled = False
        
        training_stats = []
        
        for epoch in range(epochs):
            epoch_losses = []
            
            # Use async preloader if available, otherwise standard data loader
            data_iterator = self.async_preloader if self.use_async_preloader else new_data_loader
            
            for batch in data_iterator:
                # Train on new data
                stats = self.train_step(batch, store_experience=True)
                epoch_losses.append(stats['loss'])
                
                # Experience replay
                if replay_enabled and self.replay_buffer.size() > 0:
                    replay_batch_size = int(len(batch['features']) * self.replay_ratio)
                    if replay_batch_size > 0:
                        replay_samples = self.replay_buffer.sample(replay_batch_size)
                        if replay_samples:
                            replay_batch = self._collate_experiences(replay_samples)
                            # Move replay batch to device
                            replay_batch = {k: v.to(self.device) for k, v in replay_batch.items()}
                            replay_stats = self.train_step(replay_batch, store_experience=False)
            
            avg_loss = np.mean(epoch_losses)
            training_stats.append({'epoch': epoch, 'loss': avg_loss})
            
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Evaluate performance
        if validation_loader:
            val_performance = self._evaluate(validation_loader)
            self.performance_history.append(val_performance)
            
            # Update checkpoint if improved
            if val_performance < self.checkpoint_performance:
                self.checkpoint_performance = val_performance
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                logger.info(f"New best model saved with performance: {val_performance:.4f}")
        
        # Cleanup async preloader if it was created for this update
        if self.async_preloader is not None:
            try:
                self.async_preloader.stop_preloading()
                self.async_preloader = None
                logger.info("Async preloader stopped and cleaned up")
            except Exception as e:
                logger.warning(f"Error cleaning up async preloader: {e}")
        
        return {
            'training_stats': training_stats,
            'replay_enabled': replay_enabled,
            'replay_buffer_size': self.replay_buffer.size(),
            'validation_performance': val_performance if validation_loader else None
        }
    
    def _should_adapt(self, validation_loader) -> bool:
        """Check if model adaptation is needed based on performance."""
        if validation_loader is None or len(self.performance_history) < 10:
            return False
        
        current_performance = self._evaluate(validation_loader)
        
        if self.baseline_performance is None:
            self.baseline_performance = current_performance
            return False
        
        performance_drop = (current_performance - self.baseline_performance) / self.baseline_performance
        return performance_drop > self.adaptation_threshold
    
    def _evaluate(self, data_loader) -> float:
        """Evaluate model performance."""
        self.model.eval()
        total_loss = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch in data_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                features = batch['features']
                if features.dim() == 3:
                    bsz, seq_len, _ = features.shape
                else:
                    bsz, seq_len = features.shape[0], 1
                    features = features.view(bsz, seq_len, -1)
                stock_ids = batch.get('stock_ids')
                if stock_ids is None:
                    stock_ids = torch.zeros((bsz, seq_len), dtype=torch.long, device=self.device)
                outputs = self.model(features, stock_ids)
                if isinstance(outputs, dict):
                    pred_list = []
                    for key in sorted(outputs.keys()):
                        pred_k = outputs[key]
                        if pred_k.dim() >= 2:
                            pred_k = pred_k[:, -1]
                        pred_list.append(pred_k)
                    preds_tensor = torch.stack(pred_list, dim=1) if pred_list else None
                else:
                    preds_tensor = outputs
                targets = batch['targets']
                if preds_tensor is not None:
                    if preds_tensor.dim() == 1:
                        preds_tensor = preds_tensor.unsqueeze(1)
                    if targets.dim() == 1:
                        targets = targets.unsqueeze(1)
                    m = min(preds_tensor.size(1), targets.size(1))
                    loss = self.criterion(preds_tensor[:, :m], targets[:, :m])
                else:
                    loss = self.criterion(outputs, targets)
                
                total_loss += loss.item() * len(batch['features'])
                num_samples += len(batch['features'])
        
        return total_loss / num_samples if num_samples > 0 else float('inf')
    
    def _collate_experiences(self, experiences: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate experience samples into a batch."""
        if not experiences:
            return {}
        
        batch = {}
        for key in experiences[0].keys():
            batch[key] = torch.stack([exp[key] for exp in experiences])
        
        return batch
    
    def save_state(self, filepath: str):
        """Save trainer state including replay buffer."""
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_model_state': self.best_model_state,
            'checkpoint_performance': self.checkpoint_performance,
            'baseline_performance': self.baseline_performance,
            'performance_history': list(self.performance_history),
            'replay_buffer': {
                'buffer': list(self.replay_buffer.buffer),
                'importance_weights': list(self.replay_buffer.importance_weights)
            }
        }
        
        torch.save(state, filepath)
        logger.info(f"Trainer state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load trainer state including replay buffer."""
        state = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.best_model_state = state.get('best_model_state')
        self.checkpoint_performance = state.get('checkpoint_performance', float('inf'))
        self.baseline_performance = state.get('baseline_performance')
        
        # Restore performance history
        if 'performance_history' in state:
            self.performance_history.extend(state['performance_history'])
        
        # Restore replay buffer
        if 'replay_buffer' in state:
            buffer_data = state['replay_buffer']
            self.replay_buffer.buffer.extend(buffer_data['buffer'])
            self.replay_buffer.importance_weights.extend(buffer_data['importance_weights'])
        
        logger.info(f"Trainer state loaded from {filepath}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics and status."""
        return {
            'replay_buffer_size': self.replay_buffer.size(),
            'replay_capacity': self.replay_buffer.capacity,
            'performance_history_size': len(self.performance_history),
            'baseline_performance': self.baseline_performance,
            'checkpoint_performance': self.checkpoint_performance,
            'has_best_model': self.best_model_state is not None
        }


class OnlineModelManager:
    """Manager for online model updates and deployment."""
    
    def __init__(self, 
                 model_config: Dict[str, Any],
                 checkpoint_dir: str = "checkpoints",
                 update_frequency: str = "daily",
                 max_models: int = 10):
        """
        Args:
            model_config: Configuration for model creation
            checkpoint_dir: Directory for storing model checkpoints
            update_frequency: How often to update ("daily", "weekly", etc.)
            max_models: Maximum number of models to keep
        """
        self.model_config = model_config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.update_frequency = update_frequency
        self.max_models = max_models
        
        self.current_model = None
        self.trainer = None
        
    def initialize_model(self, model_class, initial_data_loader=None):
        """Initialize the model and trainer."""
        # Create model
        self.current_model = model_class(**self.model_config)
        
        # Setup trainer
        optimizer = torch.optim.Adam(self.current_model.parameters())
        criterion = nn.MSELoss()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.trainer = IncrementalTrainer(
            model=self.current_model,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )
        
        # Initial training if data provided
        if initial_data_loader:
            logger.info("Performing initial model training")
            self.trainer.incremental_update(initial_data_loader)
        
    def update_model(self, new_data_loader, validation_loader=None) -> Dict[str, Any]:
        """Update model with new data."""
        if self.trainer is None:
            raise ValueError("Model not initialized. Call initialize_model first.")
        
        # Perform incremental update
        update_stats = self.trainer.incremental_update(
            new_data_loader, 
            validation_loader
        )
        
        # Save checkpoint
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = self.checkpoint_dir / f"model_checkpoint_{timestamp}.pt"
        self.trainer.save_state(str(checkpoint_path))
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
        
        return {
            'checkpoint_path': str(checkpoint_path),
            'update_stats': update_stats,
            'trainer_stats': self.trainer.get_statistics()
        }
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoint files."""
        checkpoints = list(self.checkpoint_dir.glob("model_checkpoint_*.pt"))
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Keep only the most recent checkpoints
        for checkpoint in checkpoints[self.max_models:]:
            checkpoint.unlink()
            logger.info(f"Removed old checkpoint: {checkpoint.name}")
    
    def load_latest_checkpoint(self):
        """Load the most recent model checkpoint."""
        checkpoints = list(self.checkpoint_dir.glob("model_checkpoint_*.pt"))
        if not checkpoints:
            logger.warning("No checkpoints found")
            return
        
        latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
        self.trainer.load_state(str(latest_checkpoint))
        logger.info(f"Loaded checkpoint: {latest_checkpoint.name}")
    
    def get_model(self):
        """Get the current model for inference."""
        return self.current_model
    
    def get_status(self) -> Dict[str, Any]:
        """Get manager status and statistics."""
        status = {
            'model_initialized': self.current_model is not None,
            'checkpoint_dir': str(self.checkpoint_dir),
            'update_frequency': self.update_frequency,
            'available_checkpoints': len(list(self.checkpoint_dir.glob("model_checkpoint_*.pt")))
        }
        
        if self.trainer:
            status['trainer_stats'] = self.trainer.get_statistics()
        
        return status
