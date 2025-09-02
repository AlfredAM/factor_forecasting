"""
Integrated training module that combines streaming data loading, 
async preloading, and incremental learning for optimal performance.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta

from src.data_processing.streaming_data_loader import StreamingDataLoader
from src.data_processing.async_preloader import AsyncDataPreloader
from src.training.incremental_learning import IncrementalTrainer, OnlineModelManager
from src.training.trainer import FactorForecastingTrainer
from src.models.models import FactorForecastingModel
from configs.config import ModelConfig

logger = logging.getLogger(__name__)


class IntegratedTrainingSystem:
    """
    Integrated training system that combines all advanced features:
    - Streaming data loading for memory efficiency
    - Async preloading for GPU utilization
    - Incremental learning for online adaptation
    - Model checkpointing and management
    """
    
    def __init__(self, 
                 config: ModelConfig,
                 device: torch.device = None,
                 use_incremental_learning: bool = True,
                 use_async_preloader: bool = True):
        """
        Initialize the integrated training system.
        
        Args:
            config: Model configuration
            device: Training device
            use_incremental_learning: Enable incremental learning
            use_async_preloader: Enable async data preloading
        """
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_incremental_learning = use_incremental_learning
        self.use_async_preloader = use_async_preloader
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.streaming_loader = None
        self.async_preloader = None
        self.incremental_trainer = None
        self.standard_trainer = None
        self.model_manager = None
        
        logger.info(f"Integrated training system initialized on {self.device}")
        logger.info(f"Incremental learning: {self.use_incremental_learning}")
        logger.info(f"Async preloading: {self.use_async_preloader}")
    
    def setup_model(self, model_class=None) -> None:
        """Setup the model, optimizer, and loss function."""
        if model_class is None:
            model_class = FactorForecastingModel
        
        # Create model config dictionary
        model_config = {
            'input_dim': len(self.config.factor_columns),
            'hidden_size': getattr(self.config, 'hidden_size', getattr(self.config, 'hidden_dim', 128)),
            'num_layers': self.config.num_layers,
            'output_dim': len(self.config.target_columns),
            'dropout': self.config.dropout,
            'model_type': self.config.model_type,
            'max_seq_len': getattr(self.config, 'sequence_length', 30),
            'num_heads': getattr(self.config, 'num_heads', 8)
        }
        
        # Create model
        self.model = model_class(model_config).to(self.device)
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Setup loss function
        if hasattr(self.config, 'loss_type') and self.config.loss_type == 'correlation':
            from src.training.trainer import CorrelationLoss
            self.criterion = CorrelationLoss()
        else:
            self.criterion = nn.MSELoss()
        
        logger.info(f"Model setup complete: {self.model.__class__.__name__}")
        logger.info(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def setup_data_loading(self, data_dir: str) -> None:
        """Setup streaming data loading with optional async preloading."""
        # Initialize streaming data loader
        self.streaming_loader = StreamingDataLoader(
            data_dir=data_dir,
            batch_size=self.config.batch_size,
            cache_size=getattr(self.config, 'cache_size', 10),
            max_memory_mb=getattr(self.config, 'max_memory_mb', 8192)
        )
        
        logger.info(f"Streaming data loader initialized")
        logger.info(f"Data files found: {len(self.streaming_loader.data_files)}")
    
    def setup_training_components(self) -> None:
        """Setup training components based on configuration."""
        if self.use_incremental_learning:
            # Setup incremental trainer
            self.incremental_trainer = IncrementalTrainer(
                model=self.model,
                optimizer=self.optimizer,
                criterion=self.criterion,
                device=self.device,
                replay_capacity=getattr(self.config, 'replay_capacity', 10000),
                replay_ratio=getattr(self.config, 'replay_ratio', 0.3),
                adaptation_threshold=getattr(self.config, 'adaptation_threshold', 0.1),
                use_async_preloader=self.use_async_preloader
            )
            
            # Setup online model manager
            self.model_manager = OnlineModelManager(
                model_config=self.config.__dict__,
                checkpoint_dir=getattr(self.config, 'checkpoint_dir', 'checkpoints'),
                max_models=getattr(self.config, 'max_models', 10)
            )
            
            logger.info("Incremental learning components initialized")
        else:
            # Use a simple built-in training loop for standard training path
            self.standard_trainer = None
            logger.info("Standard training will use a simple internal loop")
    
    def train_epoch(self, train_loader: DataLoader, 
                   validation_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """
        Train for one epoch using the appropriate trainer.
        
        Args:
            train_loader: Training data loader
            validation_loader: Optional validation data loader
            
        Returns:
            Training statistics
        """
        if self.use_incremental_learning:
            # Use incremental trainer
            return self.incremental_trainer.incremental_update(
                new_data_loader=train_loader,
                validation_loader=validation_loader,
                epochs=1
            )
        else:
            # Use standard trainer with async preloading if enabled
            if self.use_async_preloader and self.async_preloader is None:
                self.async_preloader = AsyncDataPreloader(
                    data_loader=train_loader,
                    device=self.device,
                    buffer_size=2,
                    prefetch_factor=2
                )
                self.async_preloader.start_preloading()
            
            # Training loop
            self.model.train()
            epoch_losses = []
            
            data_iterator = self.async_preloader if self.use_async_preloader else train_loader
            
            for batch in data_iterator:
                if not self.use_async_preloader:
                    batch = {k: (v.to(self.device) if hasattr(v, 'to') else v) for k, v in batch.items()}

                # Prepare inputs
                features = batch['features']
                if features.dim() == 2:
                    bsz = features.size(0)
                    features = features.view(bsz, 1, -1)
                seq_len = features.size(1)
                stock_ids = batch.get('stock_ids')
                if stock_ids is None:
                    stock_ids = torch.zeros((features.size(0), seq_len), dtype=torch.long, device=self.device)
                elif stock_ids.dim() == 1:
                    stock_ids = stock_ids.unsqueeze(1).expand(-1, seq_len)
                elif stock_ids.dim() == 2 and stock_ids.size(1) == 1:
                    stock_ids = stock_ids.expand(-1, seq_len)

                # Forward pass
                outputs = self.model(features, stock_ids)

                # Align outputs to targets
                targets = batch['targets']
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

                if preds_tensor is not None:
                    if preds_tensor.dim() == 1:
                        preds_tensor = preds_tensor.unsqueeze(1)
                    if targets.dim() == 1:
                        targets = targets.unsqueeze(1)
                    m = min(preds_tensor.size(1), targets.size(1))
                    loss = self.criterion(preds_tensor[:, :m], targets[:, :m])
                else:
                    loss = self.criterion(outputs, targets)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_losses.append(loss.item())
            
            # Cleanup async preloader after epoch
            if self.async_preloader is not None:
                self.async_preloader.stop_preloading()
                self.async_preloader = None
            
            return {
                'training_stats': [{'epoch': 0, 'loss': np.mean(epoch_losses)}],
                'avg_loss': np.mean(epoch_losses)
            }
    
    def train(self, data_dir: str, epochs: int = 100, 
              validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Main training loop.
        
        Args:
            data_dir: Directory containing training data
            epochs: Number of training epochs
            validation_split: Fraction of data for validation
            
        Returns:
            Training results
        """
        logger.info(f"Starting integrated training for {epochs} epochs")
        
        # Setup all components
        if self.model is None:
            self.setup_model()
        
        if self.streaming_loader is None:
            self.setup_data_loading(data_dir)
        
        if self.incremental_trainer is None and self.standard_trainer is None:
            self.setup_training_components()
        
        # Create data loaders using streaming approach
        train_loader, val_loader, test_loader = self.streaming_loader.create_data_loaders(
            train_ratio=1.0 - validation_split,
            val_ratio=validation_split,
            test_ratio=0.0
        )
        
        # Training loop
        training_history = []
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            logger.info(f"Starting epoch {epoch+1}/{epochs}")
            
            # Train for one epoch
            epoch_stats = self.train_epoch(train_loader, val_loader)
            
            # Extract loss information
            if 'training_stats' in epoch_stats:
                avg_loss = epoch_stats['training_stats'][0]['loss']
            else:
                avg_loss = epoch_stats.get('avg_loss', 0.0)
            
            # Validation
            val_loss = None
            if val_loader is not None:
                val_loss = self._evaluate(val_loader)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint(epoch, val_loss)
            
            training_history.append({
                'epoch': epoch,
                'train_loss': avg_loss,
                'val_loss': val_loss
            })
            
            logger.info(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}" + 
                       (f", Val Loss: {val_loss:.4f}" if val_loss else ""))
        
        logger.info("Training completed successfully")
        
        return {
            'training_history': training_history,
            'best_val_loss': best_val_loss,
            'final_model_state': self.model.state_dict()
        }
    
    def _evaluate(self, data_loader: DataLoader) -> float:
        """Evaluate model on given data loader."""
        self.model.eval()
        total_loss = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch in data_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(batch['features'])
                loss = self.criterion(outputs, batch['targets'])
                
                total_loss += loss.item() * len(batch['features'])
                num_samples += len(batch['features'])
        
        return total_loss / num_samples if num_samples > 0 else float('inf')
    
    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """Save model checkpoint."""
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"integrated_model_epoch_{epoch}_loss_{val_loss:.4f}.pt"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config.__dict__
        }, checkpoint_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def predict(self, data_loader: DataLoader) -> np.ndarray:
        """Make predictions using the trained model."""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(batch['features'])
                predictions.append(outputs.cpu().numpy())
        
        return np.concatenate(predictions, axis=0)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the training system."""
        status = {
            'device': str(self.device),
            'use_incremental_learning': self.use_incremental_learning,
            'use_async_preloader': self.use_async_preloader,
            'model_initialized': self.model is not None,
            'streaming_loader_initialized': self.streaming_loader is not None
        }
        
        if self.incremental_trainer:
            status.update(self.incremental_trainer.get_statistics())
        
        return status


def create_integrated_training_system(config: ModelConfig, 
                                     **kwargs) -> IntegratedTrainingSystem:
    """
    Factory function to create an integrated training system.
    
    Args:
        config: Model configuration
        **kwargs: Additional arguments for the training system
        
    Returns:
        Configured IntegratedTrainingSystem instance
    """
    return IntegratedTrainingSystem(config, **kwargs)
