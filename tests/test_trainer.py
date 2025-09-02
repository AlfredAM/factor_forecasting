"""
Tests for factor forecasting trainer
"""
import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import tempfile
import shutil

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.training.trainer import (
    FactorForecastingTrainer, 
    CorrelationLoss, 
    EarlyStopping, 
    LearningRateScheduler,
    MetricsTracker,
    create_trainer
)
from src.models.model_factory import create_model


class TestCorrelationLoss(unittest.TestCase):
    """Test correlation loss function"""
    
    def setUp(self):
        self.batch_size = 32
        self.target_columns = ['intra30m', 'nextT1d', 'ema1d']
        
    def test_correlation_loss(self):
        """Test correlation loss computation"""
        loss_fn = CorrelationLoss(
            correlation_weight=1.0,
            mse_weight=0.1,
            rank_weight=0.1,
            target_correlations=[0.1, 0.05, 0.08]
        )
        
        # Create dummy predictions and targets
        predictions = {
            'intra30m': torch.randn(self.batch_size),
            'nextT1d': torch.randn(self.batch_size),
            'ema1d': torch.randn(self.batch_size)
        }
        
        targets = {
            'intra30m': torch.randn(self.batch_size),
            'nextT1d': torch.randn(self.batch_size),
            'ema1d': torch.randn(self.batch_size)
        }
        
        loss = loss_fn(predictions, targets)
        
        # Check that loss is a scalar tensor
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(loss.item() > 0)


class TestEarlyStopping(unittest.TestCase):
    """Test early stopping mechanism"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.early_stopping = EarlyStopping(
            patience=3,
            min_delta=0.001,
            restore_best_weights=True,
            checkpoint_dir=self.temp_dir
        )
        self.model = nn.Linear(10, 1)
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def test_early_stopping_improvement(self):
        """Test early stopping with improving loss"""
        losses = [1.0, 0.9, 0.8, 0.7, 0.6]
        
        for i, loss in enumerate(losses):
            should_stop = self.early_stopping(loss, self.model, i)
            if i < len(losses) - 1:  # Should not stop before patience is reached
                self.assertFalse(should_stop)
                
    def test_early_stopping_no_improvement(self):
        """Test early stopping with no improvement"""
        losses = [1.0, 1.1, 1.2, 1.3, 1.4]  # No improvement
        
        for i, loss in enumerate(losses):
            should_stop = self.early_stopping(loss, self.model, i)
            if i >= 3:  # Should stop after patience
                self.assertTrue(should_stop)
                break


class TestLearningRateScheduler(unittest.TestCase):
    """Test learning rate scheduler"""
    
    def setUp(self):
        self.optimizer = torch.optim.Adam([torch.randn(10, requires_grad=True)], lr=1e-4)
        
    def test_cosine_scheduler(self):
        """Test cosine annealing scheduler"""
        scheduler = LearningRateScheduler(
            optimizer=self.optimizer,
            scheduler_type='cosine',
            warmup_steps=10,
            total_steps=100,
            initial_lr=1e-4,
            min_lr=1e-6
        )
        
        # Test warmup phase
        for step in range(10):
            # Simulate optimizer step before scheduler step
            self.optimizer.step()
            scheduler.step()
            lr = scheduler.get_lr()
            self.assertTrue(lr >= 0)  # Allow zero during warmup
            
        # Test main phase
        for step in range(10, 50):
            # Simulate optimizer step before scheduler step
            self.optimizer.step()
            scheduler.step()
            lr = scheduler.get_lr()
            self.assertTrue(lr >= 1e-6)


class TestMetricsTracker(unittest.TestCase):
    """Test metrics tracker"""
    
    def setUp(self):
        self.target_columns = ['intra30m', 'nextT1d', 'ema1d']
        self.tracker = MetricsTracker(self.target_columns)
        
    def test_metrics_tracker_update(self):
        """Test metrics tracker update"""
        batch_size = 16
        
        predictions = {
            'intra30m': torch.randn(batch_size),
            'nextT1d': torch.randn(batch_size),
            'ema1d': torch.randn(batch_size)
        }
        
        targets = {
            'intra30m': torch.randn(batch_size),
            'nextT1d': torch.randn(batch_size),
            'ema1d': torch.randn(batch_size)
        }
        
        self.tracker.update(predictions, targets)
        
        # Check that metrics can be computed
        metrics = self.tracker.compute_metrics()
        
        for target in self.target_columns:
            self.assertIn(target, metrics)
            target_metrics = metrics[target]
            self.assertIn('mse', target_metrics)
            self.assertIn('rmse', target_metrics)
            self.assertIn('correlation', target_metrics)
            self.assertIn('rank_ic', target_metrics)
            
    def test_metrics_tracker_reset(self):
        """Test metrics tracker reset"""
        batch_size = 16
        
        predictions = {
            'intra30m': torch.randn(batch_size),
            'nextT1d': torch.randn(batch_size),
            'ema1d': torch.randn(batch_size)
        }
        
        targets = {
            'intra30m': torch.randn(batch_size),
            'nextT1d': torch.randn(batch_size),
            'ema1d': torch.randn(batch_size)
        }
        
        self.tracker.update(predictions, targets)
        self.tracker.reset()
        
        # After reset, should have no data
        metrics = self.tracker.compute_metrics()
        self.assertEqual(len(metrics), 0)


class TestFactorForecastingTrainer(unittest.TestCase):
    """Test factor forecasting trainer"""
    
    def setUp(self):
        self.config = {
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
            'warmup_steps': 100,
            'total_steps': 1000,
            'early_stopping_patience': 5,
            'gradient_clip': 1.0,
            'gradient_accumulation_steps': 1,
            'use_mixed_precision': False,  # Disable for testing
            'device': 'cpu',
            'output_dir': 'test_outputs',
            'checkpoint_dir': 'test_checkpoints',
            'log_dir': 'test_logs'
        }
        
        # Create temporary directories
        self.temp_dirs = []
        for dir_name in ['test_outputs', 'test_checkpoints', 'test_logs']:
            temp_dir = tempfile.mkdtemp()
            self.temp_dirs.append(temp_dir)
            self.config[dir_name.replace('test_', '') + '_dir'] = temp_dir
        
    def tearDown(self):
        # Clean up temporary directories
        for temp_dir in self.temp_dirs:
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    def test_trainer_initialization(self):
        """Test trainer initialization"""
        model = create_model(self.config)
        trainer = create_trainer(model, self.config)
        
        self.assertIsInstance(trainer, FactorForecastingTrainer)
        self.assertIn(trainer.device.type, ['cpu', 'cuda'])
        self.assertFalse(trainer.is_distributed)
        
    def test_trainer_model_info(self):
        """Test trainer model information"""
        model = create_model(self.config)
        trainer = create_trainer(model, self.config)
        
        model_info = model.get_model_info()
        self.assertIn('total_parameters', model_info)
        self.assertIn('model_size_mb', model_info)
        
    def test_trainer_components(self):
        """Test trainer components"""
        model = create_model(self.config)
        trainer = create_trainer(model, self.config)
        
        # Check that components are initialized
        self.assertIsNotNone(trainer.optimizer)
        self.assertIsNotNone(trainer.scheduler)
        self.assertIsNotNone(trainer.early_stopping)
        self.assertIsNotNone(trainer.metrics_tracker)


if __name__ == '__main__':
    unittest.main() 