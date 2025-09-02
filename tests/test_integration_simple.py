#!/usr/bin/env python3
"""
Simplified integration test for the factor forecasting system
Tests core functionality without external dependencies
"""
import torch
import pandas as pd
import numpy as np
import sys
import os
import time
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.models.model_factory import create_model, save_model, load_model
from src.training.trainer import create_trainer, CorrelationLoss

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_data(num_stocks=100, num_days=50, num_factors=100):
    """Create sample data for testing"""
    logger.info("Creating sample data...")
    
    # Generate stock IDs
    stock_ids = [f"{i:06d}" for i in range(num_stocks)]
    
    # Generate dates
    dates = pd.date_range('2023-01-01', periods=num_days)
    
    # Create data
    data = []
    for date in dates:
        for stock_id in stock_ids:
            row = {
                'date': date,
                'sid': stock_id,
                'luld': np.random.choice([0, 1]),
                'ADV50': np.random.uniform(1000, 10000)
            }
            
            # Add factor columns
            for i in range(num_factors):
                row[str(i)] = np.random.randn()
            
            # Add target columns
            row['intra30m'] = np.random.randn()
            row['nextT1d'] = np.random.randn()
            row['ema1d'] = np.random.randn()
            
            data.append(row)
    
    df = pd.DataFrame(data)
    logger.info(f"Created sample data with {len(df)} rows")
    return df


def _helper_model_creation():
    """Test model creation and basic functionality"""
    logger.info("Testing model creation...")
    
    # Configuration
    config = {
        'num_factors': 100,
        'num_stocks': 100,
        'd_model': 128,
        'num_heads': 4,
        'num_layers': 3,
        'd_ff': 512,
        'dropout': 0.1,
        'max_seq_len': 20,
        'target_columns': ['intra30m', 'nextT1d', 'ema1d']
    }
    
    # Create model
    model = create_model(config)
    logger.info("PASS Model creation passed")
    
    # Test model info
    model_info = model.get_model_info()
    assert model_info['total_parameters'] > 0, "Model has no parameters"
    logger.info(f"PASS Model has {model_info['total_parameters']:,} parameters")
    
    # Test forward pass
    batch_size = 4
    seq_len = 10
    factors = torch.randn(batch_size, seq_len, config['num_factors'])
    stock_ids = torch.randint(0, config['num_stocks'], (batch_size, seq_len))
    
    model.eval()
    with torch.no_grad():
        predictions = model(factors, stock_ids)
    
    # Check predictions
    for target in config['target_columns']:
        assert target in predictions, f"Missing prediction for {target}"
        assert predictions[target].shape == (batch_size,), f"Wrong shape for {target}"
    
    logger.info("PASS Model forward pass passed")
    
    # Test loss computation
    targets = {
        'intra30m': torch.randn(batch_size),
        'nextT1d': torch.randn(batch_size),
        'ema1d': torch.randn(batch_size)
    }
    
    losses = model.compute_loss(predictions, targets)
    assert 'total' in losses, "Missing total loss"
    assert losses['total'].item() > 0, "Loss should be positive"
    logger.info("PASS Loss computation passed")
    
    logger.info("Model creation test completed successfully!")
    return model, config


def _helper_training_components():
    """Test training components"""
    logger.info("Testing training components...")
    
    # Create model and config
    model, config = _helper_model_creation()
    
    # Add training config
    config.update({
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'optimizer': 'adamw',
        'scheduler_type': 'cosine',
        'warmup_steps': 10,
        'total_steps': 100,
        'early_stopping_patience': 5,
        'gradient_clip': 1.0,
        'gradient_accumulation_steps': 1,
        'use_mixed_precision': False,
        'device': 'cpu',
        'output_dir': 'test_outputs',
        'checkpoint_dir': 'test_checkpoints',
        'log_dir': 'test_logs'
    })
    
    # Create trainer
    trainer = create_trainer(model, config)
    logger.info("PASS Trainer creation passed")
    
    # Test correlation loss
    loss_fn = CorrelationLoss(
        correlation_weight=1.0,
        mse_weight=0.1,
        rank_weight=0.1,
        target_correlations=[0.1, 0.05, 0.08]
    )
    
    batch_size = 4
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
    
    loss = loss_fn(predictions, targets)
    assert loss.item() > 0, "Correlation loss should be positive"
    logger.info("PASS Correlation loss passed")
    
    logger.info("Training components test completed successfully!")
    return trainer, config


def test_model_save_load():
    """Test model saving and loading"""
    logger.info("Testing model save/load...")
    
    # Create model
    model, config = _helper_model_creation()
    
    # Save model
    save_path = "outputs/integration_test_model.pth"
    save_model(model, save_path)
    logger.info("PASS Model saved")
    
    # Load model
    loaded_model = load_model(save_path)
    logger.info("PASS Model loaded")
    
    # Test that loaded model produces same output
    batch_size = 4
    seq_len = 10
    factors = torch.randn(batch_size, seq_len, config['num_factors'])
    stock_ids = torch.randint(0, config['num_stocks'], (batch_size, seq_len))
    
    model.eval()
    loaded_model.eval()
    
    with torch.no_grad():
        original_predictions = model(factors, stock_ids)
        loaded_predictions = loaded_model(factors, stock_ids)
    
    # Compare predictions
    for target in config['target_columns']:
        assert torch.allclose(original_predictions[target], loaded_predictions[target], atol=1e-6), f"Predictions don't match for {target}"
    
    logger.info("PASS Model save/load test passed")
    
    # Clean up
    os.remove(save_path)
    
    logger.info("Model save/load test completed successfully!")


def test_simple_training():
    """Test simple training loop"""
    logger.info("Testing simple training loop...")
    
    # Create model and config
    model, config = _helper_model_creation()
    
    # Add training config
    config.update({
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'optimizer': 'adamw',
        'scheduler_type': 'cosine',
        'warmup_steps': 5,
        'total_steps': 50,
        'early_stopping_patience': 3,
        'gradient_clip': 1.0,
        'gradient_accumulation_steps': 1,
        'use_mixed_precision': False,
        'device': 'cpu',
        'output_dir': 'test_outputs',
        'checkpoint_dir': 'test_checkpoints',
        'log_dir': 'test_logs'
    })
    
    trainer = create_trainer(model, config)
    
    # Create synthetic training data
    batch_size = 4
    seq_len = 10
    num_batches = 5
    
    logger.info("Running simple training loop...")
    model.train()
    
    for epoch in range(3):
        total_loss = 0
        
        for batch_idx in range(num_batches):
            # Create synthetic batch
            device = model.device if hasattr(model, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            factors = torch.randn(batch_size, seq_len, config['num_factors']).to(device)
            stock_ids = torch.randint(0, config['num_stocks'], (batch_size, seq_len)).to(device)
            targets = {
                'intra30m': torch.randn(batch_size).to(device),
                'nextT1d': torch.randn(batch_size).to(device),
                'ema1d': torch.randn(batch_size).to(device)
            }
            
            # Forward pass
            predictions = model(factors, stock_ids)
            losses = model.compute_loss(predictions, targets)
            
            # Backward pass
            trainer.optimizer.zero_grad()
            losses['total'].backward()
            trainer.optimizer.step()
            
            total_loss += losses['total'].item()
        
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch+1}, Average Loss: {avg_loss:.6f}")
    
    logger.info("PASS Simple training test completed successfully!")


def test_correlation_optimization():
    """Test correlation optimization specifically"""
    logger.info("Testing correlation optimization...")
    
    # Create model
    model, config = _helper_model_creation()
    
    # Test correlation loss with different weights
    loss_fn = CorrelationLoss(
        correlation_weight=1.0,
        mse_weight=0.1,
        rank_weight=0.1,
        target_correlations=[0.1, 0.05, 0.08]
    )
    
    batch_size = 8
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
    
    # Test different loss configurations
    loss_configs = [
        {'correlation_weight': 1.0, 'mse_weight': 0.1, 'rank_weight': 0.1},
        {'correlation_weight': 0.5, 'mse_weight': 0.5, 'rank_weight': 0.1},
        {'correlation_weight': 0.1, 'mse_weight': 1.0, 'rank_weight': 0.1}
    ]
    
    for i, loss_config in enumerate(loss_configs):
        loss_fn = CorrelationLoss(**loss_config)
        loss = loss_fn(predictions, targets)
        logger.info(f"Loss config {i+1}: {loss.item():.6f}")
        assert loss.item() > 0, "Loss should be positive"
    
    logger.info("PASS Correlation optimization test completed successfully!")


def main():
    """Run all integration tests"""
    logger.info("Starting simplified integration tests...")
    start_time = time.time()
    
    try:
        # Test model creation
        _helper_model_creation()
        
        # Test training components
        _helper_training_components()
        
        # Test model save/load
        test_model_save_load()
        
        # Test simple training
        test_simple_training()
        
        # Test correlation optimization
        test_correlation_optimization()
        
        elapsed_time = time.time() - start_time
        logger.info(f"All integration tests passed! Total time: {elapsed_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        raise


if __name__ == "__main__":
    main() 