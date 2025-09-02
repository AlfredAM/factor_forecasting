#!/usr/bin/env python3
"""
Quick validation script for factor forecasting system
Runs a comprehensive check of all core functionality
"""
import torch
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.models import create_model
from src.training.trainer import FactorForecastingTrainer, CorrelationLoss

def print_section(title):
    """Print section header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def test_model_creation():
    """Test model creation"""
    print_section("MODEL CREATION TEST")
    
    config = {
        'num_factors': 100,
        'num_stocks': 1000,
        'd_model': 256,
        'num_heads': 8,
        'num_layers': 6,
        'd_ff': 1024,
        'dropout': 0.1,
        'max_seq_len': 50,
        'target_columns': ['intra30m', 'nextT1d', 'ema1d']
    }
    
    model = create_model(config)
    model_info = model.get_model_info()
    
    print(f"Model created successfully")
    print(f"  - Parameters: {model_info['total_parameters']:,}")
    print(f"  - Size: {model_info['model_size_mb']:.2f} MB")
    print(f"  - Targets: {config['target_columns']}")
    
    return model, config

def test_forward_pass(model, config):
    """Test model forward pass"""
    print_section("FORWARD PASS TEST")
    
    batch_size = 4
    seq_len = 10
    device = next(model.parameters()).device
    factors = torch.randn(batch_size, seq_len, config['num_factors']).to(device)
    stock_ids = torch.randint(0, config['num_stocks'], (batch_size, seq_len)).to(device)
    
    model.eval()
    with torch.no_grad():
        predictions = model(factors, stock_ids)
    
    print(f"Forward pass successful")
    for target in config['target_columns']:
        shape = predictions[target].shape
        print(f"  - {target}: {shape}")
    
    return predictions

def test_loss_computation(model, predictions, config):
    """Test loss computation"""
    print_section("LOSS COMPUTATION TEST")
    
    batch_size = 4
    device = next(model.parameters()).device
    targets = {
        'intra30m': torch.randn(batch_size).to(device),
        'nextT1d': torch.randn(batch_size).to(device),
        'ema1d': torch.randn(batch_size).to(device)
    }
    
    losses = model.compute_loss(predictions, targets)
    
    print(f"Loss computation successful")
    for target, loss in losses.items():
        print(f"  - {target}: {loss.item():.6f}")
    
    return losses

def test_correlation_loss():
    """Test correlation loss function"""
    print_section("CORRELATION LOSS TEST")
    
    loss_fn = CorrelationLoss(
        correlation_weight=1.0,
        mse_weight=0.1,
        rank_weight=0.1,
        target_correlations=[0.1, 0.05, 0.08]
    )
    
    batch_size = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictions = {
        'intra30m': torch.randn(batch_size).to(device),
        'nextT1d': torch.randn(batch_size).to(device),
        'ema1d': torch.randn(batch_size).to(device)
    }
    
    targets = {
        'intra30m': torch.randn(batch_size).to(device),
        'nextT1d': torch.randn(batch_size).to(device),
        'ema1d': torch.randn(batch_size).to(device)
    }
    
    loss = loss_fn(predictions, targets)
    print(f"Correlation loss: {loss.item():.6f}")
    
    return loss

def test_training_components(model, config):
    """Test training components"""
    print_section("TRAINING COMPONENTS TEST")
    
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
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'output_dir': 'test_outputs',
        'checkpoint_dir': 'test_checkpoints',
        'log_dir': 'test_logs'
    })
    
    trainer = FactorForecastingTrainer(model, config)
    
    print(f"Trainer created successfully")
    print(f"  - Device: {trainer.device}")
    print(f"  - Distributed: {trainer.is_distributed}")
    print(f"  - Optimizer: {type(trainer.optimizer).__name__}")
    
    return trainer

def test_simple_training(model, config):
    """Test simple training loop"""
    print_section("SIMPLE TRAINING TEST")
    
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
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'output_dir': 'test_outputs',
        'checkpoint_dir': 'test_checkpoints',
        'log_dir': 'test_logs'
    })
    
    trainer = FactorForecastingTrainer(model, config)
    device = next(model.parameters()).device
    # Create synthetic training data
    batch_size = 4
    seq_len = 10
    num_batches = 3
    
    print(f"Running {num_batches} training batches...")
    model.train()
    
    for epoch in range(2):
        total_loss = 0
        
        for batch_idx in range(num_batches):
            # Create synthetic batch
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
        print(f"  Epoch {epoch+1}: Loss = {avg_loss:.6f}")
    
    print(f"Training test completed successfully")

def main():
    """Run all validation tests"""
    print("Factor Forecasting System - Quick Validation")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Test model creation
        model, config = test_model_creation()
        
        # Test forward pass
        predictions = test_forward_pass(model, config)
        
        # Test loss computation
        losses = test_loss_computation(model, predictions, config)
        
        # Test correlation loss
        correlation_loss = test_correlation_loss()
        
        # Test training components
        trainer = test_training_components(model, config)
        
        # Test simple training
        test_simple_training(model, config)
        
        elapsed_time = time.time() - start_time
        
        print_section("VALIDATION SUMMARY")
        print(f"All tests passed successfully!")
        print(f"Total time: {elapsed_time:.2f} seconds")
        print(f"System is ready for production use")
        
    except Exception as e:
        print_section("VALIDATION FAILED")
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main() 