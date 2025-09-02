#!/usr/bin/env python3
"""
Test script for TCN (Temporal Convolutional Network) model
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.models import FactorForecastingTCNModel, FactorTCN, TemporalConvNet, TemporalBlock
from configs.config import ModelConfig

def test_tcn_components():
    """Test individual TCN components"""
    print("Testing TCN components...")
    
    # Test TemporalBlock
    batch_size, seq_len, in_channels, out_channels = 4, 10, 64, 128
    kernel_size, dilation = 3, 1
    
    temporal_block = TemporalBlock(in_channels, out_channels, kernel_size, 
                                  stride=1, dilation=dilation, padding=(kernel_size-1) * dilation)
    
    x = torch.randn(batch_size, in_channels, seq_len)
    output = temporal_block(x)
    
    print(f"TemporalBlock input shape: {x.shape}")
    print(f"TemporalBlock output shape: {output.shape}")
    assert output.shape == (batch_size, out_channels, seq_len), "TemporalBlock output shape mismatch"
    print("TemporalBlock test passed")
    
    # Test TemporalConvNet
    num_inputs, num_channels = 64, [128, 256, 512]
    tcn = TemporalConvNet(num_inputs, num_channels, kernel_size=3, dropout=0.1)
    
    x = torch.randn(batch_size, seq_len, num_inputs)
    output = tcn(x)
    
    print(f"TemporalConvNet input shape: {x.shape}")
    print(f"TemporalConvNet output shape: {output.shape}")
    assert output.shape == (batch_size, seq_len, num_channels[-1]), "TemporalConvNet output shape mismatch"
    print("TemporalConvNet test passed")

def test_factor_tcn():
    """Test FactorTCN model"""
    print("\nTesting FactorTCN model...")
    
    # Create configuration
    config = {
        'input_dim': 100,
        'hidden_dim': 256,
        'num_layers': 4,
        'dropout': 0.1,
        'num_stocks': 1000,
        'sequence_length': 10,
        'target_columns': ['intra30m', 'nextT1d', 'ema1d'],
        'kernel_size': 3
    }
    
    # Create model
    model = FactorTCN(config)
    
    # Create sample input
    batch_size, seq_len, feature_dim = 8, 10, 100
    features = torch.randn(batch_size, seq_len, feature_dim)
    stock_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Forward pass
    with torch.no_grad():
        predictions = model(features, stock_ids)
    
    print(f"Input features shape: {features.shape}")
    print(f"Stock IDs shape: {stock_ids.shape}")
    print(f"Predictions shape: {predictions['intra30m'].shape}")
    
    # Check output shapes
    for target_name in config['target_columns']:
        assert predictions[target_name].shape == (batch_size, seq_len), f"Output shape mismatch for {target_name}"
        print(f"{target_name} prediction shape: {predictions[target_name].shape}")
    
    print("FactorTCN test passed")

def test_factor_forecasting_tcn_model():
    """Test FactorForecastingTCNModel"""
    print("\nTesting FactorForecastingTCNModel...")
    
    # Create configuration
    config = {
        'input_dim': 100,
        'hidden_dim': 256,
        'num_layers': 4,
        'dropout': 0.1,
        'num_stocks': 1000,
        'sequence_length': 10,
        'target_columns': ['intra30m', 'nextT1d', 'ema1d'],
        'kernel_size': 3
    }
    
    # Create model
    model = FactorForecastingTCNModel(config)
    
    # Create sample input
    batch_size, seq_len, feature_dim = 8, 10, 100
    features = torch.randn(batch_size, seq_len, feature_dim)
    stock_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Create sample targets
    targets = {}
    for target_name in config['target_columns']:
        targets[target_name] = torch.randn(batch_size, seq_len)
    
    # Forward pass
    with torch.no_grad():
        predictions = model(features, stock_ids)
    
    # Compute loss
    losses = model.compute_loss(predictions, targets)
    
    print(f"Model predictions: {list(predictions.keys())}")
    print(f"Losses: {list(losses.keys())}")
    print(f"Total loss: {losses['total_loss'].item():.6f}")
    
    # Get model info
    model_info = model.get_model_info()
    print(f"Model type: {model_info['model_type']}")
    print(f"Total parameters: {model_info['total_parameters']:,}")
    print(f"Trainable parameters: {model_info['trainable_parameters']:,}")
    
    print("FactorForecastingTCNModel test passed")

def test_model_creation():
    """Test model creation function"""
    print("\nTesting model creation function...")
    
    from src.models.models import create_model
    
    # Test TCN model creation
    tcn_config = {
        'model_type': 'tcn',
        'input_dim': 100,
        'hidden_dim': 256,
        'num_layers': 4,
        'dropout': 0.1,
        'num_stocks': 1000,
        'sequence_length': 10,
        'target_columns': ['intra30m', 'nextT1d', 'ema1d'],
        'kernel_size': 3
    }
    
    tcn_model = create_model(tcn_config)
    print(f"Created TCN model: {type(tcn_model).__name__}")
    
    # Test Transformer model creation
    transformer_config = {
        'model_type': 'transformer',
        'input_dim': 100,
        'hidden_dim': 256,
        'num_layers': 4,
        'num_heads': 8,
        'dropout': 0.1,
        'num_stocks': 1000,
        'sequence_length': 10,
        'target_columns': ['intra30m', 'nextT1d', 'ema1d']
    }
    
    transformer_model = create_model(transformer_config)
    print(f"Created Transformer model: {type(transformer_model).__name__}")
    
    print("Model creation test passed")

def test_tcn_vs_transformer():
    """Compare TCN and Transformer models"""
    print("\nComparing TCN vs Transformer models...")
    
    # Common configuration
    common_config = {
        'input_dim': 100,
        'hidden_dim': 256,
        'num_layers': 4,
        'dropout': 0.1,
        'num_stocks': 1000,
        'sequence_length': 10,
        'target_columns': ['intra30m', 'nextT1d', 'ema1d']
    }
    
    # TCN configuration
    tcn_config = common_config.copy()
    tcn_config['model_type'] = 'tcn'
    tcn_config['kernel_size'] = 3
    
    # Transformer configuration
    transformer_config = common_config.copy()
    transformer_config['model_type'] = 'transformer'
    transformer_config['num_heads'] = 8
    
    # Create models
    from src.models.models import create_model
    
    tcn_model = create_model(tcn_config)
    transformer_model = create_model(transformer_config)
    
    # Create sample input
    batch_size, seq_len, feature_dim = 8, 10, 100
    features = torch.randn(batch_size, seq_len, feature_dim)
    stock_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Test forward pass
    with torch.no_grad():
        tcn_predictions = tcn_model(features, stock_ids)
        transformer_predictions = transformer_model(features, stock_ids)
    
    # Compare outputs
    print("Model Comparison:")
    print(f"{'Metric':<20} {'TCN':<15} {'Transformer':<15}")
    print("-" * 50)
    
    tcn_info = tcn_model.get_model_info()
    transformer_info = transformer_model.get_model_info()
    
    print(f"{'Model Type':<20} {tcn_info['model_type']:<15} {transformer_info['model_type']:<15}")
    print(f"{'Total Parameters':<20} {tcn_info['total_parameters']:<15,} {transformer_info['total_parameters']:<15,}")
    print(f"{'Trainable Parameters':<20} {tcn_info['trainable_parameters']:<15,} {transformer_info['trainable_parameters']:<15,}")
    
    # Compare prediction shapes
    for target_name in common_config['target_columns']:
        tcn_shape = tcn_predictions[target_name].shape
        transformer_shape = transformer_predictions[target_name].shape
        print(f"{target_name + ' shape':<20} {str(tcn_shape):<15} {str(transformer_shape):<15}")
    
    print("TCN vs Transformer comparison completed")

def main():
    """Main test function"""
    print(" Testing TCN (Temporal Convolutional Network) Model")
    print("=" * 60)
    
    try:
        # Test individual components
        test_tcn_components()
        
        # Test FactorTCN
        test_factor_tcn()
        
        # Test FactorForecastingTCNModel
        test_factor_forecasting_tcn_model()
        
        # Test model creation
        test_model_creation()
        
        # Compare models
        test_tcn_vs_transformer()
        
        print("\nAll TCN model tests passed successfully!")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 