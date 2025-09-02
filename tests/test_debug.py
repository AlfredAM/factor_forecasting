#!/usr/bin/env python3
"""
Debug script to test model saving and loading
"""
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.model_factory import create_model, save_model, load_model

def test_model_save_load():
    """Test model saving and loading with detailed debugging"""
    
    # Configuration
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
    
    print("Creating model...")
    model = create_model(config)
    
    # Test input
    batch_size = 4
    seq_len = 10
    factors = torch.randn(batch_size, seq_len, config['num_factors'])
    stock_ids = torch.randint(0, config['num_stocks'], (batch_size, seq_len))
    
    print("Getting original predictions...")
    model.eval()
    with torch.no_grad():
        original_predictions = model(factors, stock_ids)
    
    print("Original predictions:")
    for target, pred in original_predictions.items():
        if target != 'correlation_weights':
            print(f"  {target}: {pred[:3]}")  # Show first 3 values
    
    # Save model
    save_path = "outputs/debug_model.pth"
    print(f"Saving model to {save_path}...")
    save_model(model, save_path)
    
    # Load model
    print("Loading model...")
    loaded_model = load_model(save_path)
    
    print("Getting loaded predictions...")
    loaded_model.eval()
    with torch.no_grad():
        loaded_predictions = loaded_model(factors, stock_ids)
    
    print("Loaded predictions:")
    for target, pred in loaded_predictions.items():
        if target != 'correlation_weights':
            print(f"  {target}: {pred[:3]}")  # Show first 3 values
    
    # Compare predictions
    print("\nComparing predictions:")
    for target in ['intra30m', 'nextT1d', 'ema1d']:
        orig = original_predictions[target]
        loaded = loaded_predictions[target]
        
        diff = torch.abs(orig - loaded)
        max_diff = torch.max(diff)
        mean_diff = torch.mean(diff)
        
        print(f"  {target}:")
        print(f"    Max difference: {max_diff:.8f}")
        print(f"    Mean difference: {mean_diff:.8f}")
        print(f"    All close (1e-6): {torch.allclose(orig, loaded, atol=1e-6)}")
        print(f"    All close (1e-5): {torch.allclose(orig, loaded, atol=1e-5)}")
        print(f"    All close (1e-4): {torch.allclose(orig, loaded, atol=1e-4)}")
    
    # Clean up
    os.remove(save_path)
    print("\nTest completed!")

if __name__ == "__main__":
    test_model_save_load() 