#!/usr/bin/env python3
"""
Integration test for the complete factor forecasting system
Tests the entire pipeline from data processing to model training and inference
"""
import torch
import pandas as pd
import numpy as np
import sys
import os
import time
import logging
import tempfile

import types

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from configs.config import config
from src.data_processing.data_processor import MultiFileDataProcessor, DataManager, create_training_dataloaders
from src.models.models import create_model, FactorForecastingModel
from src.training.trainer import FactorForecastingTrainer
from src.data_processing.data_pipeline import create_continuous_data_loaders
from src.training.trainer import CorrelationLoss

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_data(num_stocks=20, num_days=10, num_factors=100):
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
            for i in range(100):
                row[f'factor_{i}'] = np.random.randn()
            # Add target columns
            target_columns = ['intra30m', 'nextT1d', 'ema1d']
            for t in target_columns:
                row[t] = np.random.randn()
            data.append(row)
    
    df = pd.DataFrame(data)
    logger.info(f"Created sample data with {len(df)} rows and {len([c for c in df.columns if c.startswith('factor_')])} factor columns")
    return df


def test_data_processing():
    """Test data processing pipeline"""
    logger.info("Testing data processing pipeline...")
    
    # Create sample data
    data = create_sample_data(num_stocks=50, num_days=20, num_factors=100)

    # Create temporary directory and parquet files for testing
    import tempfile
    import os
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create some mock parquet files
        for i in range(10):  # Add more dates
            # Use YYYYMMDD format as expected by DailyDataLoader
            date_str = f"202401{i+1:02d}"
            file_path = Path(temp_dir) / f"{date_str}.parquet"
            
            # Create mock data for this date
            mock_data = create_sample_data(num_stocks=20, num_days=1, num_factors=100)
            mock_data.to_parquet(file_path, index=False)
        
                    # mock config, using temporary directory
            class ModelConfig:
                limit_up_down_column = 'luld'
                weight_column = 'ADV50'
                factor_columns = [f'factor_{i}' for i in range(100)]
                target_columns = ['intra30m', 'nextT1d', 'ema1d']
                data_dir = temp_dir
                data_path = temp_dir
                sequence_length = 3  # Use smaller sequence length
                prediction_horizon = 1
                batch_size = 2
                num_workers = 0
                train_ratio = 0.7
                val_ratio = 0.15
                test_ratio = 0.15
        config = ModelConfig()
    
    # Test data processor
        processor = MultiFileDataProcessor(config)
    
    # Clean data
        cleaned_data = processor._clean_data(data)
        assert len(cleaned_data) > 0, "Data cleaning failed"
        logger.info("PASS Data cleaning passed")
    
    # Normalize data
    normalized_data = processor._preprocess_data(cleaned_data)
    assert len(normalized_data) > 0, "Data normalization failed"
    logger.info("PASS Data normalization passed")
    
    # Test data pipeline
    try:
        # Create continuous data loaders
        train_loader, val_loader, test_loader = create_continuous_data_loaders(config)
        
        # Test the data loaders
        assert len(train_loader) > 0, "Train loader is empty"
        logger.info(f"PASS Train loader has {len(train_loader)} batches")

        # Test getting a batch from train loader
        for batch in train_loader:
            assert 'factors' in batch, "Batch missing factors"
            assert 'targets' in batch, "Batch missing targets"
            assert 'stock_ids' in batch, "Batch missing stock_ids"
            logger.info(f"PASS Batch shape: factors={batch['factors'].shape}, targets={batch['targets']['intra30m'].shape}")
            break  # Just test the first batch
            
    except Exception as e:
        logger.error(f"Failed to create continuous data loaders: {e}")
        raise
    
    logger.info("Data processing pipeline test completed successfully!")


def _helper_model_creation():
    """Helper function for model creation and basic functionality"""
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
    """Helper function for training components"""
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
    trainer = FactorForecastingTrainer(model, config)
    logger.info("PASS Trainer creation passed")
    
    # Test correlation loss
    loss_fn = trainer.loss_fn if hasattr(trainer, 'loss_fn') and trainer.loss_fn is not None else CorrelationLoss(
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
    # Assuming save_model and load_model are defined elsewhere or need to be imported
    # For now, we'll just save and load the model object directly
    torch.save(model.state_dict(), save_path)
    logger.info("PASS Model saved")
    
    # Load model
    loaded_model = FactorForecastingModel(config)
    loaded_model.load_state_dict(torch.load(save_path))
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


def test_end_to_end():
    """Test end-to-end pipeline"""
    logger.info("Testing end-to-end pipeline...")
    
    # Create sample data
    data = create_sample_data(num_stocks=20, num_days=10, num_factors=100)
    
    # Write temporary parquet file
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
        data.to_parquet(tmp.name)
        tmp_path = tmp.name
    
    # Configure data_path to point to parquet file
    config = types.SimpleNamespace(
        num_factors=100,
        num_stocks=20,
        d_model=64,
        num_heads=2,
        num_layers=2,
        d_ff=256,
        dropout=0.1,
        max_seq_len=10,
        target_columns=['intra30m', 'nextT1d', 'ema1d'],
        learning_rate=1e-3,
        weight_decay=1e-5,
        optimizer='adamw',
        scheduler_type='cosine',
        warmup_steps=5,
        total_steps=50,
        early_stopping_patience=3,
        gradient_clip=1.0,
        gradient_accumulation_steps=1,
        use_mixed_precision=False,
        device='cpu',
        output_dir='test_outputs',
        checkpoint_dir='test_checkpoints',
        log_dir='test_logs',
        data_path=tmp_path,
        sequence_length=3,
        batch_size=2,
        num_workers=0,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    # Get batch through DataLoader
    train_loader, val_loader, test_loader = create_continuous_data_loaders(config)
    model = create_model(config)
    trainer = FactorForecastingTrainer(model, config)
    
    logger.info("Running simple training loop...")
    model.train()
    for epoch in range(3):
        total_loss = 0
        num_batches = 0
        for batch in train_loader:
            factors = batch['factors']
            stock_ids = batch['stock_ids']
            targets = {k: batch['targets'][k] for k in batch['targets']}
            
            predictions = model(factors, stock_ids)
            losses = model.compute_loss(predictions, targets)
            trainer.optimizer.zero_grad()
            losses['total'].backward()
            trainer.optimizer.step()
            total_loss += losses['total'].item()
            num_batches += 1
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        logger.info(f"Epoch {epoch+1}, Average Loss: {avg_loss:.6f}")
    # Clean up temporary file
    os.remove(tmp_path)
    
    logger.info("End-to-end pipeline test completed successfully!")


def main():
    """Run all integration tests"""
    logger.info("Starting integration tests...")
    start_time = time.time()
    
    try:
        # Test data processing
        test_data_processing()
        
        # Test model creation
        _helper_model_creation()
        
        # Test training components
        _helper_training_components()
        
        # Test model save/load
        test_model_save_load()
        
        # Test end-to-end pipeline
        test_end_to_end()
        
        elapsed_time = time.time() - start_time
        logger.info(f"All integration tests passed! Total time: {elapsed_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        raise


if __name__ == "__main__":
    main() 