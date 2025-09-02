#!/usr/bin/env python3
"""
Test script for rolling window training with three targets
"""

import os
import sys
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import ModelConfig
from src.training.rolling_train import RollingWindowTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_rolling_training():
    """Test rolling window training with synthetic data"""
    
    # Create configuration for testing
    config = ModelConfig()
    
    # Test configuration
    config.data_dir = "data/test_data"  # Use test data directory
    config.start_date = "2018-01-01"
    config.end_date = "2020-12-31"
    config.batch_size = 32
    config.num_epochs = 5  # Small number for testing
    config.learning_rate = 1e-4
    config.d_model = 128  # Smaller model for testing
    config.num_layers = 2
    config.num_heads = 4
    
    # Loss function configuration
    config.correlation_weight = 1.0
    config.mse_weight = 0.1
    config.rank_weight = 0.1
    config.target_correlations = [0.1, 0.05, 0.08]
    
    # Training configuration
    config.min_train_years = 1
    config.prediction_years = [2019, 2020]  # Only test 2 years
    
    # Sequence configuration
    config.sequence_length = 10
    config.min_sequence_length = 5
    
    # Set training mode
    config.training_mode = "rolling_window"
    config.loss_function = "correlation_loss"
    
    # Log configuration
    logger.info("="*80)
    logger.info("TESTING ROLLING WINDOW TRAINING")
    logger.info("="*80)
    logger.info(f"Data directory: {config.data_dir}")
    logger.info(f"Date range: {config.start_date} to {config.end_date}")
    logger.info(f"Target columns: {config.target_columns}")
    logger.info(f"Target correlations: {config.target_correlations}")
    logger.info(f"Prediction years: {config.prediction_years}")
    logger.info(f"Model architecture: d_model={config.d_model}, layers={config.num_layers}, heads={config.num_heads}")
    logger.info(f"Training: batch_size={config.batch_size}, epochs={config.num_epochs}, lr={config.learning_rate}")
    logger.info("="*80)
    
    try:
        # Create trainer and run
        trainer = RollingWindowTrainer(config)
        results = trainer.run_rolling_window_training()
        
        # Print results
        logger.info("\n" + "="*80)
        logger.info("TEST RESULTS")
        logger.info("="*80)
        
        for test_year, result in results.items():
            if 'correlations' in result:
                correlations = result['correlations']
                logger.info(f"\nYear {test_year} (trained on {result['train_years']}):")
                for key, value in correlations.items():
                    if 'correlation' in key:
                        logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"\nYear {test_year}: ERROR - {result.get('error', 'Unknown error')}")
        
        logger.info("Test completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rolling_training() 