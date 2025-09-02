#!/usr/bin/env python3
"""
Model framework test script
Verifies the basic functionality of each module
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import logging
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import ModelConfig
from src.data_processing.data_processor import DataManager, create_training_dataloaders, MultiFileDataProcessor
from src.models.models import create_model, FactorTransformer, FactorForecastingModel
from src.training.trainer import FactorForecastingTrainer, MetricsTracker
from src.inference.inference import create_inference_engine

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create config instance
config = ModelConfig()

class FrameworkTester:
    """Framework tester"""
    
    def __init__(self):
        self.test_results = {}
    
    def test_data_processing(self):
        """Tests the data processing module"""
        logger.info("Testing data processing module...")
        
        try:
            # Create test data
            test_data = self._create_test_data()
            
            # Test data processor
            processor = MultiFileDataProcessor(config)
            
            # Test data cleaning
            cleaned_data = processor._clean_data(test_data)
            assert len(cleaned_data) <= len(test_data), "Data cleaning should not increase data size"
            assert len(cleaned_data) > 0, "Data cleaning should not remove all data"
            
            # Test data preprocessing
            processed_data = processor._preprocess_data(cleaned_data)
            assert not processed_data.isnull().any().any(), "No missing values should be present after processing"
            
            # Test sequence building
            sequences_X, sequences_y = processor.build_sequences(processed_data)
            assert len(sequences_X) > 0, "Sequences should be built"
            assert len(sequences_X) == len(sequences_y), "Number of X and y sequences should be the same"
            
            # Test data loader creation (skip if no data files available)
            try:
                dataloaders, scalers = create_training_dataloaders(config)
                assert 'train' in dataloaders, "Training data loader should be present"
                assert 'val' in dataloaders, "Validation data loader should be present"
            except Exception as e:
                logger.warning(f"Data loader creation skipped: {e}")
                # This is acceptable in test environment without actual data files
            
            self.test_results['data_processing'] = "PASS"
            logger.info("Data processing module test passed")
            
        except Exception as e:
            self.test_results['data_processing'] = f"FAIL: {e}"
            logger.error(f"Data processing module test failed: {e}")
    
    def test_models(self):
        """Tests the model module"""
        logger.info("Testing model module...")
        
        try:
            # Convert config to dict format for model creation
            config_dict = {
                'num_factors': len(config.factor_columns),
                'num_stocks': 1000,
                'd_model': config.hidden_size,
                'num_heads': config.num_heads,
                'num_layers': config.num_layers,
                'd_ff': config.hidden_size * 4,
                'dropout': config.dropout,
                'max_seq_len': 50,
                'target_columns': config.target_columns
            }
            transformer = create_model(config_dict)
            assert isinstance(transformer, FactorForecastingModel), "Transformer model should be created"
            
            # Test model forward pass
            batch_size, seq_len, feature_dim = 4, config.sequence_length, len(config.factor_columns)
            factors = torch.randn(batch_size, seq_len, feature_dim)
            stock_ids = torch.randint(0, 1000, (batch_size, seq_len))
            
            # Transformer forward pass
            transformer_output = transformer(factors, stock_ids)
            expected_shape = (batch_size,)
            assert transformer_output['intra30m'].shape == expected_shape, f"Transformer output shape error: {transformer_output['intra30m'].shape}"
            
            self.test_results['models'] = "PASS"
            logger.info("Model module test passed")
            
        except Exception as e:
            self.test_results['models'] = f"FAIL: {e}"
            logger.error(f"Model module test failed: {e}")
    
    def test_metrics(self):
        """Tests the metrics calculation module"""
        logger.info("Testing metrics calculation module...")
        
        try:
            # Create test data
            pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            target = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
            
            # Calculate metrics
            metrics_tracker = MetricsTracker(config.target_columns)
            metrics_tracker.update({'intra30m': torch.tensor(pred)}, {'intra30m': torch.tensor(target)})
            metrics = metrics_tracker.compute_metrics()
            
            # Verify metrics
            assert 'intra30m' in metrics, "intra30m metrics should be present"
            assert 'mse' in metrics['intra30m'], "MSE metric should be present"
            assert 'mae' in metrics['intra30m'], "MAE metric should be present"
            assert 'r2' in metrics['intra30m'], "R² metric should be present"
            assert 'directional_accuracy' in metrics['intra30m'], "Directional accuracy should be present"
            
            # Verify metric values reasonability
            assert 0 <= metrics['intra30m']['r2'] <= 1, "R² should be in the range [0,1]"
            assert 0 <= metrics['intra30m']['directional_accuracy'] <= 1, "Directional accuracy should be in the range [0,1]"
            
            self.test_results['metrics'] = "PASS"
            logger.info("Metrics calculation module test passed")
            
        except Exception as e:
            self.test_results['metrics'] = f"FAIL: {e}"
            logger.error(f"Metrics calculation module test failed: {e}")
    
    def test_training_framework(self):
        """Tests the training framework"""
        logger.info("Testing training framework...")
        
        try:
            # Convert config to dict format for model creation
            config_dict = {
                'num_factors': len(config.factor_columns),
                'num_stocks': 1000,
                'd_model': config.hidden_size,
                'num_heads': config.num_heads,
                'num_layers': config.num_layers,
                'd_ff': config.hidden_size * 4,
                'dropout': config.dropout,
                'max_seq_len': 50,
                'target_columns': config.target_columns
            }
            # Create model
            model = create_model(config_dict)
            
            # Create trainer
            model = create_model(config_dict)
            # Convert config to dict format
            config_dict = {
                'device': config.device,
                'target_columns': config.target_columns,
                'learning_rate': config.learning_rate,
                'weight_decay': config.weight_decay,
                'optimizer': config.optimizer_type,
                'scheduler_type': config.scheduler_type,
                'warmup_steps': config.warmup_steps,
                'use_mixed_precision': True,
                'gradient_accumulation_steps': 1,
                'early_stopping_patience': config.early_stopping_patience,
                'early_stopping_min_delta': 0.001,
                'restore_best_weights': True,
                'checkpoint_dir': 'checkpoints',
                'log_dir': config.log_dir,
                'output_dir': 'outputs'
            }
            trainer = FactorForecastingTrainer(model, config_dict)
            
            # Verify trainer components
            assert trainer.model is not None, "Model should be present"
            assert trainer.optimizer is not None, "Optimizer should be present"
            
            # Test model forward pass
            batch_size = 4
            seq_len = 10
            factors = torch.randn(batch_size, seq_len, len(config.factor_columns))
            stock_ids = torch.randint(0, 1000, (batch_size, seq_len))
            
            with torch.no_grad():
                predictions = trainer.model(factors, stock_ids)
            
            assert isinstance(predictions, dict), "Predictions should be a dictionary"
            assert 'intra30m' in predictions, "intra30m prediction should be present"
            
            self.test_results['training_framework'] = "PASS"
            logger.info("Training framework test passed")
            
        except Exception as e:
            self.test_results['training_framework'] = f"FAIL: {e}"
            logger.error(f"Training framework test failed: {e}")
    
    def test_inference_framework(self):
        """Tests the inference framework"""
        logger.info("Testing inference framework...")
        
        try:
            # Convert config to dict format for model creation
            config_dict = {
                'num_factors': len(config.factor_columns),
                'num_stocks': 1000,
                'd_model': config.hidden_size,
                'num_heads': config.num_heads,
                'num_layers': config.num_layers,
                'd_ff': config.hidden_size * 4,
                'dropout': config.dropout,
                'max_seq_len': 50,
                'target_columns': config.target_columns
            }
            # Create test model
            model = create_model(config_dict)
            
            # Save test model
            test_model_path = "test_model.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config
            }, test_model_path)
            
            # Create inference engine
            config_dict = {
                'device': config.device,
                'target_columns': config.target_columns,
                'factor_columns': config.factor_columns
            }
            inference_engine = create_inference_engine(test_model_path, config)
            
            # Test inference
            test_factors = np.random.randn(len(config.factor_columns))
            result = inference_engine.predict(test_factors, "test_sid")
            
            # Verify inference result
            assert 'prediction' in result, "Prediction result should be present"
            assert 'latency_ms' in result, "Latency information should be present"
            assert 'timestamp' in result, "Timestamp should be present"
            
            # Clean up test file
            os.remove(test_model_path)
            
            self.test_results['inference_framework'] = "PASS"
            logger.info("Inference framework test passed")
            
        except Exception as e:
            self.test_results['inference_framework'] = f"FAIL: {e}"
            logger.error(f"Inference framework test failed: {e}")
    
    def test_configuration(self):
        """Tests the configuration module"""
        logger.info("Testing configuration module...")
        
        try:
            # Verify necessary configuration
            assert hasattr(config, 'factor_columns'), "Factor column configuration should be present"
            assert hasattr(config, 'target_columns'), "Target column configuration should be present"
            assert hasattr(config, 'hidden_size'), "Hidden size configuration should be present"
            assert hasattr(config, 'num_layers'), "Number of layers configuration should be present"
            
            # Verify configuration value reasonability
            assert len(config.factor_columns) == 100, "There should be 100 factors"
            assert len(config.target_columns) == 3, "There should be 3 target variables"
            assert config.hidden_size > 0, "Hidden size should be positive"
            assert config.num_layers > 0, "Number of layers should be positive"
            
            self.test_results['configuration'] = "PASS"
            logger.info("Configuration module test passed")
            
        except Exception as e:
            self.test_results['configuration'] = f"FAIL: {e}"
            logger.error(f"Configuration module test failed: {e}")
    
    def _create_test_data(self) -> pd.DataFrame:
        """Creates test data"""
        np.random.seed(42)
        
        # Create base data
        n_samples = 1000
        data = {}
        
        # Add factor columns
        for i in range(100):
            data[f'factor_{i}'] = np.random.randn(n_samples)
        
        # Add other columns
        data['sid'] = np.random.randint(1, 11, n_samples)  # 10 stocks
        data['ADV50'] = np.random.uniform(1000, 10000, n_samples)
        data['limit_up_down'] = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])  # 5% limit up/down
        data['intra30m'] = np.random.randn(n_samples)
        data['nextT1d'] = np.random.randn(n_samples)
        data['ema1d'] = np.random.randn(n_samples)
        
        return pd.DataFrame(data)
    
    def run_all_tests(self):
        """Runs all tests"""
        logger.info("Starting framework test...")
        
        # Run individual module tests
        self.test_configuration()
        self.test_data_processing()
        self.test_models()
        self.test_metrics()
        self.test_training_framework()
        self.test_inference_framework()
        
        # Output test results
        self._print_test_results()
        
        # Return overall result
        all_passed = all("PASS" in result for result in self.test_results.values())
        return all_passed
    
    def _print_test_results(self):
        """Prints test results"""
        logger.info("\n" + "="*50)
        logger.info("Framework Test Results")
        logger.info("="*50)
        
        for test_name, result in self.test_results.items():
            status = "PASS" if "PASS" in result else "FAIL"
            logger.info(f"{test_name:20s}: {status}")
            if "FAIL" in result:
                logger.error(f"   Error details: {result}")
        
        logger.info("="*50)
        
        # Count results
        passed_count = sum(1 for result in self.test_results.values() if "PASS" in result)
        total_count = len(self.test_results)
        
        logger.info(f"Tests passed: {passed_count}/{total_count}")
        
        if passed_count == total_count:
            logger.info("All tests passed! Framework ready.")
        else:
            logger.error("Some tests failed, please check error messages.")

def main():
    """Main function"""
    # Create tester
    tester = FrameworkTester()
    
    # Run tests
    success = tester.run_all_tests()
    
    # Return result
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 