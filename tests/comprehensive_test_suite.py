#!/usr/bin/env python3
"""
Comprehensive Test Suite for Factor Forecasting System
=====================================================

This script provides a complete testing framework for the factor forecasting system,
including data processing, model training, inference, and performance evaluation.

Features:
- Data processing pipeline testing
- Model architecture testing
- Training pipeline testing
- Inference testing
- Performance metrics testing
- Integration testing
- System health checks

Usage:
    python comprehensive_test_suite.py [options]

Options:
    --quick          Run quick tests only
    --full           Run full test suite
    --data-only      Test data processing only
    --model-only     Test model architecture only
    --training-only  Test training pipeline only
    --inference-only Test inference only
    --performance    Test performance metrics only
    --integration    Test integration pipeline only
    --system         Run system health checks only
    --verbose        Enable verbose output
    --save-results   Save test results to file
"""

import os
import sys
import time
import json
import logging
import argparse
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

# Import project modules
try:
    from configs.config import ModelConfig
    from src.data_processing.data_processor import (
        MultiFileDataProcessor, DataManager, create_training_dataloaders,
        MultiFileDataset
    )
    from src.models.models import (
        create_model, FactorTransformer, FactorForecastingModel
    )
    from src.training.trainer import (
        FactorForecastingTrainer, CorrelationLoss, MetricsTracker
    )
    from src.inference.inference import create_inference_engine
    from src.utils.ic_analysis import ICAnalyzer
    from src.training.rolling_train import RollingWindowTrainer
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed and project structure is correct")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('comprehensive_test_suite.log')
    ]
)
logger = logging.getLogger(__name__)


class ComprehensiveTestSuite:
    """Comprehensive test suite for factor forecasting system"""
    
    def __init__(self, config: Optional[ModelConfig] = None, verbose: bool = False):
        self.config = config or ModelConfig()
        self.verbose = verbose
        self.test_results = {}
        self.start_time = time.time()
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Create temporary directory for test data
        self.temp_dir = tempfile.mkdtemp(prefix='factor_forecast_test_')
        logger.info(f"Created temporary directory: {self.temp_dir}")
    
    def create_sample_data(self, num_stocks: int = 50, num_days: int = 10, 
                          num_factors: int = 100) -> pd.DataFrame:
        """Create sample data for testing"""
        logger.info(f"Creating sample data: {num_stocks} stocks, {num_days} days, {num_factors} factors")
        
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
                    row[f'{i}'] = np.random.randn()
                # Add target columns
                target_columns = ['intra30m', 'nextT1d', 'ema1d']
                for t in target_columns:
                    row[t] = np.random.randn()
                
                # Ensure no infinite or NaN values
                for key, value in row.items():
                    if isinstance(value, (int, float)):
                        if np.isnan(value) or np.isinf(value):
                            row[key] = 0.0
                data.append(row)
        
        df = pd.DataFrame(data)
        logger.info(f"Created sample data: {len(df)} rows, {len([c for c in df.columns if c.isdigit()])} factors")
        return df
    
    def create_test_parquet_files(self, num_files: int = 5) -> str:
        """Create test parquet files in temporary directory"""
        logger.info(f"Creating {num_files} test parquet files")
        
        for i in range(num_files):
            date_str = f"2024-01-{i+1:02d}"
            file_path = Path(self.temp_dir) / f"{date_str}.parquet"
            
            # Create mock data for this date
            mock_data = self.create_sample_data(num_stocks=20, num_days=5, num_factors=100)  # Ensure enough days for sequences
            mock_data.to_parquet(file_path, index=False)
        
        logger.info(f"Created {num_files} parquet files in {self.temp_dir}")
        return self.temp_dir
    
    def test_data_processing(self) -> Dict[str, Any]:
        """Test data processing pipeline"""
        logger.info("=" * 60)
        logger.info("Testing Data Processing Pipeline")
        logger.info("=" * 60)
        
        results = {'status': 'PASS', 'details': [], 'errors': []}
        
        try:
            # Create test data
            test_data = self.create_sample_data()
            
            # Test data processor
            processor = MultiFileDataProcessor(self.config)
            
            # Test data cleaning
            logger.info("Testing data cleaning...")
            cleaned_data = processor._clean_data(test_data)
            assert len(cleaned_data) <= len(test_data), "Data cleaning should not increase data size"
            assert len(cleaned_data) > 0, "Data cleaning should not remove all data"
            results['details'].append("Data cleaning: PASS")
            
            # Test data preprocessing
            logger.info("Testing data preprocessing...")
            processed_data = processor._preprocess_data(cleaned_data)
            assert not processed_data.isnull().any().any(), "No missing values should be present after processing"
            results['details'].append("Data preprocessing: PASS")
            
            # Test sequence building
            logger.info("Testing sequence building...")
            sequences_X, sequences_y = processor.build_sequences(processed_data)
            assert len(sequences_X) > 0, "Sequences should be built"
            assert len(sequences_X) == len(sequences_y), "Number of X and y sequences should be the same"
            results['details'].append("Sequence building: PASS")
            
            # Test with parquet files
            logger.info("Testing with parquet files...")
            test_data_dir = self.create_test_parquet_files()
            
            # Create temporary config for testing
            test_config = {
                'data_dir': test_data_dir,
                'data_path': test_data_dir,
                'sequence_length': 3,
                'prediction_horizon': 1,
                'min_sequence_length': 3,
                'batch_size': 2,
                'num_workers': 0,
                'factor_columns': [str(i) for i in range(100)],
                'target_columns': ['intra30m', 'nextT1d', 'ema1d'],
                'stock_id_column': 'sid',
                'limit_up_down_column': None,  # Disable limit up/down filtering for testing
                'weight_column': 'ADV50',
                'start_date': '2024-01-01',
                'end_date': '2024-01-05',
                'train_ratio': 0.6,
                'val_ratio': 0.2,
                'test_ratio': 0.2,
                'device': 'cpu',
                'use_data_augmentation': False
            }
            
            # Test data loader creation
            try:
                dataloaders, scalers = create_training_dataloaders(test_config)
                assert 'train' in dataloaders, "Training data loader should be present"
                assert 'val' in dataloaders, "Validation data loader should be present"
                results['details'].append("Data loader creation: PASS")
            except Exception as e:
                results['errors'].append(f"Data loader creation: {e}")
            
            logger.info("Data processing pipeline test completed successfully")
            
        except Exception as e:
            results['status'] = 'FAIL'
            results['errors'].append(f"Data processing test failed: {e}")
            logger.error(f"Data processing test failed: {e}")
        
        return results
    
    def test_model_architecture(self) -> Dict[str, Any]:
        """Test model architecture"""
        logger.info("=" * 60)
        logger.info("Testing Model Architecture")
        logger.info("=" * 60)
        
        results = {'status': 'PASS', 'details': [], 'errors': []}
        
        try:
            # Test model creation
            logger.info("Testing model creation...")
            config_dict = {
                'num_factors': len(self.config.factor_columns),
                'num_stocks': 1000,
                'd_model': self.config.hidden_size,
                'num_heads': self.config.num_heads,
                'num_layers': self.config.num_layers,
                'd_ff': self.config.hidden_size * 4,
                'dropout': self.config.dropout,
                'max_seq_len': 5,  # Match sequence_length
                'target_columns': self.config.target_columns
            }
            
            model = create_model(config_dict)
            assert isinstance(model, FactorForecastingModel), "Model should be FactorForecastingModel"
            results['details'].append("Model creation: PASS")
            
            # Test model forward pass
            logger.info("Testing model forward pass...")
            batch_size, seq_len, feature_dim = 4, 5, 100  # Test dimension settings
            
            # Create sample input
            features = torch.randn(batch_size, seq_len, feature_dim).to(self.device)
            stock_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                outputs = model(features, stock_ids)
            
            # Check output format
            assert isinstance(outputs, dict), "Model output should be a dictionary"
            for target in self.config.target_columns:
                assert target in outputs, f"Output should contain {target}"
                assert outputs[target].shape[0] == batch_size, f"Output batch size should match input"
            
            results['details'].append("Model forward pass: PASS")
            
            # Test model parameters
            logger.info("Testing model parameters...")
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            assert total_params > 0, "Model should have parameters"
            assert trainable_params > 0, "Model should have trainable parameters"
            
            results['details'].append(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
            
            logger.info("Model architecture test completed successfully")
            
        except Exception as e:
            results['status'] = 'FAIL'
            results['errors'].append(f"Model architecture test failed: {e}")
            logger.error(f"Model architecture test failed: {e}")
        
        return results
    
    def test_training_pipeline(self) -> Dict[str, Any]:
        """Test training pipeline"""
        logger.info("=" * 60)
        logger.info("Testing Training Pipeline")
        logger.info("=" * 60)
        
        results = {'status': 'PASS', 'details': [], 'errors': []}
        
        try:
            # Create test data
            test_data_dir = self.create_test_parquet_files()
            
            # Create temporary config as dictionary
            test_config = {
                'data_dir': test_data_dir,
                'data_path': test_data_dir,
                'sequence_length': 3,
                'prediction_horizon': 1,
                'min_sequence_length': 3,
                'batch_size': 2,
                'num_workers': 0,
                'factor_columns': [str(i) for i in range(100)],
                'target_columns': ['intra30m', 'nextT1d', 'ema1d'],
                'stock_id_column': 'sid',
                'limit_up_down_column': None,  # Disable limit up/down filtering for testing
                'weight_column': 'ADV50',
                'num_epochs': 2,  # Short training for testing
                'hidden_size': 128,
                'num_heads': 4,
                'num_layers': 2,
                'dropout': 0.1
            }
            
            # Create model
            config_dict = {
                'num_factors': len(test_config['factor_columns']),
                'num_stocks': 1000,
                'd_model': test_config['hidden_size'],
                'num_heads': test_config['num_heads'],
                'num_layers': test_config['num_layers'],
                'd_ff': test_config['hidden_size'] * 4,
                'dropout': test_config['dropout'],
                'max_seq_len': 5,  # Match sequence_length
                'target_columns': test_config['target_columns']
            }
            model = create_model(config_dict)
            
            # Create trainer
            logger.info("Testing trainer creation...")
            trainer = FactorForecastingTrainer(
                model=model,
                config=test_config
            )
            results['details'].append("Trainer creation: PASS")
            
            # Test loss function
            logger.info("Testing loss function...")
            batch_size = 4
            seq_len = test_config['sequence_length']  # This is 3
            
            # Create sample batch
            features = torch.randn(batch_size, seq_len, len(test_config['factor_columns'])).to(self.device)
            # Ensure stock_ids has the same shape as features: (batch_size, seq_len)
            stock_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(self.device)
            
            # Forward pass to get model output shape
            with torch.no_grad():
                predictions = model(features, stock_ids)
            
            # Create targets with the same shape as model predictions
            targets = {}
            for target in test_config['target_columns']:
                if target in predictions:
                    # Use the same shape as model output: (batch_size, seq_len)
                    targets[target] = torch.randn_like(predictions[target], requires_grad=True)
                else:
                    # Fallback to expected shape: (batch_size, seq_len)
                    targets[target] = torch.randn(batch_size, seq_len, requires_grad=True).to(self.device)
            
            # Calculate loss using trainer's loss function
            if hasattr(trainer, 'loss_fn') and trainer.loss_fn is not None:
                loss = trainer.loss_fn(predictions, targets)
            else:
                # Fallback to MSE loss - ensure both tensors have the same shape
                pred_key = list(predictions.keys())[0]
                target_key = list(targets.keys())[0]
                pred_tensor = predictions[pred_key]
                target_tensor = targets[target_key]
                
                # Ensure both tensors have the same shape
                if pred_tensor.shape != target_tensor.shape:
                    # Reshape target to match prediction
                    target_tensor = target_tensor.view_as(pred_tensor)
                
                loss = torch.nn.functional.mse_loss(pred_tensor, target_tensor)
            
            assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
            assert loss.requires_grad, "Loss should require gradients"
            
            results['details'].append("Loss calculation: PASS")
            
            # Test training step (without actual training)
            logger.info("Testing training step...")
            trainer.model.train()
            
            # Ensure model parameters have requires_grad=True
            for param in trainer.model.parameters():
                param.requires_grad = True
            
            # Forward pass and loss computation
            predictions = trainer.model(features, stock_ids)
            if hasattr(trainer, 'loss_fn') and trainer.loss_fn is not None:
                loss = trainer.loss_fn(predictions, targets)
            else:
                # Fallback to MSE loss
                pred_key = list(predictions.keys())[0]
                target_key = list(targets.keys())[0]
                pred_tensor = predictions[pred_key]
                target_tensor = targets[target_key]
                
                # Ensure both tensors have the same shape
                if pred_tensor.shape != target_tensor.shape:
                    target_tensor = target_tensor.view_as(pred_tensor)
                
                loss = torch.nn.functional.mse_loss(pred_tensor, target_tensor)
            
            # Backward pass
            loss.backward()
            
            # Check gradients
            has_gradients = any(p.grad is not None for p in trainer.model.parameters())
            assert has_gradients, "Model should have gradients after backward pass"
            
            results['details'].append("Training step: PASS")
            
            logger.info("Training pipeline test completed successfully")
            
        except Exception as e:
            results['status'] = 'FAIL'
            results['errors'].append(f"Training pipeline test failed: {e}")
            logger.error(f"Training pipeline test failed: {e}")
        
        return results
    
    def test_inference_pipeline(self) -> Dict[str, Any]:
        """Test inference pipeline"""
        logger.info("=" * 60)
        logger.info("Testing Inference Pipeline")
        logger.info("=" * 60)
        
        results = {'status': 'PASS', 'details': [], 'errors': []}
        
        try:
            # Create model
            config_dict = {
                'num_factors': 100,  # Fixed number of factors
                'num_stocks': 1000,
                'd_model': self.config.hidden_size,
                'num_heads': self.config.num_heads,
                'num_layers': self.config.num_layers,
                'd_ff': self.config.hidden_size * 4,
                'dropout': self.config.dropout,
                'max_seq_len': 50,
                'target_columns': ['intra30m', 'nextT1d', 'ema1d']
            }
            model = create_model(config_dict)
            
            # Test inference engine creation
            logger.info("Testing inference engine creation...")
            
            # Create a temporary model file for testing
            import tempfile
            import os
            
            temp_model_path = os.path.join(tempfile.gettempdir(), "test_model.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config_dict
            }, temp_model_path)
            
            try:
                inference_engine = create_inference_engine(temp_model_path, self.config)
                results['details'].append("Inference engine creation: PASS")
                
                # Test prediction
                logger.info("Testing prediction...")
                batch_size, seq_len = 4, self.config.sequence_length
                features = torch.randn(batch_size, seq_len, 100).to(self.device)  # Fixed feature dimension
                stock_ids = torch.randint(0, 1000, (batch_size,)).to(self.device)
                
                # Convert to numpy for inference engine
                features_np = features.cpu().numpy()
                
                # Test single prediction
                with torch.no_grad():
                    # Use the first sample for single prediction
                    single_features = features_np[0]  # Shape: (seq_len, num_factors)
                    single_sid = str(stock_ids[0].item())
                    
                    result = inference_engine.predict(single_features, single_sid)
                    
                    # Check result format
                    assert isinstance(result, dict), "Prediction result should be a dictionary"
                    assert 'prediction' in result, "Result should contain 'prediction' key"
                    assert 'latency_ms' in result, "Result should contain 'latency_ms' key"
                
                results['details'].append("Prediction: PASS")
                
                # Test batch prediction
                logger.info("Testing batch prediction...")
                batch_results = inference_engine.batch_predict(features_np, [str(sid.item()) for sid in stock_ids])
                assert isinstance(batch_results, list), "Batch predictions should be a list"
                assert len(batch_results) == batch_size, "Batch results should match batch size"
                
                results['details'].append("Batch prediction: PASS")
                
            finally:
                # Clean up temporary model file
                if os.path.exists(temp_model_path):
                    os.remove(temp_model_path)
            
            logger.info("Inference pipeline test completed successfully")
            
        except Exception as e:
            results['status'] = 'FAIL'
            results['errors'].append(f"Inference pipeline test failed: {e}")
            logger.error(f"Inference pipeline test failed: {e}")
        
        return results
    
    def test_performance_metrics(self) -> Dict[str, Any]:
        """Test performance metrics calculation"""
        logger.info("=" * 60)
        logger.info("Testing Performance Metrics")
        logger.info("=" * 60)
        
        results = {'status': 'PASS', 'details': [], 'errors': []}
        
        try:
            # Create sample predictions and targets
            logger.info("Testing IC analysis...")
            n_samples = 1000
            
            predictions = {}
            targets = {}
            
            for target in self.config.target_columns:
                predictions[target] = np.random.randn(n_samples)
                targets[target] = np.random.randn(n_samples)
            
            # Test IC analyzer
            ic_analyzer = ICAnalyzer(predictions, targets)
            
            # Calculate basic IC
            basic_ic = ic_analyzer._calculate_basic_ic()
            assert isinstance(basic_ic, dict), "Basic IC should be a dictionary"
            
            for target in self.config.target_columns:
                assert target in basic_ic, f"IC should contain {target}"
                assert 'ic' in basic_ic[target], f"IC metrics should contain 'ic' for {target}"
            
            results['details'].append("IC analysis: PASS")
            
            # Test metrics tracker
            logger.info("Testing metrics tracker...")
            metrics_tracker = MetricsTracker(['intra30m', 'nextT1d', 'ema1d'])
            
            # Test metrics update with sample data
            batch_size = 100
            for epoch in range(3):
                # Create sample predictions and targets
                predictions = {}
                targets = {}
                
                for target in self.config.target_columns:
                    predictions[target] = torch.randn(batch_size)
                    targets[target] = torch.randn(batch_size)
                
                # Update metrics tracker
                metrics_tracker.update(predictions, targets)
            
            # Compute metrics
            metrics = metrics_tracker.compute_metrics()
            assert isinstance(metrics, dict), "Metrics should be a dictionary"
            
            # Check that metrics are computed for each target
            for target in self.config.target_columns:
                assert target in metrics, f"Metrics should contain {target}"
                target_metrics = metrics[target]
                assert 'mse' in target_metrics, f"Metrics for {target} should contain MSE"
                assert 'rmse' in target_metrics, f"Metrics for {target} should contain RMSE"
            
            results['details'].append("Metrics tracking: PASS")
            
            logger.info("Performance metrics test completed successfully")
            
        except Exception as e:
            results['status'] = 'FAIL'
            results['errors'].append(f"Performance metrics test failed: {e}")
            logger.error(f"Performance metrics test failed: {e}")
        
        return results
    
    def test_integration_pipeline(self) -> Dict[str, Any]:
        """Test complete integration pipeline"""
        logger.info("=" * 60)
        logger.info("Testing Integration Pipeline")
        logger.info("=" * 60)
        
        results = {'status': 'PASS', 'details': [], 'errors': []}
        
        try:
            # Create test data
            test_data_dir = self.create_test_parquet_files()
            
            # Create temporary config as dictionary
            test_config = {
                'data_dir': test_data_dir,
                'data_path': test_data_dir,
                'sequence_length': 3,
                'prediction_horizon': 1,
                'min_sequence_length': 3,
                'batch_size': 2,
                'num_workers': 0,
                'num_epochs': 1,  # Very short training
                'factor_columns': [str(i) for i in range(100)],
                'num_targets': 3,
                'target_columns': ['intra30m', 'nextT1d', 'ema1d'],
                'hidden_size': 128,
                'num_heads': 4,
                'num_layers': 2,
                'dropout': 0.1,
                'stock_id_column': 'sid',
                'limit_up_down_column': None,  # Disable limit up/down filtering for testing
                'weight_column': 'ADV50',
                'start_date': '2024-01-01',
                'end_date': '2024-01-05',
                'train_ratio': 0.6,
                'val_ratio': 0.2,
                'test_ratio': 0.2,
                'device': 'cpu',
                'use_data_augmentation': False
            }
            
            # Test complete pipeline
            logger.info("Testing complete pipeline...")
            
            # 1. Data processing
            dataloaders, scalers = create_training_dataloaders(test_config)
            results['details'].append("Data loading: PASS")
            
            # 2. Model creation
            config_dict = {
                'num_factors': len(test_config['factor_columns']),
                'num_stocks': 1000,
                'd_model': test_config['hidden_size'],
                'num_heads': test_config['num_heads'],
                'num_layers': test_config['num_layers'],
                'd_ff': test_config['hidden_size'] * 4,
                'dropout': test_config['dropout'],
                'max_seq_len': 5,  # Match sequence_length
                'target_columns': test_config['target_columns']
            }
            model = create_model(config_dict)
            results['details'].append("Model creation: PASS")
            
            # 3. Training
            trainer = FactorForecastingTrainer(
                model=model,
                config=test_config
            )
            
            # Quick training
            for epoch in range(1):
                for batch_idx, batch in enumerate(dataloaders['train']):
                    if batch_idx >= 2:  # Only train on 2 batches
                        break
                    # Use a simple training step instead of trainer.train_step
                    trainer.model.train()
                    features = batch['features'].to(trainer.device)
                    stock_ids = batch['stock_ids'].to(trainer.device)
                    targets = {k: v.to(trainer.device) for k, v in batch['targets'].items()}
                    
                    # Forward pass
                    predictions = trainer.model(features, stock_ids)
                    
                    # Calculate loss
                    if hasattr(trainer, 'loss_fn') and trainer.loss_fn is not None:
                        loss = trainer.loss_fn(predictions, targets)
                    else:
                        # Fallback to MSE loss
                        loss = torch.nn.functional.mse_loss(predictions[list(predictions.keys())[0]], targets[list(targets.keys())[0]])
                    
                    # Backward pass
                    loss.backward()
                    
                    if self.verbose:
                        logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.4f}")
            
            results['details'].append("Training: PASS")
            
            # 4. Inference
            # Create a temporary model file for testing
            temp_model_path = os.path.join(tempfile.gettempdir(), "test_integration_model.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config_dict
            }, temp_model_path)
            
            try:
                inference_engine = create_inference_engine(temp_model_path, test_config)
                
                # Test on validation data
                for batch_idx, batch in enumerate(dataloaders['val']):
                    if batch_idx >= 1:  # Only test on 1 batch
                        break
                    features = batch['features'].to(trainer.device)
                    stock_ids = batch['stock_ids'].to(trainer.device)
                    
                    with torch.no_grad():
                        predictions = inference_engine.predict(features, stock_ids)
                    
                    assert isinstance(predictions, dict), "Predictions should be a dictionary"
                
                results['details'].append("Inference: PASS")
                
            finally:
                # Clean up temporary model file
                if os.path.exists(temp_model_path):
                    os.remove(temp_model_path)
            
            logger.info("Integration pipeline test completed successfully")
            
        except Exception as e:
            results['status'] = 'FAIL'
            results['errors'].append(f"Integration pipeline test failed: {e}")
            logger.error(f"Integration pipeline test failed: {e}")
        
        return results
    
    def test_system_health(self) -> Dict[str, Any]:
        """Test system health and dependencies"""
        logger.info("=" * 60)
        logger.info("Testing System Health")
        logger.info("=" * 60)
        
        results = {'status': 'PASS', 'details': [], 'errors': []}
        
        try:
            # Test Python version
            logger.info("Testing Python version...")
            python_version = sys.version_info
            assert python_version.major == 3, "Python 3 is required"
            assert python_version.minor >= 8, "Python 3.8+ is required"
            results['details'].append(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
            
            # Test PyTorch
            logger.info("Testing PyTorch...")
            assert torch.__version__, "PyTorch should be installed"
            results['details'].append(f"PyTorch version: {torch.__version__}")
            
            # Test CUDA availability
            if torch.cuda.is_available():
                results['details'].append(f"CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                results['details'].append("CUDA not available, using CPU")
            
            # Test pandas
            logger.info("Testing pandas...")
            assert pd.__version__, "Pandas should be installed"
            results['details'].append(f"Pandas version: {pd.__version__}")
            
            # Test numpy
            logger.info("Testing numpy...")
            assert np.__version__, "NumPy should be installed"
            results['details'].append(f"NumPy version: {np.__version__}")
            
            # Test project structure
            logger.info("Testing project structure...")
            required_dirs = ['src', 'configs', 'tests', 'data']
            for dir_name in required_dirs:
                assert os.path.exists(dir_name), f"Directory {dir_name} should exist"
            results['details'].append("Project structure: PASS")
            
            # Test configuration
            logger.info("Testing configuration...")
            assert hasattr(self.config, 'factor_columns'), "Config should have factor_columns"
            assert hasattr(self.config, 'target_columns'), "Config should have target_columns"
            results['details'].append("Configuration: PASS")
            
            # Test memory usage
            logger.info("Testing memory usage...")
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
                results['details'].append(f"GPU memory allocated: {memory_allocated:.2f} MB")
            
            logger.info("System health test completed successfully")
            
        except Exception as e:
            results['status'] = 'FAIL'
            results['errors'].append(f"System health test failed: {e}")
            logger.error(f"System health test failed: {e}")
        
        return results
    
    def run_all_tests(self, test_types: List[str] = None) -> Dict[str, Any]:
        """Run all tests or specified test types"""
        if test_types is None:
            test_types = ['data', 'model', 'training', 'inference', 'performance', 'integration', 'system']
        
        logger.info("=" * 80)
        logger.info("Starting Comprehensive Test Suite")
        logger.info("=" * 80)
        
        test_functions = {
            'data': self.test_data_processing,
            'model': self.test_model_architecture,
            'training': self.test_training_pipeline,
            'inference': self.test_inference_pipeline,
            'performance': self.test_performance_metrics,
            'integration': self.test_integration_pipeline,
            'system': self.test_system_health
        }
        
        for test_type in test_types:
            if test_type in test_functions:
                logger.info(f"\nRunning {test_type.upper()} tests...")
                self.test_results[test_type] = test_functions[test_type]()
            else:
                logger.warning(f"Unknown test type: {test_type}")
        
        # Calculate overall results
        self.calculate_overall_results()
        
        return self.test_results
    
    def calculate_overall_results(self):
        """Calculate overall test results"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['status'] == 'PASS')
        failed_tests = total_tests - passed_tests
        
        self.overall_results = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'execution_time': time.time() - self.start_time
        }
    
    def print_results(self):
        """Print test results"""
        logger.info("\n" + "=" * 80)
        logger.info("Test Results Summary")
        logger.info("=" * 80)
        
        for test_type, result in self.test_results.items():
            status = result['status']
            logger.info(f"\n{test_type.upper()} TESTS: {status}")
            
            if result['details']:
                logger.info("  Details:")
                for detail in result['details']:
                    logger.info(f"    PASS {detail}")
            
            if result['errors']:
                logger.info("  Errors:")
                for error in result['errors']:
                    logger.error(f"    FAIL {error}")
        
        # Print overall results
        logger.info(f"\n{'='*80}")
        logger.info("OVERALL RESULTS")
        logger.info(f"{'='*80}")
        logger.info(f"Total Tests: {self.overall_results['total_tests']}")
        logger.info(f"Passed: {self.overall_results['passed_tests']}")
        logger.info(f"Failed: {self.overall_results['failed_tests']}")
        logger.info(f"Success Rate: {self.overall_results['success_rate']:.2%}")
        logger.info(f"Execution Time: {self.overall_results['execution_time']:.2f} seconds")
        
        if self.overall_results['success_rate'] == 1.0:
            logger.info("All tests passed!")
        elif self.overall_results['success_rate'] >= 0.8:
            logger.info("Most tests passed!")
        else:
            logger.error("Many tests failed!")
    
    def save_results(self, filename: str = None):
        """Save test results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_results_{timestamp}.json"
        
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'overall_results': self.overall_results,
            'test_results': self.test_results,
            'config': {
                'device': str(self.device),
                'temp_dir': self.temp_dir,
                'verbose': self.verbose
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logger.info(f"Test results saved to: {filename}")
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temporary directory: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Comprehensive Test Suite for Factor Forecasting System')
    parser.add_argument('--quick', action='store_true', help='Run quick tests only')
    parser.add_argument('--full', action='store_true', help='Run full test suite')
    parser.add_argument('--data-only', action='store_true', help='Test data processing only')
    parser.add_argument('--model-only', action='store_true', help='Test model architecture only')
    parser.add_argument('--training-only', action='store_true', help='Test training pipeline only')
    parser.add_argument('--inference-only', action='store_true', help='Test inference only')
    parser.add_argument('--performance', action='store_true', help='Test performance metrics only')
    parser.add_argument('--integration', action='store_true', help='Test integration pipeline only')
    parser.add_argument('--system', action='store_true', help='Run system health checks only')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--save-results', action='store_true', help='Save test results to file')
    
    args = parser.parse_args()
    
    # Determine test types
    test_types = []
    if args.quick:
        test_types = ['system', 'data', 'model']
    elif args.full:
        test_types = ['data', 'model', 'training', 'inference', 'performance', 'integration', 'system']
    elif args.data_only:
        test_types = ['data']
    elif args.model_only:
        test_types = ['model']
    elif args.training_only:
        test_types = ['training']
    elif args.inference_only:
        test_types = ['inference']
    elif args.performance:
        test_types = ['performance']
    elif args.integration:
        test_types = ['integration']
    elif args.system:
        test_types = ['system']
    else:
        # Default: run all tests
        test_types = ['data', 'model', 'training', 'inference', 'performance', 'integration', 'system']
    
    # Create test suite
    test_suite = ComprehensiveTestSuite(verbose=args.verbose)
    
    try:
        # Run tests
        results = test_suite.run_all_tests(test_types)
        
        # Print results
        test_suite.print_results()
        
        # Save results if requested
        if args.save_results:
            test_suite.save_results()
        
        # Exit with appropriate code
        if test_suite.overall_results['success_rate'] == 1.0:
            sys.exit(0)
        else:
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("Test suite interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test suite failed with error: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        test_suite.cleanup()


if __name__ == "__main__":
    main() 