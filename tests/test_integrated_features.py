#!/usr/bin/env python3
"""
Comprehensive test suite for integrated features:
- Async preloading
- Incremental learning
- Streaming data loading
- Integration between all components
"""
import sys
import os
import unittest
import tempfile
import shutil
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configs.config import ModelConfig
from src.training.integrated_training import IntegratedTrainingSystem, create_integrated_training_system
from src.training.incremental_learning import IncrementalTrainer, ExperienceReplay
from src.data_processing.async_preloader import AsyncDataPreloader
from src.data_processing.streaming_data_loader import StreamingDataLoader
from src.models.models import FactorForecastingModel
import logging

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestIntegratedFeatures(unittest.TestCase):
    """Test suite for integrated features."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = ModelConfig()
        self.config.batch_size = 32
        self.config.sequence_length = 10
        self.config.hidden_dim = 64
        self.config.num_layers = 2
        self.config.learning_rate = 0.01
        self.config.num_epochs = 2  # Small for testing
        
        # Create test data
        self._create_test_data()
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _create_test_data(self):
        """Create synthetic test data."""
        np.random.seed(42)
        
        # Create multiple daily files
        start_date = pd.Timestamp('2023-01-01')
        
        for i in range(5):  # 5 days of data
            date = start_date + pd.Timedelta(days=i)
            
            # Create synthetic factor data
            n_stocks = 100
            n_factors = len(self.config.factor_columns)
            
            data = {
                'date': [date] * n_stocks,
                'sid': list(range(n_stocks)),
                'time': [date] * n_stocks
            }
            
            # Add factor columns
            for j, factor in enumerate(self.config.factor_columns):
                data[factor] = np.random.randn(n_stocks) * 0.1 + j * 0.01
            
            # Add target columns
            for j, target in enumerate(self.config.target_columns):
                data[target] = np.random.randn(n_stocks) * 0.05 + j * 0.005
            
            df = pd.DataFrame(data)
            
            # Save as parquet file
            filename = f"{date.strftime('%Y%m%d')}.parquet"
            filepath = os.path.join(self.temp_dir, filename)
            df.to_parquet(filepath, index=False)
        
        logger.info(f"Created test data in {self.temp_dir}")
    
    def test_experience_replay(self):
        """Test experience replay buffer functionality."""
        logger.info("Testing ExperienceReplay...")
        
        replay_buffer = ExperienceReplay(capacity=100)
        
        # Test adding experiences
        for i in range(50):
            experience = {
                'features': torch.randn(10, 20),
                'targets': torch.randn(10, 3)
            }
            replay_buffer.add(experience, importance=float(i))
        
        self.assertEqual(replay_buffer.size(), 50)
        
        # Test sampling
        samples = replay_buffer.sample(batch_size=10)
        self.assertEqual(len(samples), 10)
        
        # Test buffer overflow
        for i in range(60):
            experience = {
                'features': torch.randn(10, 20),
                'targets': torch.randn(10, 3)
            }
            replay_buffer.add(experience, importance=float(i))
        
        self.assertEqual(replay_buffer.size(), 100)  # Should be capped at capacity
        
        logger.info(" ExperienceReplay test passed")
    
    def test_async_preloader(self):
        """Test async data preloader functionality."""
        logger.info("Testing AsyncDataPreloader...")
        
        # Create a simple dataset
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, size=100):
                self.size = size
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                return {
                    'features': torch.randn(20),
                    'targets': torch.randn(3)
                }
        
        dataset = SimpleDataset(50)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
        
        device = torch.device('cpu')  # Use CPU for testing
        
        # Test async preloader  
        def data_loader_func():
            return data_loader
        
        preloader = AsyncDataPreloader(
            data_loader_func=data_loader_func,
            queue_size=2,
            prefetch_size=1
        )
        
        preloader.start_preloading()
        
        # Consume some batches
        batch_count = 0
        for batch in preloader:
            self.assertIn('features', batch)
            self.assertIn('targets', batch)
            self.assertEqual(batch['features'].device, device)
            batch_count += 1
            if batch_count >= 3:  # Test a few batches
                break
        
        preloader.stop_preloading()
        
        logger.info(" AsyncDataPreloader test passed")
    
    def test_streaming_data_loader(self):
        """Test streaming data loader functionality."""
        logger.info("Testing StreamingDataLoader...")
        
        streaming_loader = StreamingDataLoader(
            data_dir=self.temp_dir,
            batch_size=16,
            cache_size=3,
            max_memory_mb=512
        )
        
        # Test data file discovery
        self.assertGreater(len(streaming_loader.data_files), 0)
        
        # Test data loader creation
        train_loader, val_loader, test_loader = streaming_loader.create_data_loaders(
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2
        )
        
        # Test data loading
        batch_count = 0
        for batch in train_loader:
            self.assertIn('features', batch)
            self.assertIn('targets', batch)
            batch_count += 1
            if batch_count >= 2:  # Test a few batches
                break
        
        self.assertGreater(batch_count, 0)
        
        logger.info(" StreamingDataLoader test passed")
    
    def test_incremental_trainer(self):
        """Test incremental trainer functionality."""
        logger.info("Testing IncrementalTrainer...")
        
        device = torch.device('cpu')
        
        # Create model config
        model_config = {
            'input_dim': len(self.config.factor_columns),
            'hidden_size': 32,
            'num_layers': 2,
            'output_dim': len(self.config.target_columns),
            'dropout': 0.1,
            'model_type': 'transformer'
        }
        
        model = FactorForecastingModel(model_config).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        
        # Create incremental trainer
        trainer = IncrementalTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            replay_capacity=100,
            use_async_preloader=False  # Disable for testing
        )
        
        # Test training step
        batch = {
            'features': torch.randn(8, self.config.sequence_length, len(self.config.factor_columns)),
            'targets': torch.randn(8, len(self.config.target_columns))
        }
        
        stats = trainer.train_step(batch, store_experience=True)
        self.assertIn('loss', stats)
        self.assertIsInstance(stats['loss'], float)
        
        # Test replay buffer
        self.assertGreater(trainer.replay_buffer.size(), 0)
        
        logger.info(" IncrementalTrainer test passed")
    
    def test_integrated_training_system(self):
        """Test the complete integrated training system."""
        logger.info("Testing IntegratedTrainingSystem...")
        
        device = torch.device('cpu')
        
        # Create integrated training system
        training_system = create_integrated_training_system(
            config=self.config,
            device=device,
            use_incremental_learning=True,
            use_async_preloader=False  # Disable for stability in testing
        )
        
        # Test status before setup
        status = training_system.get_status()
        self.assertFalse(status['model_initialized'])
        
        # Setup model
        training_system.setup_model()
        status = training_system.get_status()
        self.assertTrue(status['model_initialized'])
        
        # Setup data loading
        training_system.setup_data_loading(self.temp_dir)
        status = training_system.get_status()
        self.assertTrue(status['streaming_loader_initialized'])
        
        # Setup training components
        training_system.setup_training_components()
        
        # Test short training run
        logger.info("Running short training test...")
        results = training_system.train(
            data_dir=self.temp_dir,
            epochs=1,  # Very short for testing
            validation_split=0.3
        )
        
        # Verify results
        self.assertIn('training_history', results)
        self.assertIn('best_val_loss', results)
        self.assertGreater(len(results['training_history']), 0)
        
        logger.info(" IntegratedTrainingSystem test passed")
    
    def test_integration_with_different_configurations(self):
        """Test integration with different model configurations."""
        logger.info("Testing integration with different configurations...")
        
        configurations = [
            {'model_type': 'transformer', 'use_incremental': True, 'use_async': False},
            {'model_type': 'lstm', 'use_incremental': False, 'use_async': False},
            {'model_type': 'tcn', 'use_incremental': True, 'use_async': False}
        ]
        
        for i, config_dict in enumerate(configurations):
            logger.info(f"Testing configuration {i+1}: {config_dict}")
            
            config = ModelConfig()
            config.model_type = config_dict['model_type']
            config.batch_size = 16
            config.num_epochs = 1
            
            training_system = create_integrated_training_system(
                config=config,
                device=torch.device('cpu'),
                use_incremental_learning=config_dict['use_incremental'],
                use_async_preloader=config_dict['use_async']
            )
            
            # Quick setup and training test
            training_system.setup_model()
            training_system.setup_data_loading(self.temp_dir)
            training_system.setup_training_components()
            
            # Very short training
            results = training_system.train(
                data_dir=self.temp_dir,
                epochs=1,
                validation_split=0.3
            )
            
            self.assertIsInstance(results['best_val_loss'], float)
            logger.info(f" Configuration {i+1} test passed")
        
        logger.info(" All configuration tests passed")
    
    def run_all_tests(self):
        """Run all tests in sequence."""
        logger.info("=" * 50)
        logger.info("RUNNING INTEGRATED FEATURES TEST SUITE")
        logger.info("=" * 50)
        
        tests = [
            self.test_experience_replay,
            self.test_async_preloader,
            self.test_streaming_data_loader,
            self.test_incremental_trainer,
            self.test_integrated_training_system,
            self.test_integration_with_different_configurations
        ]
        
        passed = 0
        failed = 0
        
        for test in tests:
            try:
                self.setUp()
                test()
                passed += 1
            except Exception as e:
                logger.error(f"Test {test.__name__} failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                failed += 1
            finally:
                self.tearDown()
        
        logger.info("=" * 50)
        logger.info("TEST RESULTS")
        logger.info("=" * 50)
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Total: {passed + failed}")
        
        if failed == 0:
            logger.info("ALL TESTS PASSED!")
        else:
            logger.warning(f"WARNING: {failed} test(s) failed")
        
        return failed == 0


def main():
    """Main function to run the test suite."""
    tester = TestIntegratedFeatures()
    success = tester.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
