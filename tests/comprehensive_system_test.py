#!/usr/bin/env python3
"""
Comprehensive System Test: Quantitative finance time series prediction project integrity verification
Test data loading, model training, memory management and other key components
"""
import os
import sys
import pytest
import logging
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import json
from typing import Dict, Any

# Add project path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_processing.adaptive_memory_manager import create_memory_manager
from data_processing.optimized_streaming_loader import OptimizedStreamingDataLoader
from models.advanced_tcn_attention import create_advanced_model
from utils.model_benchmarker import ModelBenchmarker

logger = logging.getLogger(__name__)


class ComprehensiveSystemTest:
    """Comprehensive system test class"""
    
    def __init__(self):
        self.test_results = {}
        self.temp_dir = None
        
    def setup_test_environment(self):
        """Set up test environment"""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        logger.info("Comprehensive system test started...")
        
        return True
    
    def create_test_data(self) -> bool:
        """Create test data"""
        try:
            logger.info("Creating test data...")
            
            # Create simulated parquet data files
            data_dir = Path(self.temp_dir) / "test_data"
            data_dir.mkdir(exist_ok=True)
            
            # Create 3 test files
            for i, date in enumerate(["20180101", "20180102", "20180103"]):
                # Generate simulated data
                n_stocks = 1000
                n_records = 5000
                
                data = {
                    'sid': np.random.randint(0, n_stocks, n_records),
                    **{f'factor_{j}': np.random.randn(n_records) for j in range(100)},
                    'intra30m': np.random.randn(n_records) * 0.01,
                    'nextT1d': np.random.randn(n_records) * 0.02,
                    'ema1d': np.random.randn(n_records) * 0.015
                }
                
                df = pd.DataFrame(data)
                df.to_parquet(data_dir / f"{date}.parquet")
            
            self.test_data_dir = data_dir
            logger.info(f"Test data creation completed: {data_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create test data: {e}")
            return False
    
    def test_memory_manager(self) -> bool:
        """Test adaptive memory manager"""
        try:
            logger.info("Testing adaptive memory manager...")
            
            # Create memory manager
            memory_manager = create_memory_manager()
            
            # Test memory status retrieval
            status = memory_manager.get_memory_status()
            assert 'system' in status
            assert 'budget' in status
            
            # Test batch size calculation
            batch_size = memory_manager.calculate_optimal_batch_size(
                data_sample_mb=50,
                sequence_length=10,
                feature_dim=100
            )
            assert isinstance(batch_size, int)
            assert batch_size > 0
            
            # Test cache size calculation
            cache_size = memory_manager.calculate_optimal_cache_size(100)
            assert isinstance(cache_size, int)
            assert cache_size > 0
            
            self.test_results['memory_manager'] = {
                'status': 'PASS',
                'optimal_batch_size': batch_size,
                'optimal_cache_size': cache_size
            }
            
            logger.info("Memory manager test passed")
            return True
            
        except Exception as e:
            logger.error(f"Memory manager test failed: {e}")
            self.test_results['memory_manager'] = {'status': 'FAIL', 'error': str(e)}
            return False
    
    def test_streaming_loader(self) -> bool:
        """Test optimized streaming data loader"""
        try:
            logger.info("Testing optimized streaming data loader...")
            
            if not hasattr(self, 'test_data_dir'):
                raise ValueError("Test data not created")
            
            # Create memory manager
            memory_manager = create_memory_manager()
            
            # Create streaming loader
            with OptimizedStreamingDataLoader(
                str(self.test_data_dir),
                memory_manager=memory_manager
            ) as loader:
                
                # Test data stream
                data_count = 0
                total_rows = 0
                
                for df in loader.stream_by_date_range():
                    assert isinstance(df, pd.DataFrame)
                    assert len(df) > 0
                    data_count += 1
                    total_rows += len(df)
                    
                    if data_count >= 3:  # Test only first 3 files
                        break
                
                # Test statistics
                stats = loader.get_comprehensive_stats()
                assert 'data_files' in stats
                assert 'memory_manager' in stats
                
                self.test_results['streaming_loader'] = {
                    'status': 'PASS',
                    'files_processed': data_count,
                    'total_rows': total_rows,
                    'stats': stats
                }
            
            logger.info(f"Streaming loader test passed - processed {data_count} files, {total_rows} rows")
            return True
            
        except Exception as e:
            logger.error(f"Streaming loader test failed: {e}")
            self.test_results['streaming_loader'] = {'status': 'FAIL', 'error': str(e)}
            return False
    
    def test_model_creation(self) -> bool:
        """Test model creation and inference"""
        try:
            logger.info("Testing model creation and inference...")
            
            # Model configuration
            config = {
                'input_dim': 100,
                'hidden_dim': 128,
                'num_layers': 4,
                'num_heads': 8,
                'dropout': 0.1,
                'num_stocks': 1000,
                'sequence_length': 10,
                'target_columns': ['intra30m', 'nextT1d', 'ema1d'],
                'kernel_size': 3,
                'use_relative_pos': True,
                'use_multi_scale': True,
                'use_adaptive': True
            }
            
            # Create model
            model = create_advanced_model(config)
            model.eval()
            
            # Test inference
            batch_size = 8
            sequence_length = 10
            feature_dim = 100
            
            factors = torch.randn(batch_size, sequence_length, feature_dim)
            stock_ids = torch.randint(0, 1000, (batch_size, sequence_length))
            
            with torch.no_grad():
                predictions = model(factors, stock_ids)
            
            # Validate output
            assert isinstance(predictions, dict)
            for target in config['target_columns']:
                assert target in predictions
                assert predictions[target].shape == (batch_size, sequence_length)
            
            # Get model information
            model_info = model.get_model_info()
            
            self.test_results['model_creation'] = {
                'status': 'PASS',
                'model_type': model_info['model_type'],
                'total_parameters': model_info['total_parameters'],
                'output_shapes': {k: list(v.shape) for k, v in predictions.items()}
            }
            
            logger.info(f"Model test passed - parameter count: {model_info['total_parameters']:,}")
            return True
            
        except Exception as e:
            logger.error(f"Model creation test failed: {e}")
            self.test_results['model_creation'] = {'status': 'FAIL', 'error': str(e)}
            return False
    
    def test_data_pipeline_integration(self) -> bool:
        """Test data pipeline integration"""
        try:
            logger.info("Testing data pipeline integration...")
            
            if not hasattr(self, 'test_data_dir'):
                raise ValueError("Test data not created")
            
            # Create memory manager
            memory_manager = create_memory_manager()
            
            # Create streaming loader
            with OptimizedStreamingDataLoader(
                str(self.test_data_dir),
                memory_manager=memory_manager
            ) as loader:
                
                # Simulate training data pipeline
                factor_columns = [f'factor_{i}' for i in range(100)]
                target_columns = ['intra30m', 'nextT1d', 'ema1d']
                
                total_sequences = 0
                
                for df in loader.stream_by_date_range(
                    required_columns=['sid'] + factor_columns + target_columns
                ):
                    # Simulate sequence creation
                    if len(df) < 10:
                        continue
                    
                    # Group by stock
                    for stock_id, stock_group in df.groupby('sid'):
                        if len(stock_group) >= 10:
                            # Create a sequence
                            features = stock_group[factor_columns].iloc[:10].values
                            targets = stock_group[target_columns].iloc[9].values
                            
                            # Validate data integrity
                            if not np.isnan(features).any() and not np.isnan(targets).any():
                                total_sequences += 1
                    
                    if total_sequences >= 100:  # Limit test quantity
                        break
                
                self.test_results['data_pipeline'] = {
                    'status': 'PASS',
                    'sequences_created': total_sequences
                }
            
            logger.info(f"Data pipeline test passed - created {total_sequences} sequences")
            return True
            
        except Exception as e:
            logger.error(f"Data pipeline test failed: {e}")
            self.test_results['data_pipeline'] = {'status': 'FAIL', 'error': str(e)}
            return False
    
    def test_training_simulation(self) -> bool:
        """Test training simulation"""
        try:
            logger.info("Testing training simulation...")
            
            # Create small model configuration
            config = {
                'input_dim': 100,
                'hidden_dim': 64,
                'num_layers': 2,
                'num_heads': 4,
                'dropout': 0.1,
                'num_stocks': 100,
                'sequence_length': 5,
                'target_columns': ['nextT1d'],
                'kernel_size': 3
            }
            
            # Create model
            model = create_advanced_model(config)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # Simulate training steps
            model.train()
            training_losses = []
            
            for step in range(5):  # Train only 5 steps
                # Create simulated batch
                batch_size = 16
                factors = torch.randn(batch_size, 5, 100)
                stock_ids = torch.randint(0, 100, (batch_size, 5))
                targets = torch.randn(batch_size, 5)
                
                # Forward pass
                predictions = model(factors, stock_ids)
                targets_dict = {'nextT1d': targets}
                
                # Calculate loss
                loss_dict = model.compute_loss(predictions, targets_dict)
                loss = loss_dict['total_loss']
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                training_losses.append(loss.item())
            
            # Validate training process
            assert len(training_losses) == 5
            assert all(isinstance(loss, float) for loss in training_losses)
            
            self.test_results['training_simulation'] = {
                'status': 'PASS',
                'training_losses': training_losses,
                'final_loss': training_losses[-1]
            }
            
            logger.info(f"Training simulation test passed - final loss: {training_losses[-1]:.6f}")
            return True
            
        except Exception as e:
            logger.error(f"Training simulation test failed: {e}")
            self.test_results['training_simulation'] = {'status': 'FAIL', 'error': str(e)}
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests"""
        logger.info("Starting comprehensive system test")
        logger.info("=" * 80)
        
        # Set up test environment
        self.setup_test_environment()
        
        # Test item list
        tests = [
            ('Create test data', self.create_test_data),
            ('Memory manager', self.test_memory_manager),
            ('Streaming data loader', self.test_streaming_loader),
            ('Model creation and inference', self.test_model_creation),
            ('Data pipeline integration', self.test_data_pipeline_integration),
            ('Training simulation', self.test_training_simulation)
        ]
        
        # Execute tests
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            logger.info(f"\nExecuting test: {test_name}")
            try:
                if test_func():
                    passed_tests += 1
                    logger.info(f"PASS {test_name}")
                else:
                    logger.error(f"FAIL {test_name}")
            except Exception as e:
                logger.error(f"ERROR {test_name}: {e}")
        
        # Generate test report
        test_summary = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': passed_tests / total_tests,
            'detailed_results': self.test_results
        }
        
        # Output test results
        logger.info("\n" + "=" * 80)
        logger.info("Comprehensive system test completed")
        logger.info("=" * 80)
        logger.info(f"Total tests: {total_tests}")
        logger.info(f"Passed tests: {passed_tests}")
        logger.info(f"Failed tests: {total_tests - passed_tests}")
        logger.info(f"Success rate: {passed_tests / total_tests:.1%}")
        
        # Save test results
        if self.temp_dir:
            results_file = Path(self.temp_dir) / "test_results.json"
            with open(results_file, 'w') as f:
                json.dump(test_summary, f, indent=2, default=str)
            logger.info(f"Test results saved: {results_file}")
        
        return test_summary
    
    def cleanup(self):
        """Clean up test environment"""
        import shutil
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
            logger.info("Test environment cleaned up")


def main():
    """Main function"""
    # Create test instance
    test_runner = ComprehensiveSystemTest()
    
    try:
        # Run all tests
        results = test_runner.run_all_tests()
        
        # Determine overall test result
        if results['success_rate'] >= 0.8:  # 80% pass rate
            print(f"\nSystem tests passed! Success rate: {results['success_rate']:.1%}")
            return 0
        else:
            print(f"\nSystem tests failed to meet requirements! Success rate: {results['success_rate']:.1%}")
            return 1
            
    except Exception as e:
        logger.error(f"Test execution exception: {e}")
        return 1
    finally:
        test_runner.cleanup()


if __name__ == "__main__":
    exit(main())
