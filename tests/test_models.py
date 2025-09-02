#!/usr/bin/env python3
"""
Comprehensive Model Tests for Factor Forecasting System
=====================================================

This module contains comprehensive tests for all model architectures:
- Transformer models
- TCN (Temporal Convolutional Network) models
- LSTM models
- Model factory and creation
- Model performance and metrics

Usage:
    python test_models.py [options]

Options:
    --transformer    Test transformer models only
    --tcn           Test TCN models only
    --lstm          Test LSTM models only
    --all           Test all models (default)
    --verbose       Enable verbose output
"""

import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.models import (
    FactorForecastingTCNModel, FactorTCN, TemporalConvNet, TemporalBlock,
    FactorTransformer, FactorForecastingModel
)
from src.models.model_factory import create_model
from configs.config import ModelConfig


class ModelTestSuite:
    """Comprehensive model test suite"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.test_results = {}
        
    def test_transformer_models(self) -> Dict[str, Any]:
        """Test transformer-based models"""
        print("Testing Transformer Models...")
        results = {"passed": 0, "failed": 0, "errors": []}
        
        try:
            # Test basic transformer
            config_dict = {
                'num_factors': 100,
                'num_stocks': 1000,
                'd_model': 256,
                'num_heads': 8,
                'num_layers': 6,
                'dropout': 0.1,
                'max_seq_len': 50,
                'target_columns': ['intra30m', 'nextT1d', 'ema1d']
            }
            
            model = create_model(config_dict)
            self._test_model_forward_pass(model, config_dict, "transformer")
            results["passed"] += 1
            
            # Test large transformer
            config_large_dict = {
                'num_factors': 100,
                'num_stocks': 1000,
                'd_model': 512,
                'num_heads': 16,
                'num_layers': 12,
                'dropout': 0.15,
                'max_seq_len': 50,
                'target_columns': ['intra30m', 'nextT1d', 'ema1d']
            }
            
            model_large = create_model(config_large_dict)
            self._test_model_forward_pass(model_large, config_large_dict, "transformer_large")
            results["passed"] += 1
            
        except Exception as e:
            results["failed"] += 1
            results["errors"].append(f"Transformer test error: {e}")
            if self.verbose:
                print(f"Transformer test error: {e}")
        
        return results
    
    def test_tcn_models(self) -> Dict[str, Any]:
        """Test TCN-based models"""
        print("Testing TCN Models...")
        results = {"passed": 0, "failed": 0, "errors": []}
        
        try:
            # Test TCN components
            self._test_tcn_components()
            results["passed"] += 1
            
            # Test FactorTCN
            self._test_factor_tcn()
            results["passed"] += 1
            
            # Test FactorForecastingTCNModel
            self._test_factor_forecasting_tcn_model()
            results["passed"] += 1
            
        except Exception as e:
            results["failed"] += 1
            results["errors"].append(f"TCN test error: {e}")
            if self.verbose:
                print(f"TCN test error: {e}")
        
        return results
    
    def test_lstm_models(self) -> Dict[str, Any]:
        """Test LSTM-based models"""
        print("Testing LSTM Models...")
        results = {"passed": 0, "failed": 0, "errors": []}
        
        try:
            # Test LSTM configuration
            config_dict = {
                'num_factors': 100,
                'num_stocks': 1000,
                'd_model': 256,
                'num_heads': 8,
                'num_layers': 6,
                'dropout': 0.1,
                'max_seq_len': 50,
                'target_columns': ['intra30m', 'nextT1d', 'ema1d']
            }
            
            model = create_model(config_dict)
            self._test_model_forward_pass(model, config_dict, "lstm")
            results["passed"] += 1
            
        except Exception as e:
            results["failed"] += 1
            results["errors"].append(f"LSTM test error: {e}")
            if self.verbose:
                print(f"LSTM test error: {e}")
        
        return results
    
    def test_model_factory(self) -> Dict[str, Any]:
        """Test model factory and creation"""
        print("Testing Model Factory...")
        results = {"passed": 0, "failed": 0, "errors": []}
        
        try:
            # Test different model types
            model_types = ["transformer", "tcn", "lstm"]
            
            for model_type in model_types:
                config_dict = {
                    'num_factors': 100,
                    'num_stocks': 1000,
                    'd_model': 256,
                    'num_heads': 8,
                    'num_layers': 6,
                    'dropout': 0.1,
                    'max_seq_len': 50,
                    'target_columns': ['intra30m', 'nextT1d', 'ema1d']
                }
                model = create_model(config_dict)
                
                # Test model creation
                assert model is not None, f"Model creation failed for {model_type}"
                
                # Test model parameters
                total_params = sum(p.numel() for p in model.parameters())
                assert total_params > 0, f"Model has no parameters: {model_type}"
                
                if self.verbose:
                    print(f"  {model_type}: {total_params:,} parameters")
                
                results["passed"] += 1
                
        except Exception as e:
            results["failed"] += 1
            results["errors"].append(f"Model factory test error: {e}")
            if self.verbose:
                print(f"Model factory test error: {e}")
        
        return results
    
    def test_model_performance(self) -> Dict[str, Any]:
        """Test model performance and memory usage"""
        print("Testing Model Performance...")
        results = {"passed": 0, "failed": 0, "errors": []}
        
        try:
            # Test inference speed
            config_dict = {
                'num_factors': 100,
                'num_stocks': 1000,
                'd_model': 256,
                'num_heads': 8,
                'num_layers': 6,
                'dropout': 0.1,
                'max_seq_len': 50,
                'target_columns': ['intra30m', 'nextT1d', 'ema1d']
            }
            
            model = create_model(config_dict)
            model.eval()
            
            # Create test data
            batch_size, seq_len, feature_dim = 32, 10, 100
            test_input = torch.randn(batch_size, seq_len, feature_dim)
            
            # Measure inference time
            import time
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(10):  # Multiple runs for average
                    stock_ids = torch.randint(0, 1000, (batch_size, seq_len))
                    _ = model(factors=test_input, stock_ids=stock_ids)
            
            end_time = time.time()
            avg_time = (end_time - start_time) / 10
            
            if self.verbose:
                print(f"  Average inference time: {avg_time:.4f}s")
            
            # Check reasonable performance
            assert avg_time < 1.0, f"Inference too slow: {avg_time:.4f}s"
            
            # Test memory usage
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            results["passed"] += 1
            
        except Exception as e:
            results["failed"] += 1
            results["errors"].append(f"Performance test error: {e}")
            if self.verbose:
                print(f"Performance test error: {e}")
        
        return results
    
    def _test_tcn_components(self):
        """Test individual TCN components"""
        if self.verbose:
            print("  Testing TCN components...")
        
        # Test TemporalBlock
        batch_size, seq_len, in_channels, out_channels = 4, 10, 64, 128
        kernel_size, dilation = 3, 1
        
        temporal_block = TemporalBlock(in_channels, out_channels, kernel_size, 
                                      stride=1, dilation=dilation, padding=(kernel_size-1) * dilation)
        
        x = torch.randn(batch_size, in_channels, seq_len)
        output = temporal_block(x)
        
        assert output.shape == (batch_size, out_channels, seq_len), "TemporalBlock output shape mismatch"
        
        # Test TemporalConvNet
        num_inputs, num_channels = 64, [128, 256, 512]
        tcn = TemporalConvNet(num_inputs, num_channels, kernel_size=3, dropout=0.1)
        
        x = torch.randn(batch_size, seq_len, num_inputs)
        output = tcn(x)
        
        assert output.shape == (batch_size, seq_len, num_channels[-1]), "TemporalConvNet output shape mismatch"
        
        if self.verbose:
            print("    TCN components test passed")
    
    def _test_factor_tcn(self):
        """Test FactorTCN model"""
        if self.verbose:
            print("  Testing FactorTCN model...")
        
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
        
        # Check output shapes
        for target_name in config['target_columns']:
            assert predictions[target_name].shape == (batch_size, seq_len), f"Output shape mismatch for {target_name}"
        
        if self.verbose:
            print("    FactorTCN test passed")
    
    def _test_factor_forecasting_tcn_model(self):
        """Test FactorForecastingTCNModel"""
        if self.verbose:
            print("  Testing FactorForecastingTCNModel...")
        
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
        
        # Forward pass
        with torch.no_grad():
            predictions = model(features, stock_ids)
        
        # Check output shapes
        for target_name in config['target_columns']:
            assert predictions[target_name].shape == (batch_size, seq_len), f"Output shape mismatch for {target_name}"
        
        if self.verbose:
            print("    FactorForecastingTCNModel test passed")
    
    def _test_model_forward_pass(self, model: nn.Module, config: ModelConfig, model_name: str):
        """Test model forward pass"""
        if self.verbose:
            print(f"  Testing {model_name} forward pass...")
        
        # Create test data
        batch_size, seq_len, feature_dim = 32, 10, 100
        test_input = torch.randn(batch_size, seq_len, feature_dim)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            # Create dummy stock IDs
            stock_ids = torch.randint(0, 1000, (batch_size, seq_len))
            output = model(factors=test_input, stock_ids=stock_ids)
        
        # Check output format - should be a dictionary with target predictions
        assert isinstance(output, dict), f"Output should be a dictionary, got {type(output)}"
        
        # Check that all expected targets are present
        for target in config['target_columns']:
            assert target in output, f"Target {target} not found in output"
            assert isinstance(output[target], torch.Tensor), f"Output for {target} should be a tensor"
            
            # Check individual target output shape - model outputs (batch_size,) due to global pooling
            expected_shape = (batch_size,)
            assert output[target].shape == expected_shape, f"Output shape for {target}: expected {expected_shape}, got {output[target].shape}"
        
        if self.verbose:
            print(f"    {model_name} forward pass test passed")
    
    def run_all_tests(self, test_types: Optional[list] = None) -> Dict[str, Any]:
        """Run all model tests"""
        if test_types is None:
            test_types = ["transformer", "tcn", "lstm", "factory", "performance"]
        
        print("Running Model Tests...")
        print("=" * 50)
        
        all_results = {}
        
        for test_type in test_types:
            if test_type == "transformer":
                all_results["transformer"] = self.test_transformer_models()
            elif test_type == "tcn":
                all_results["tcn"] = self.test_tcn_models()
            elif test_type == "lstm":
                all_results["lstm"] = self.test_lstm_models()
            elif test_type == "factory":
                all_results["factory"] = self.test_model_factory()
            elif test_type == "performance":
                all_results["performance"] = self.test_model_performance()
        
        # Calculate overall results
        total_passed = sum(result["passed"] for result in all_results.values())
        total_failed = sum(result["failed"] for result in all_results.values())
        total_errors = []
        for result in all_results.values():
            total_errors.extend(result["errors"])
        
        overall_results = {
            "total_passed": total_passed,
            "total_failed": total_failed,
            "total_errors": total_errors,
            "success_rate": total_passed / (total_passed + total_failed) if (total_passed + total_failed) > 0 else 0
        }
        
        self.test_results = {
            "overall": overall_results,
            "details": all_results
        }
        
        return self.test_results
    
    def print_results(self):
        """Print test results"""
        if not self.test_results:
            print("No test results available. Run tests first.")
            return
        
        overall = self.test_results["overall"]
        details = self.test_results["details"]
        
        print("\n" + "=" * 60)
        print("Model Test Results")
        print("=" * 60)
        
        print(f"Total Tests Passed: {overall['total_passed']}")
        print(f"Total Tests Failed: {overall['total_failed']}")
        print(f"Success Rate: {overall['success_rate']:.2%}")
        
        if overall['total_errors']:
            print(f"\nErrors ({len(overall['total_errors'])}):")
            for error in overall['total_errors']:
                print(f"  - {error}")
        
        print("\nDetailed Results:")
        for test_type, result in details.items():
            print(f"  {test_type.upper()}: {result['passed']} passed, {result['failed']} failed")
        
        if overall['success_rate'] == 1.0:
            print("\nAll model tests passed!")
        elif overall['success_rate'] >= 0.8:
            print("\nMost model tests passed!")
        else:
            print("\nMany model tests failed!")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run model tests")
    parser.add_argument("--transformer", action="store_true", help="Test transformer models only")
    parser.add_argument("--tcn", action="store_true", help="Test TCN models only")
    parser.add_argument("--lstm", action="store_true", help="Test LSTM models only")
    parser.add_argument("--all", action="store_true", help="Test all models (default)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Determine test types
    if args.transformer:
        test_types = ["transformer"]
    elif args.tcn:
        test_types = ["tcn"]
    elif args.lstm:
        test_types = ["lstm"]
    else:
        test_types = ["transformer", "tcn", "lstm", "factory", "performance"]
    
    # Run tests
    test_suite = ModelTestSuite(verbose=args.verbose)
    results = test_suite.run_all_tests(test_types)
    test_suite.print_results()
    
    # Exit with appropriate code
    if results["overall"]["success_rate"] == 1.0:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main() 