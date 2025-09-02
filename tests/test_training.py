#!/usr/bin/env python3
"""
Comprehensive Training Tests for Factor Forecasting System
========================================================

This module contains comprehensive tests for all training components:
- Trainer classes
- Loss functions
- Optimizers and schedulers
- Early stopping
- Metrics tracking
- Rolling window training

Usage:
    python test_training.py [options]

Options:
    --trainer       Test trainer classes only
    --loss          Test loss functions only
    --optimizer     Test optimizers only
    --rolling       Test rolling window training only
    --all           Test all components (default)
    --verbose       Enable verbose output
"""

import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import tempfile
import shutil
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.training.trainer import (
    FactorForecastingTrainer, 
    CorrelationLoss, 
    EarlyStopping, 
    LearningRateScheduler,
    MetricsTracker,
    create_trainer
)
from src.training.rolling_train import RollingWindowTrainer
from src.models.model_factory import create_model
from configs.config import ModelConfig


class TrainingTestSuite:
    """Comprehensive training test suite"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.test_results = {}
        self.temp_dir = None
        
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test environment"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_loss_functions(self) -> Dict[str, Any]:
        """Test loss functions"""
        print("Testing Loss Functions...")
        results = {"passed": 0, "failed": 0, "errors": []}
        
        try:
            # Test correlation loss
            loss_fn = CorrelationLoss(
                correlation_weight=1.0,
                mse_weight=0.1,
                rank_weight=0.1,
                target_correlations=[0.1, 0.05, 0.08]
            )
            
            # Create dummy predictions and targets
            batch_size = 32
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
            
            # Check that loss is a scalar tensor
            assert torch.is_tensor(loss), "Loss should be a tensor"
            assert loss.dim() == 0, "Loss should be a scalar"
            assert loss.item() > 0, "Loss should be positive"
            
            results["passed"] += 1
            
            if self.verbose:
                print("  Correlation loss test passed")
                
        except Exception as e:
            results["failed"] += 1
            results["errors"].append(f"Loss function test error: {e}")
            if self.verbose:
                print(f"  Loss function test error: {e}")
        
        return results
    
    def test_early_stopping(self) -> Dict[str, Any]:
        """Test early stopping mechanism"""
        print("Testing Early Stopping...")
        results = {"passed": 0, "failed": 0, "errors": []}
        
        try:
            self.setUp()
            
            early_stopping = EarlyStopping(
                patience=3,
                min_delta=0.001,
                restore_best_weights=True,
                checkpoint_dir=self.temp_dir
            )
            model = nn.Linear(10, 1)
            
            # Test with improving loss
            losses = [1.0, 0.9, 0.8, 0.7, 0.6]
            
            for i, loss in enumerate(losses):
                should_stop = early_stopping(loss, model, i)
                if i < len(losses) - 1:  # Should not stop before patience is reached
                    assert not should_stop, "Should not stop with improving loss"
            
            results["passed"] += 1
            
            # Test with no improvement
            early_stopping_no_improve = EarlyStopping(
                patience=3,
                min_delta=0.001,
                restore_best_weights=True,
                checkpoint_dir=self.temp_dir
            )
            
            losses_no_improve = [1.0, 1.1, 1.2, 1.3, 1.4]  # No improvement
            
            for i, loss in enumerate(losses_no_improve):
                should_stop = early_stopping_no_improve(loss, model, i)
                if i >= 3:  # Should stop after patience
                    assert should_stop, "Should stop with no improvement"
                    break
            
            results["passed"] += 1
            
            if self.verbose:
                print("  Early stopping test passed")
                
        except Exception as e:
            results["failed"] += 1
            results["errors"].append(f"Early stopping test error: {e}")
            if self.verbose:
                print(f"  Early stopping test error: {e}")
        finally:
            self.tearDown()
        
        return results
    
    def test_learning_rate_scheduler(self) -> Dict[str, Any]:
        """Test learning rate scheduler"""
        print("Testing Learning Rate Scheduler...")
        results = {"passed": 0, "failed": 0, "errors": []}
        
        try:
            # Test cosine scheduler
            optimizer = torch.optim.Adam([torch.randn(10, requires_grad=True)], lr=1e-3)
            scheduler = LearningRateScheduler(
                optimizer=optimizer,
                scheduler_type="cosine",
                warmup_steps=100,
                total_steps=1000
            )
            
            # Test scheduler step
            for step in range(10):
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                assert current_lr >= 0, "Learning rate should be non-negative"
            
            results["passed"] += 1
            
            if self.verbose:
                print("  Learning rate scheduler test passed")
                
        except Exception as e:
            results["failed"] += 1
            results["errors"].append(f"Learning rate scheduler test error: {e}")
            if self.verbose:
                print(f"  Learning rate scheduler test error: {e}")
        
        return results
    
    def test_metrics_tracker(self) -> Dict[str, Any]:
        """Test metrics tracker"""
        print("Testing Metrics Tracker...")
        results = {"passed": 0, "failed": 0, "errors": []}
        
        try:
            tracker = MetricsTracker(['intra30m', 'nextT1d', 'ema1d'])
            
            # Test metrics update
            predictions = {
                'intra30m': torch.randn(10),
                'nextT1d': torch.randn(10),
                'ema1d': torch.randn(10)
            }
            targets = {
                'intra30m': torch.randn(10),
                'nextT1d': torch.randn(10),
                'ema1d': torch.randn(10)
            }
            
            tracker.update(predictions, targets)
            
            # Check that metrics are stored
            assert len(tracker.predictions['intra30m']) == 10, "Predictions not stored correctly"
            assert len(tracker.targets['intra30m']) == 10, "Targets not stored correctly"
            
            # Test metrics reset
            tracker.reset()
            assert len(tracker.predictions['intra30m']) == 0, "Metrics not reset correctly"
            
            results["passed"] += 1
            
            if self.verbose:
                print("  Metrics tracker test passed")
                
        except Exception as e:
            results["failed"] += 1
            results["errors"].append(f"Metrics tracker test error: {e}")
            if self.verbose:
                print(f"  Metrics tracker test error: {e}")
        
        return results
    
    def test_trainer(self) -> Dict[str, Any]:
        """Test trainer class"""
        print("Testing Trainer...")
        results = {"passed": 0, "failed": 0, "errors": []}
        
        try:
            self.setUp()
            
            # Create configuration
            config_dict = {
                'num_factors': 100,
                'num_stocks': 1000,
                'd_model': 128,  # Smaller for testing
                'num_heads': 4,
                'num_layers': 2,
                'dropout': 0.1,
                'max_seq_len': 10,
                'target_columns': ['intra30m', 'nextT1d', 'ema1d']
            }
            
            # Create model
            model = create_model(config_dict)
            
            # Create trainer
            trainer = FactorForecastingTrainer(
                model=model,
                config=config_dict,
                loss_fn=CorrelationLoss()
            )
            
            # Test trainer initialization
            assert trainer.model is not None, "Model not initialized"
            assert trainer.optimizer is not None, "Optimizer not initialized"
            assert trainer.loss_fn is not None, "Loss function not initialized"
            
            results["passed"] += 1
            
            if self.verbose:
                print("  Trainer initialization test passed")
                
        except Exception as e:
            results["failed"] += 1
            results["errors"].append(f"Trainer test error: {e}")
            if self.verbose:
                print(f"  Trainer test error: {e}")
        finally:
            self.tearDown()
        
        return results
    
    def test_rolling_window_training(self) -> Dict[str, Any]:
        """Test rolling window training"""
        print("Testing Rolling Window Training...")
        results = {"passed": 0, "failed": 0, "errors": []}
        
        try:
            self.setUp()
            
            # Create sample data
            data = self._create_sample_data()
            
            # Create rolling window trainer
            config = ModelConfig(
                model_type="transformer",
                hidden_size=128,
                num_layers=2,
                num_heads=4,
                batch_size=16,
                learning_rate=1e-3,
                num_epochs=1,  # Short for testing
                rolling_window_years=1,
                min_train_years=1,
                prediction_years=[2020]
            )
            
            trainer = RollingWindowTrainer(config)
            
            # Test trainer creation
            assert trainer is not None, "Rolling window trainer not created"
            
            results["passed"] += 1
            
            if self.verbose:
                print("  Rolling window trainer test passed")
                
        except Exception as e:
            results["failed"] += 1
            results["errors"].append(f"Rolling window training test error: {e}")
            if self.verbose:
                print(f"  Rolling window training test error: {e}")
        finally:
            self.tearDown()
        
        return results
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample data for testing"""
        np.random.seed(42)
        
        # Create sample data
        num_stocks = 10
        num_days = 5
        num_factors = 100
        
        data = []
        for stock_id in range(num_stocks):
            for day in range(num_days):
                row = {
                    'sid': f'stock_{stock_id}',
                    'date': f'2020-01-{day+1:02d}',
                    'intra30m': np.random.randn(),
                    'nextT1d': np.random.randn(),
                    'ema1d': np.random.randn(),
                    'ADV50': np.random.uniform(1000, 10000),
                    'luld': np.random.choice([0, 1])
                }
                
                # Add factor columns
                for i in range(num_factors):
                    row[str(i)] = np.random.randn()
                
                data.append(row)
        
        return pd.DataFrame(data)
    
    def run_all_tests(self, test_types: Optional[list] = None) -> Dict[str, Any]:
        """Run all training tests"""
        if test_types is None:
            test_types = ["loss", "early_stopping", "scheduler", "metrics", "trainer", "rolling"]
        
        print("Running Training Tests...")
        print("=" * 50)
        
        all_results = {}
        
        for test_type in test_types:
            if test_type == "loss":
                all_results["loss"] = self.test_loss_functions()
            elif test_type == "early_stopping":
                all_results["early_stopping"] = self.test_early_stopping()
            elif test_type == "scheduler":
                all_results["scheduler"] = self.test_learning_rate_scheduler()
            elif test_type == "metrics":
                all_results["metrics"] = self.test_metrics_tracker()
            elif test_type == "trainer":
                all_results["trainer"] = self.test_trainer()
            elif test_type == "rolling":
                all_results["rolling"] = self.test_rolling_window_training()
        
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
        print("Training Test Results")
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
            print("\nAll training tests passed!")
        elif overall['success_rate'] >= 0.8:
            print("\nMost training tests passed!")
        else:
            print("\nMany training tests failed!")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run training tests")
    parser.add_argument("--loss", action="store_true", help="Test loss functions only")
    parser.add_argument("--early-stopping", action="store_true", help="Test early stopping only")
    parser.add_argument("--scheduler", action="store_true", help="Test schedulers only")
    parser.add_argument("--metrics", action="store_true", help="Test metrics only")
    parser.add_argument("--trainer", action="store_true", help="Test trainer only")
    parser.add_argument("--rolling", action="store_true", help="Test rolling window training only")
    parser.add_argument("--all", action="store_true", help="Test all components (default)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Determine test types
    test_types = []
    if args.loss:
        test_types.append("loss")
    if args.early_stopping:
        test_types.append("early_stopping")
    if args.scheduler:
        test_types.append("scheduler")
    if args.metrics:
        test_types.append("metrics")
    if args.trainer:
        test_types.append("trainer")
    if args.rolling:
        test_types.append("rolling")
    
    if not test_types:  # Default to all
        test_types = ["loss", "early_stopping", "scheduler", "metrics", "trainer", "rolling"]
    
    # Run tests
    test_suite = TrainingTestSuite(verbose=args.verbose)
    results = test_suite.run_all_tests(test_types)
    test_suite.print_results()
    
    # Exit with appropriate code
    if results["overall"]["success_rate"] == 1.0:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main() 