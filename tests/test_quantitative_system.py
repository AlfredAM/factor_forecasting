#!/usr/bin/env python3
"""
Comprehensive Test Suite for Quantitative Finance System
Tests all quantitative finance components and their integration
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import tempfile
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import test modules
from configs.config import ModelConfig
from src.training.quantitative_loss import create_quantitative_loss_function, QuantitativeCorrelationLoss
from src.utils.quantitative_metrics import QuantitativePerformanceAnalyzer
from src.utils.risk_management import RiskManager, RiskLimits
from src.data_processing.quantitative_data_processor import QuantitativeDataConfig


class TestQuantitativeSystem(unittest.TestCase):
    """Comprehensive test suite for quantitative finance system"""
    
    def setUp(self):
        """Set up test fixtures"""
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Suppress warnings for cleaner test output
        warnings.filterwarnings('ignore')
        
        # Create test configuration
        self.config = ModelConfig()
        self.config.loss_function_type = 'quantitative_correlation'
        self.config.correlation_weight = 1.0
        self.config.mse_weight = 0.1
        self.config.rank_correlation_weight = 0.2
        self.config.risk_penalty_weight = 0.1
        self.config.target_correlations = [0.08, 0.05, 0.03]
        
        # Generate test data
        self.batch_size = 100
        self.n_features = 50
        self.target_names = ['intra30m', 'nextT1d', 'ema1d']
        
        # Create synthetic data with realistic financial characteristics
        self._generate_test_data()
    
    def _generate_test_data(self):
        """Generate realistic financial test data"""
        # True underlying factor
        true_factor = np.random.randn(self.batch_size)
        
        # Add noise with different characteristics for each target
        self.predictions = {
            'intra30m': true_factor * 0.3 + np.random.randn(self.batch_size) * 0.02,
            'nextT1d': true_factor * 0.2 + np.random.randn(self.batch_size) * 0.015,
            'ema1d': true_factor * 0.15 + np.random.randn(self.batch_size) * 0.01
        }
        
        self.targets = {
            'intra30m': true_factor * 0.25 + np.random.randn(self.batch_size) * 0.025,
            'nextT1d': true_factor * 0.18 + np.random.randn(self.batch_size) * 0.018,
            'ema1d': true_factor * 0.12 + np.random.randn(self.batch_size) * 0.012
        }
        
        # Convert to tensors
        self.pred_tensors = {k: torch.tensor(v, dtype=torch.float32) 
                           for k, v in self.predictions.items()}
        self.target_tensors = {k: torch.tensor(v, dtype=torch.float32) 
                             for k, v in self.targets.items()}
        
        # Sample weights (market cap weights)
        self.weights = torch.rand(self.batch_size)
        
        # Time series data
        self.dates = [f"2023-01-{i%30+1:02d}" for i in range(self.batch_size)]
        self.stock_ids = [f"STOCK_{i%20}" for i in range(self.batch_size)]
        
        # Portfolio data
        self.positions = np.random.randn(self.batch_size) * 0.01
        self.positions = self.positions / np.sum(np.abs(self.positions)) * 1.5  # 1.5x leverage
        
        self.returns = np.random.randn(50) * 0.01  # 50 days of returns
        self.returns[20:25] = -0.03  # Simulate drawdown
    
    def test_quantitative_loss_function(self):
        """Test quantitative correlation loss function"""
        print("\n" + "="*60)
        print("Testing Quantitative Loss Function")
        print("="*60)
        
        # Test loss function creation
        loss_fn = create_quantitative_loss_function(self.config)
        self.assertIsNotNone(loss_fn, "Loss function should be created successfully")
        
        # Test loss computation
        loss_dict = loss_fn(self.pred_tensors, self.target_tensors, self.weights)
        
        # Verify loss components
        self.assertIn('total_loss', loss_dict, "Total loss should be present")
        self.assertIn('mse_losses', loss_dict, "MSE losses should be present")
        self.assertIn('correlation_losses', loss_dict, "Correlation losses should be present")
        self.assertIn('rank_losses', loss_dict, "Rank losses should be present")
        self.assertIn('risk_penalties', loss_dict, "Risk penalties should be present")
        
        # Check loss values are reasonable
        total_loss = loss_dict['total_loss']
        self.assertGreater(total_loss, 0, "Total loss should be positive")
        self.assertLess(total_loss, 10, "Total loss should be reasonable")
        
        # Test individual target losses
        for target in self.target_names:
            self.assertIn(target, loss_dict['mse_losses'], f"MSE loss for {target} should exist")
            self.assertIn(target, loss_dict['correlation_losses'], f"Correlation loss for {target} should exist")
            
            mse_loss = loss_dict['mse_losses'][target]
            self.assertGreater(mse_loss, 0, f"MSE loss for {target} should be positive")
            self.assertLess(mse_loss, 1, f"MSE loss for {target} should be reasonable")
        
        print(f" Total Loss: {total_loss:.6f}")
        for target in self.target_names:
            mse = loss_dict['mse_losses'][target]
            corr = loss_dict['correlation_losses'][target]
            rank = loss_dict['rank_losses'][target]
            risk = loss_dict['risk_penalties'][target]
            print(f" {target}: MSE={mse:.4f}, Corr={corr:.4f}, Rank={rank:.4f}, Risk={risk:.4f}")
    
    def test_performance_analyzer(self):
        """Test quantitative performance analyzer"""
        print("\n" + "="*60)
        print("Testing Performance Analyzer")
        print("="*60)
        
        # Create analyzer
        analyzer = QuantitativePerformanceAnalyzer(
            target_names=self.target_names,
            risk_free_rate=0.02,
            transaction_cost=0.001
        )
        
        # Add test data
        # Convert tensor to numpy safely
        weights_numpy = self.weights.detach().cpu().numpy() if hasattr(self.weights, 'numpy') else np.array(self.weights)
        
        analyzer.add_predictions(
            self.predictions, 
            self.targets, 
            self.dates, 
            self.stock_ids, 
            weights_numpy
        )
        
        # Compute metrics
        metrics = analyzer.compute_comprehensive_metrics()
        
        # Verify metrics for each target
        for target in self.target_names:
            self.assertIn(target, metrics, f"Metrics for {target} should exist")
            
            target_metrics = metrics[target]
            
            # Check basic metrics
            self.assertIsNotNone(target_metrics.correlation, f"Correlation for {target} should exist")
            self.assertIsNotNone(target_metrics.rank_ic, f"Rank IC for {target} should exist")
            self.assertGreater(target_metrics.mse, 0, f"MSE for {target} should be positive")
            self.assertGreater(target_metrics.mae, 0, f"MAE for {target} should be positive")
            
            # Check correlation range
            self.assertGreaterEqual(target_metrics.correlation, -1, f"Correlation for {target} should be >= -1")
            self.assertLessEqual(target_metrics.correlation, 1, f"Correlation for {target} should be <= 1")
            
            # Check IC statistics
            self.assertIsNotNone(target_metrics.ic_mean, f"IC mean for {target} should exist")
            self.assertIsNotNone(target_metrics.ic_std, f"IC std for {target} should exist")
            self.assertIsNotNone(target_metrics.ic_ir, f"IC IR for {target} should exist")
            
            print(f" {target}:")
            print(f"    Correlation: {target_metrics.correlation:.4f}")
            print(f"    Rank IC: {target_metrics.rank_ic:.4f}")
            print(f"    MSE: {target_metrics.mse:.6f}")
            print(f"    IC IR: {target_metrics.ic_ir:.4f}")
            print(f"    Sharpe Ratio: {target_metrics.sharpe_ratio:.4f}")
        
        # Generate report
        report = analyzer.generate_performance_report(metrics)
        self.assertIsInstance(report, str, "Report should be a string")
        self.assertGreater(len(report), 100, "Report should have substantial content")
        
        print(" Performance report generated successfully")
    
    def test_risk_manager(self):
        """Test risk management system"""
        print("\n" + "="*60)
        print("Testing Risk Management System")
        print("="*60)
        
        # Create risk limits
        risk_limits = RiskLimits(
            max_leverage=2.0,
            max_concentration=0.05,
            max_volatility=0.15,
            max_drawdown=0.10,
            min_ic=0.01
        )
        
        # Create risk manager
        risk_manager = RiskManager(risk_limits)
        
        # Test position limit checks
        position_checks = risk_manager.check_position_limits(self.positions)
        self.assertIsInstance(position_checks, dict, "Position checks should return dict")
        self.assertIn('leverage_ok', position_checks, "Leverage check should exist")
        self.assertIn('concentration_ok', position_checks, "Concentration check should exist")
        
        print(f" Position Checks: {position_checks}")
        
        # Test performance limit checks
        performance_checks = risk_manager.check_performance_limits(
            self.returns, 
            np.array(list(self.predictions.values())[0]),
            np.array(list(self.targets.values())[0])
        )
        self.assertIsInstance(performance_checks, dict, "Performance checks should return dict")
        
        print(f" Performance Checks: {performance_checks}")
        
        # Test risk metrics computation
        risk_metrics = risk_manager.compute_risk_metrics(self.positions, self.returns)
        
        # Verify risk metrics
        self.assertGreater(risk_metrics.leverage, 0, "Leverage should be positive")
        self.assertGreater(risk_metrics.concentration, 0, "Concentration should be positive")
        self.assertGreaterEqual(risk_metrics.volatility, 0, "Volatility should be non-negative")
        self.assertGreaterEqual(risk_metrics.max_drawdown, 0, "Max drawdown should be non-negative")
        
        print(f" Risk Metrics:")
        print(f"    Leverage: {risk_metrics.leverage:.3f}")
        print(f"    Concentration: {risk_metrics.concentration:.3f}")
        print(f"    Volatility: {risk_metrics.volatility:.3f}")
        print(f"    Max Drawdown: {risk_metrics.max_drawdown:.3f}")
        print(f"    Sharpe Ratio: {risk_metrics.sharpe_ratio:.3f}")
        
        # Test position adjustment
        adjusted_positions = risk_manager.apply_risk_constraints(self.positions)
        adjusted_leverage = np.sum(np.abs(adjusted_positions))
        
        self.assertLessEqual(adjusted_leverage, risk_limits.max_leverage + 1e-6, 
                           "Adjusted leverage should respect limits")
        
        print(f" Position Adjustment:")
        print(f"    Original leverage: {np.sum(np.abs(self.positions)):.3f}")
        print(f"    Adjusted leverage: {adjusted_leverage:.3f}")
        
        # Test risk report generation
        all_checks = {**position_checks, **performance_checks}
        risk_report = risk_manager.generate_risk_report(risk_metrics, all_checks)
        
        self.assertIsInstance(risk_report, str, "Risk report should be a string")
        self.assertGreater(len(risk_report), 200, "Risk report should have substantial content")
        
        print(" Risk report generated successfully")
    
    def test_integrated_system(self):
        """Test integrated quantitative finance system"""
        print("\n" + "="*60)
        print("Testing Integrated System")
        print("="*60)
        
        # Test complete workflow
        try:
            # 1. Create loss function
            loss_fn = create_quantitative_loss_function(self.config)
            print(" Loss function created")
            
            # 2. Compute loss
            loss_dict = loss_fn(self.pred_tensors, self.target_tensors, self.weights)
            total_loss = loss_dict['total_loss']
            print(f" Loss computed: {total_loss:.6f}")
            
            # 3. Analyze performance
            analyzer = QuantitativePerformanceAnalyzer()
            analyzer.add_predictions(self.predictions, self.targets, self.dates, self.stock_ids)
            metrics = analyzer.compute_comprehensive_metrics()
            print(f" Performance metrics computed for {len(metrics)} targets")
            
            # 4. Manage risk
            risk_manager = RiskManager()
            risk_metrics = risk_manager.compute_risk_metrics(self.positions, self.returns)
            adjusted_positions = risk_manager.apply_risk_constraints(self.positions)
            print(" Risk management applied")
            
            # 5. Generate reports
            perf_report = analyzer.generate_performance_report(metrics)
            risk_report = risk_manager.generate_risk_report(risk_metrics, {})
            print(" Reports generated")
            
            # Verify integration
            self.assertGreater(total_loss, 0, "Loss should be positive")
            self.assertGreater(len(metrics), 0, "Should have performance metrics")
            self.assertIsNotNone(risk_metrics, "Should have risk metrics")
            self.assertEqual(len(adjusted_positions), len(self.positions), "Positions should be same length")
            
            print(" Integrated system test passed")
            
        except Exception as e:
            self.fail(f"Integrated system test failed: {str(e)}")
    
    def test_configuration_compatibility(self):
        """Test configuration compatibility"""
        print("\n" + "="*60)
        print("Testing Configuration Compatibility")
        print("="*60)
        
        # Test ModelConfig compatibility
        config = ModelConfig()
        
        # Test quantitative data config
        quant_config = QuantitativeDataConfig()
        self.assertIsNotNone(quant_config, "Quantitative data config should be created")
        
        # Test configuration parameters
        required_params = [
            'loss_function_type', 'correlation_weight', 'mse_weight',
            'rank_correlation_weight', 'target_correlations'
        ]
        
        for param in required_params:
            self.assertTrue(hasattr(config, param), f"Config should have {param}")
            print(f" {param}: {getattr(config, param)}")
        
        # Test quantitative-specific parameters
        quant_params = [
            'risk_penalty_weight', 'max_leverage', 'transaction_cost',
            'use_adaptive_loss', 'volatility_window'
        ]
        
        for param in quant_params:
            self.assertTrue(hasattr(config, param), f"Config should have quantitative param {param}")
            print(f" {param}: {getattr(config, param)}")
        
        print(" Configuration compatibility verified")
    
    def test_error_handling(self):
        """Test error handling and edge cases"""
        print("\n" + "="*60)
        print("Testing Error Handling")
        print("="*60)
        
        # Test with empty data
        empty_predictions = {target: np.array([]) for target in self.target_names}
        empty_targets = {target: np.array([]) for target in self.target_names}
        
        analyzer = QuantitativePerformanceAnalyzer()
        analyzer.add_predictions(empty_predictions, empty_targets)
        metrics = analyzer.compute_comprehensive_metrics()
        
        self.assertEqual(len(metrics), 0, "Should handle empty data gracefully")
        print(" Empty data handled gracefully")
        
        # Test with NaN data
        nan_predictions = {target: np.full(10, np.nan) for target in self.target_names}
        nan_targets = {target: np.full(10, np.nan) for target in self.target_names}
        
        analyzer = QuantitativePerformanceAnalyzer()
        analyzer.add_predictions(nan_predictions, nan_targets)
        metrics = analyzer.compute_comprehensive_metrics()
        
        self.assertEqual(len(metrics), 0, "Should handle NaN data gracefully")
        print(" NaN data handled gracefully")
        
        # Test risk manager with edge cases
        risk_manager = RiskManager()
        
        # Test with zero positions
        zero_positions = np.zeros(10)
        checks = risk_manager.check_position_limits(zero_positions)
        self.assertTrue(all(checks.values()), "Zero positions should pass all checks")
        print(" Zero positions handled correctly")
        
        # Test with extreme positions
        extreme_positions = np.array([1.0, -1.0] + [0.0] * 8)  # 200% leverage
        adjusted = risk_manager.apply_risk_constraints(extreme_positions)
        adjusted_leverage = np.sum(np.abs(adjusted))
        self.assertLessEqual(adjusted_leverage, 2.1, "Extreme positions should be constrained")
        print(" Extreme positions constrained correctly")
        
        print(" Error handling tests passed")


def run_comprehensive_test():
    """Run comprehensive test suite"""
    print("=" * 80)
    print("QUANTITATIVE FINANCE SYSTEM COMPREHENSIVE TEST")
    print("=" * 80)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestQuantitativeSystem)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    successful = total_tests - failures - errors
    
    print(f"Total Tests: {total_tests}")
    print(f"Successful: {successful}")
    print(f"Failures: {failures}")
    print(f"Errors: {errors}")
    
    if failures == 0 and errors == 0:
        print("\nALL TESTS PASSED! Quantitative finance system is ready for production.")
    else:
        print(f"\nWARNING: {failures + errors} tests failed. Please review and fix issues.")
        
        if result.failures:
            print("\nFAILURES:")
            for test, traceback in result.failures:
                print(f"- {test}: {traceback}")
        
        if result.errors:
            print("\nERRORS:")
            for test, traceback in result.errors:
                print(f"- {test}: {traceback}")
    
    print("=" * 80)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
