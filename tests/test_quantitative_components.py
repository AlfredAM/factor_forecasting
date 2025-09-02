#!/usr/bin/env python3
"""
Quantitative Finance Components Test
Tests individual quantitative finance components without complex dependencies
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Test configurations
from configs.config import ModelConfig

def test_configuration():
    """Test quantitative finance configuration"""
    print("Testing Configuration...")
    
    config = ModelConfig()
    
    # Check quantitative finance specific parameters
    required_params = [
        'loss_function_type', 'correlation_weight', 'mse_weight',
        'rank_correlation_weight', 'target_correlations',
        'risk_penalty_weight', 'max_leverage', 'transaction_cost'
    ]
    
    for param in required_params:
        if hasattr(config, param):
            value = getattr(config, param)
            print(f" {param}: {value}")
        else:
            print(f" Missing parameter: {param}")
            return False
    
    return True

def test_quantitative_loss():
    """Test quantitative loss function"""
    print("\nTesting Quantitative Loss Function...")
    
    try:
        from src.training.quantitative_loss import create_quantitative_loss_function, QuantitativeCorrelationLoss
        
        # Create test configuration
        class TestConfig:
            correlation_weight = 1.0
            mse_weight = 0.1
            rank_correlation_weight = 0.2
            risk_penalty_weight = 0.1
            target_correlations = [0.08, 0.05, 0.03]
            max_leverage = 2.0
            transaction_cost = 0.001
            use_adaptive_loss = True
            volatility_window = 20
            regime_sensitivity = 0.1
        
        config = TestConfig()
        
        # Test loss function creation
        loss_fn = create_quantitative_loss_function(config)
        print(" Loss function created successfully")
        
        # Test basic functionality with numpy arrays
        import torch
        batch_size = 50
        
        # Create test data
        predictions = {
            'intra30m': torch.randn(batch_size) * 0.02,
            'nextT1d': torch.randn(batch_size) * 0.015,
            'ema1d': torch.randn(batch_size) * 0.01
        }
        
        targets = {
            'intra30m': torch.randn(batch_size) * 0.025,
            'nextT1d': torch.randn(batch_size) * 0.018,
            'ema1d': torch.randn(batch_size) * 0.012
        }
        
        weights = torch.rand(batch_size)
        
        # Compute loss
        loss_dict = loss_fn(predictions, targets, weights)
        
        print(f" Total loss computed: {loss_dict['total_loss']:.6f}")
        print(" Loss components:")
        for target in ['intra30m', 'nextT1d', 'ema1d']:
            if target in loss_dict['mse_losses']:
                mse = loss_dict['mse_losses'][target]
                corr = loss_dict['correlation_losses'][target]
                print(f"    {target}: MSE={mse:.4f}, Corr={corr:.4f}")
        
        return True
        
    except Exception as e:
        print(f" Error testing quantitative loss: {e}")
        return False

def test_performance_analyzer():
    """Test performance analyzer"""
    print("\nTesting Performance Analyzer...")
    
    try:
        from src.utils.quantitative_metrics import QuantitativePerformanceAnalyzer
        
        # Create analyzer
        analyzer = QuantitativePerformanceAnalyzer(
            target_names=['intra30m', 'nextT1d', 'ema1d'],
            risk_free_rate=0.02,
            transaction_cost=0.001
        )
        print(" Performance analyzer created")
        
        # Generate test data
        n_samples = 100
        
        # Create correlated predictions and targets
        true_factor = np.random.randn(n_samples)
        
        predictions = {
            'intra30m': true_factor * 0.3 + np.random.randn(n_samples) * 0.02,
            'nextT1d': true_factor * 0.2 + np.random.randn(n_samples) * 0.015,
            'ema1d': true_factor * 0.15 + np.random.randn(n_samples) * 0.01
        }
        
        targets = {
            'intra30m': true_factor * 0.25 + np.random.randn(n_samples) * 0.025,
            'nextT1d': true_factor * 0.18 + np.random.randn(n_samples) * 0.018,
            'ema1d': true_factor * 0.12 + np.random.randn(n_samples) * 0.012
        }
        
        dates = [f"2023-01-{i%30+1:02d}" for i in range(n_samples)]
        stock_ids = [f"STOCK_{i%20}" for i in range(n_samples)]
        weights = np.random.uniform(0.5, 2.0, n_samples)
        
        # Add data
        analyzer.add_predictions(predictions, targets, dates, stock_ids, weights)
        print(" Test data added")
        
        # Compute metrics
        metrics = analyzer.compute_comprehensive_metrics()
        print(f" Metrics computed for {len(metrics)} targets")
        
        # Verify results
        for target, metric in metrics.items():
            print(f"    {target}: Correlation={metric.correlation:.4f}, IC_IR={metric.ic_ir:.4f}")
        
        return True
        
    except Exception as e:
        print(f" Error testing performance analyzer: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_risk_manager():
    """Test risk management system"""
    print("\nTesting Risk Manager...")
    
    try:
        from src.utils.risk_management import RiskManager, RiskLimits
        
        # Create risk limits
        risk_limits = RiskLimits(
            max_leverage=2.0,
            max_concentration=0.05,
            max_volatility=0.15,
            max_drawdown=0.10
        )
        
        # Create risk manager
        risk_manager = RiskManager(risk_limits)
        print(" Risk manager created")
        
        # Test data
        n_stocks = 50
        positions = np.random.randn(n_stocks) * 0.02
        positions = positions / np.sum(np.abs(positions)) * 1.5  # 1.5x leverage
        
        returns = np.random.randn(30) * 0.01
        
        # Test position checks
        position_checks = risk_manager.check_position_limits(positions)
        print(f" Position checks: {position_checks}")
        
        # Test risk metrics
        risk_metrics = risk_manager.compute_risk_metrics(positions, returns)
        print(f" Risk metrics computed:")
        print(f"    Leverage: {risk_metrics.leverage:.3f}")
        print(f"    Volatility: {risk_metrics.volatility:.3f}")
        print(f"    Max Drawdown: {risk_metrics.max_drawdown:.3f}")
        
        # Test position adjustment
        adjusted_positions = risk_manager.apply_risk_constraints(positions)
        adjusted_leverage = np.sum(np.abs(adjusted_positions))
        print(f" Position adjustment: {np.sum(np.abs(positions)):.3f} -> {adjusted_leverage:.3f}")
        
        return True
        
    except Exception as e:
        print(f" Error testing risk manager: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_processor():
    """Test quantitative data processor"""
    print("\nTesting Quantitative Data Processor...")
    
    try:
        from src.data_processing.quantitative_data_processor import QuantitativeDataConfig
        
        # Test configuration creation
        config = QuantitativeDataConfig(
            train_start_date="2020-01-01",
            train_end_date="2020-06-30",
            val_start_date="2020-07-01",
            val_end_date="2020-09-30",
            test_start_date="2020-10-01",
            test_end_date="2020-12-31",
            sequence_length=10,
            batch_size=64,
            min_stock_history_days=30
        )
        
        print(" Quantitative data config created")
        print(f"    Training: {config.train_start_date} to {config.train_end_date}")
        print(f"    Validation: {config.val_start_date} to {config.val_end_date}")
        print(f"    Test: {config.test_start_date} to {config.test_end_date}")
        print(f"    Sequence length: {config.sequence_length}")
        print(f"    Min stock history: {config.min_stock_history_days} days")
        
        return True
        
    except Exception as e:
        print(f" Error testing data processor: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 80)
    print("QUANTITATIVE FINANCE COMPONENTS TEST")
    print("=" * 80)
    
    tests = [
        test_configuration,
        test_quantitative_loss,
        test_performance_analyzer,
        test_risk_manager,
        test_data_processor
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f" Test {test_func.__name__} failed: {e}")
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("ALL TESTS PASSED! Quantitative finance system is ready.")
    else:
        print(f"WARNING: {total - passed} tests failed. Please review issues.")
    
    print("=" * 80)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
