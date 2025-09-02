"""
Factor Forecasting System Test Suite
===================================

This package contains comprehensive tests for the Factor Forecasting System.

Test Modules:
- test_models.py: Model architecture tests
- test_training.py: Training pipeline tests
- test_data_processing.py: Data processing tests
- test_integration.py: Integration tests
- test_system.py: System-level tests
- comprehensive_test_suite.py: Complete test suite
- quick_validation.py: Quick validation tests

Usage:
    python -m tests.test_models
    python -m tests.test_training
    python -m tests.test_data_processing
    python -m tests.comprehensive_test_suite
"""

__version__ = "1.0.0"
__author__ = "Factor Forecasting Team"

# Import main test modules
from . import test_models
from . import test_training
from . import test_data_processing
from . import comprehensive_test_suite
from . import quick_validation

__all__ = [
    "test_models",
    "test_training", 
    "test_data_processing",
    "comprehensive_test_suite",
    "quick_validation"
] 