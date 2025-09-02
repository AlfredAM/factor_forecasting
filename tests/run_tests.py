#!/usr/bin/env python3
"""
Unified Test Runner for Factor Forecasting System
================================================

This script provides a unified interface to run all tests in the Factor Forecasting System.
It can run individual test modules or the complete test suite.

Usage:
    python tests/run_tests.py [options]

Options:
    --models           Run model tests only
    --training         Run training tests only
    --data             Run data processing tests only
    --integration      Run integration tests only
    --system           Run system tests only
    --quick            Run quick validation tests only
    --comprehensive    Run comprehensive test suite only
    --all              Run all tests (default)
    --verbose          Enable verbose output
    --parallel         Run tests in parallel
    --save-results     Save test results to file
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from tests import test_models, test_training, test_data_processing, comprehensive_test_suite, quick_validation


class UnifiedTestRunner:
    """Unified test runner for Factor Forecasting System"""
    
    def __init__(self, verbose: bool = False, parallel: bool = False):
        self.verbose = verbose
        self.parallel = parallel
        self.test_results = {}
        self.start_time = time.time()
        
    def run_model_tests(self) -> Dict[str, Any]:
        """Run model tests"""
        print("Running Model Tests...")
        print("-" * 40)
        
        try:
            test_suite = test_models.ModelTestSuite(verbose=self.verbose)
            results = test_suite.run_all_tests()
            return results
        except Exception as e:
            return {
                "overall": {
                    "total_passed": 0,
                    "total_failed": 1,
                    "total_errors": [f"Model test error: {e}"],
                    "success_rate": 0.0
                },
                "details": {}
            }
    
    def run_training_tests(self) -> Dict[str, Any]:
        """Run training tests"""
        print("Running Training Tests...")
        print("-" * 40)
        
        try:
            test_suite = test_training.TrainingTestSuite(verbose=self.verbose)
            results = test_suite.run_all_tests()
            return results
        except Exception as e:
            return {
                "overall": {
                    "total_passed": 0,
                    "total_failed": 1,
                    "total_errors": [f"Training test error: {e}"],
                    "success_rate": 0.0
                },
                "details": {}
            }
    
    def run_data_processing_tests(self) -> Dict[str, Any]:
        """Run data processing tests"""
        print("Running Data Processing Tests...")
        print("-" * 40)
        
        try:
            test_suite = test_data_processing.DataProcessingTestSuite(verbose=self.verbose)
            results = test_suite.run_all_tests()
            return results
        except Exception as e:
            return {
                "overall": {
                    "total_passed": 0,
                    "total_failed": 1,
                    "total_errors": [f"Data processing test error: {e}"],
                    "success_rate": 0.0
                },
                "details": {}
            }
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run comprehensive test suite"""
        print("Running Comprehensive Test Suite...")
        print("-" * 40)
        
        try:
            test_suite = comprehensive_test_suite.ComprehensiveTestSuite(verbose=self.verbose)
            results = test_suite.run_all_tests()
            return results
        except Exception as e:
            return {
                "overall": {
                    "total_passed": 0,
                    "total_failed": 1,
                    "total_errors": [f"Comprehensive test error: {e}"],
                    "success_rate": 0.0
                },
                "details": {}
            }
    
    def run_quick_validation(self) -> Dict[str, Any]:
        """Run quick validation tests"""
        print("Running Quick Validation Tests...")
        print("-" * 40)
        
        try:
            # Import and run quick validation
            import quick_validation
            results = quick_validation.run_quick_validation()
            return results
        except Exception as e:
            return {
                "overall": {
                    "total_passed": 0,
                    "total_failed": 1,
                    "total_errors": [f"Quick validation error: {e}"],
                    "success_rate": 0.0
                },
                "details": {}
            }
    
    def run_all_tests(self, test_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run all specified tests"""
        if test_types is None:
            test_types = ["models", "training", "data", "comprehensive", "quick"]
        
        print("Factor Forecasting System - Unified Test Runner")
        print("=" * 60)
        print(f"Test Types: {', '.join(test_types)}")
        print(f"Verbose: {self.verbose}")
        print(f"Parallel: {self.parallel}")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        all_results = {}
        
        for test_type in test_types:
            if test_type == "models":
                all_results["models"] = self.run_model_tests()
            elif test_type == "training":
                all_results["training"] = self.run_training_tests()
            elif test_type == "data":
                all_results["data_processing"] = self.run_data_processing_tests()
            elif test_type == "comprehensive":
                all_results["comprehensive"] = self.run_comprehensive_tests()
            elif test_type == "quick":
                all_results["quick_validation"] = self.run_quick_validation()
        
        # Calculate overall results
        total_passed = sum(result["overall"]["total_passed"] for result in all_results.values())
        total_failed = sum(result["overall"]["total_failed"] for result in all_results.values())
        total_errors = []
        for result in all_results.values():
            total_errors.extend(result["overall"]["total_errors"])
        
        execution_time = time.time() - self.start_time
        
        overall_results = {
            "total_passed": total_passed,
            "total_failed": total_failed,
            "total_errors": total_errors,
            "success_rate": total_passed / (total_passed + total_failed) if (total_passed + total_failed) > 0 else 0,
            "execution_time": execution_time
        }
        
        self.test_results = {
            "overall": overall_results,
            "details": all_results,
            "metadata": {
                "test_types": test_types,
                "verbose": self.verbose,
                "parallel": self.parallel,
                "start_time": self.start_time,
                "end_time": time.time()
            }
        }
        
        return self.test_results
    
    def print_results(self):
        """Print comprehensive test results"""
        if not self.test_results:
            print("No test results available. Run tests first.")
            return
        
        overall = self.test_results["overall"]
        details = self.test_results["details"]
        metadata = self.test_results["metadata"]
        
        print("\n" + "=" * 80)
        print("Factor Forecasting System - Test Results Summary")
        print("=" * 80)
        
        print(f"Execution Time: {overall['execution_time']:.2f} seconds")
        print(f"Total Tests Passed: {overall['total_passed']}")
        print(f"Total Tests Failed: {overall['total_failed']}")
        print(f"Overall Success Rate: {overall['success_rate']:.2%}")
        
        if overall['total_errors']:
            print(f"\nErrors ({len(overall['total_errors'])}):")
            for error in overall['total_errors']:
                print(f"  - {error}")
        
        print("\nDetailed Results by Test Type:")
        print("-" * 40)
        for test_type, result in details.items():
            overall_result = result["overall"]
            print(f"{test_type.upper():<20} | "
                  f"Passed: {overall_result['total_passed']:>3} | "
                  f"Failed: {overall_result['total_failed']:>3} | "
                  f"Rate: {overall_result['success_rate']:>6.1%}")
        
        print("-" * 40)
        
        # Overall assessment
        if overall['success_rate'] == 1.0:
            print("\nAll tests passed successfully!")
        elif overall['success_rate'] >= 0.9:
            print("\nMost tests passed with minor issues.")
        elif overall['success_rate'] >= 0.8:
            print("\nMost tests passed but some issues detected.")
        elif overall['success_rate'] >= 0.6:
            print("\nMany tests failed, review required.")
        else:
            print("\nCritical test failures detected!")
        
        print("=" * 80)
    
    def save_results(self, filename: Optional[str] = None):
        """Save test results to file"""
        if not self.test_results:
            print("No test results to save.")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_results_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.test_results, f, indent=2, default=str)
            print(f"Test results saved to: {filename}")
        except Exception as e:
            print(f"Error saving test results: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Unified Test Runner for Factor Forecasting System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/run_tests.py --all                    # Run all tests
  python tests/run_tests.py --models --training      # Run specific tests
  python tests/run_tests.py --comprehensive          # Run comprehensive suite
  python tests/run_tests.py --quick --verbose        # Quick validation with verbose output
        """
    )
    
    # Test type options
    parser.add_argument("--models", action="store_true", help="Run model tests only")
    parser.add_argument("--training", action="store_true", help="Run training tests only")
    parser.add_argument("--data", action="store_true", help="Run data processing tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--system", action="store_true", help="Run system tests only")
    parser.add_argument("--quick", action="store_true", help="Run quick validation tests only")
    parser.add_argument("--comprehensive", action="store_true", help="Run comprehensive test suite only")
    parser.add_argument("--all", action="store_true", help="Run all tests (default)")
    
    # Execution options
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--save-results", action="store_true", help="Save test results to file")
    
    args = parser.parse_args()
    
    # Determine test types
    test_types = []
    if args.models:
        test_types.append("models")
    if args.training:
        test_types.append("training")
    if args.data:
        test_types.append("data")
    if args.integration:
        test_types.append("integration")
    if args.system:
        test_types.append("system")
    if args.quick:
        test_types.append("quick")
    if args.comprehensive:
        test_types.append("comprehensive")
    
    if not test_types:  # Default to all
        test_types = ["models", "training", "data", "comprehensive", "quick"]
    
    # Run tests
    runner = UnifiedTestRunner(verbose=args.verbose, parallel=args.parallel)
    results = runner.run_all_tests(test_types)
    runner.print_results()
    
    # Save results if requested
    if args.save_results:
        runner.save_results()
    
    # Exit with appropriate code
    if results["overall"]["success_rate"] >= 0.8:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main() 