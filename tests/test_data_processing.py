#!/usr/bin/env python3
"""
Comprehensive Data Processing Tests for Factor Forecasting System
==============================================================

This module contains comprehensive tests for all data processing components:
- Data loading and preprocessing
- Dataset classes
- DataLoader creation
- Feature engineering
- Data validation

Usage:
    python test_data_processing.py [options]

Options:
    --loading       Test data loading only
    --preprocessing Test data preprocessing only
    --datasets      Test dataset classes only
    --validation    Test data validation only
    --all           Test all components (default)
    --verbose       Enable verbose output
"""

import sys
import torch
import numpy as np
import pandas as pd
import tempfile
import shutil
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_processing.data_processor import (
    MultiFileDataProcessor, DataManager, create_training_dataloaders,
    MultiFileDataset
)
from src.data_processing.streaming_data_loader import StreamingDataLoader, StreamingDataset
from configs.config import ModelConfig


class DataProcessingTestSuite:
    """Comprehensive data processing test suite"""
    
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
    
    def test_data_loading(self) -> Dict[str, Any]:
        """Test data loading functionality"""
        print("Testing Data Loading...")
        results = {"passed": 0, "failed": 0, "errors": []}
        
        try:
            # Create sample data
            data = self._create_sample_data()
            
            # Save to parquet file
            data_path = os.path.join(self.temp_dir, "test_data.parquet")
            data.to_parquet(data_path, index=False)
            
            # Test loading
            loaded_data = pd.read_parquet(data_path)
            
            # Check data integrity
            assert len(loaded_data) == len(data), "Data length mismatch"
            assert list(loaded_data.columns) == list(data.columns), "Column mismatch"
            
            results["passed"] += 1
            
            if self.verbose:
                print("  Data loading test passed")
                
        except Exception as e:
            results["failed"] += 1
            results["errors"].append(f"Data loading test error: {e}")
            if self.verbose:
                print(f"  Data loading test error: {e}")
        
        return results
    
    def test_data_preprocessing(self) -> Dict[str, Any]:
        """Test data preprocessing functionality"""
        print("Testing Data Preprocessing...")
        results = {"passed": 0, "failed": 0, "errors": []}
        
        try:
            # Create sample data
            data = self._create_sample_data()
            
            # Test preprocessing
            config = ModelConfig()
            
            # Check required columns
            required_columns = config.factor_columns + config.target_columns + [config.stock_id_column]
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                # Add missing columns
                for col in missing_columns:
                    if col in config.factor_columns:
                        data[col] = np.random.randn(len(data))
                    elif col in config.target_columns:
                        data[col] = np.random.randn(len(data))
                    else:
                        data[col] = f"stock_{np.random.randint(0, 10)}"
            
            # Test data validation
            assert len(data) > 0, "Data is empty"
            assert not data.isnull().all().any(), "Data contains all-null columns"
            
            results["passed"] += 1
            
            if self.verbose:
                print("  Data preprocessing test passed")
                
        except Exception as e:
            results["failed"] += 1
            results["errors"].append(f"Data preprocessing test error: {e}")
            if self.verbose:
                print(f"  Data preprocessing test error: {e}")
        
        return results
    
    def test_dataset_classes(self) -> Dict[str, Any]:
        """Test dataset classes"""
        print("Testing Dataset Classes...")
        results = {"passed": 0, "failed": 0, "errors": []}
        
        try:
            # Create sample data
            data = self._create_sample_data()
            
            # Test StreamingDataset (replacing SingleFileDataset)
            config = ModelConfig()
            
            # Create temporary data directory
            temp_data_dir = os.path.join(self.temp_dir, "streaming_test")
            os.makedirs(temp_data_dir, exist_ok=True)
            
            # Save sample data as parquet file
            data_path = os.path.join(temp_data_dir, "test_data.parquet")
            data.to_parquet(data_path, index=False)
            
            # Create streaming data loader
            streaming_loader = StreamingDataLoader(
                data_dir=temp_data_dir,
                batch_size=100,
                cache_size=2,
                max_memory_mb=512
            )
            
            # Test basic functionality
            assert len(streaming_loader.data_files) > 0, "No data files found"
            
            results["passed"] += 1
            
            # Test MultiFileDataset
            multi_dataset = MultiFileDataset(
                dataframes=[data],  # Use dataframes parameter
                config=config,
                mode="train"
            )
            
            # Test multi-file dataset
            assert len(multi_dataset) > 0, "Multi-file dataset is empty"
            
            sample = multi_dataset[0]
            assert isinstance(sample, dict), "Multi-file sample should be a dictionary"
            assert "features" in sample, "Sample should contain features"
            assert "targets" in sample, "Sample should contain targets"
            
            results["passed"] += 1
            
            if self.verbose:
                print("  Dataset classes test passed")
                
        except Exception as e:
            results["failed"] += 1
            results["errors"].append(f"Dataset classes test error: {e}")
            if self.verbose:
                print(f"  Dataset classes test error: {e}")
        
        return results
    
    def test_dataloader_creation(self) -> Dict[str, Any]:
        """Test DataLoader creation"""
        print("Testing DataLoader Creation...")
        results = {"passed": 0, "failed": 0, "errors": []}
        
        try:
            # Create sample data
            data = self._create_sample_data()
            
            # Test DataLoader creation
            config = ModelConfig()
            config.batch_size = 16
            config.data_dir = self.temp_dir
            
            # Save sample data
            temp_data_dir = os.path.join(self.temp_dir, "dataloader_test")
            os.makedirs(temp_data_dir, exist_ok=True)
            data.to_parquet(os.path.join(temp_data_dir, "2020-01-01.parquet"), index=False)
            
            config.data_dir = temp_data_dir
            
            dataloaders, scalers = create_training_dataloaders(config)
            train_loader = dataloaders['train']
            val_loader = dataloaders['val']
            test_loader = dataloaders['test']
            
            # Check DataLoaders
            assert train_loader is not None, "Train DataLoader not created"
            assert val_loader is not None, "Val DataLoader not created"
            assert test_loader is not None, "Test DataLoader not created"
            
            # Test batch iteration
            for batch in train_loader:
                assert isinstance(batch, dict), "Batch should be a dictionary"
                assert "features" in batch, "Batch should contain features"
                assert "targets" in batch, "Batch should contain targets"
                break  # Just test first batch
            
            results["passed"] += 1
            
            if self.verbose:
                print("  DataLoader creation test passed")
                
        except Exception as e:
            results["failed"] += 1
            results["errors"].append(f"DataLoader creation test error: {e}")
            if self.verbose:
                print(f"  DataLoader creation test error: {e}")
        
        return results
    
    def test_data_validation(self) -> Dict[str, Any]:
        """Test data validation"""
        print("Testing Data Validation...")
        results = {"passed": 0, "failed": 0, "errors": []}
        
        try:
            # Create sample data
            data = self._create_sample_data()
            
            # Test data validation
            config = ModelConfig()
            
            # Check data types
            for col in config.factor_columns[:10]:  # Check first 10 factors
                if col in data.columns:
                    assert pd.api.types.is_numeric_dtype(data[col]), f"Factor column {col} should be numeric"
            
            for col in config.target_columns:
                if col in data.columns:
                    assert pd.api.types.is_numeric_dtype(data[col]), f"Target column {col} should be numeric"
            
            # Check for missing values
            missing_factors = data[config.factor_columns[:10]].isnull().sum().sum()
            assert missing_factors == 0, "Factor columns should not have missing values"
            
            # Check for infinite values
            infinite_factors = np.isinf(data[config.factor_columns[:10]]).sum().sum()
            assert infinite_factors == 0, "Factor columns should not have infinite values"
            
            results["passed"] += 1
            
            if self.verbose:
                print("  Data validation test passed")
                
        except Exception as e:
            results["failed"] += 1
            results["errors"].append(f"Data validation test error: {e}")
            if self.verbose:
                print(f"  Data validation test error: {e}")
        
        return results
    
    def test_no_overlap_date_windows(self):
        val_end = datetime.strptime('2018-11-30', '%Y-%m-%d')
        test_start = datetime.strptime('2018-12-01', '%Y-%m-%d')
        assert val_end < test_start
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample data for testing"""
        np.random.seed(42)
        
        # Create sample data
        num_stocks = 20
        num_days = 10
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
        """Run all data processing tests"""
        if test_types is None:
            test_types = ["loading", "preprocessing", "datasets", "dataloader", "validation"]
        
        print("Running Data Processing Tests...")
        print("=" * 50)
        
        # Set up test environment
        self.setUp()
        
        all_results = {}
        
        try:
            for test_type in test_types:
                if test_type == "loading":
                    all_results["loading"] = self.test_data_loading()
                elif test_type == "preprocessing":
                    all_results["preprocessing"] = self.test_data_preprocessing()
                elif test_type == "datasets":
                    all_results["datasets"] = self.test_dataset_classes()
                elif test_type == "dataloader":
                    all_results["dataloader"] = self.test_dataloader_creation()
                elif test_type == "validation":
                    all_results["validation"] = self.test_data_validation()
        finally:
            # Clean up test environment
            self.tearDown()
        
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
        print("Data Processing Test Results")
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
            print("\nAll data processing tests passed!")
        elif overall['success_rate'] >= 0.8:
            print("\nMost data processing tests passed!")
        else:
            print("\nMany data processing tests failed!")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run data processing tests")
    parser.add_argument("--loading", action="store_true", help="Test data loading only")
    parser.add_argument("--preprocessing", action="store_true", help="Test data preprocessing only")
    parser.add_argument("--datasets", action="store_true", help="Test dataset classes only")
    parser.add_argument("--dataloader", action="store_true", help="Test DataLoader creation only")
    parser.add_argument("--validation", action="store_true", help="Test data validation only")
    parser.add_argument("--all", action="store_true", help="Test all components (default)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Determine test types
    test_types = []
    if args.loading:
        test_types.append("loading")
    if args.preprocessing:
        test_types.append("preprocessing")
    if args.datasets:
        test_types.append("datasets")
    if args.dataloader:
        test_types.append("dataloader")
    if args.validation:
        test_types.append("validation")
    
    if not test_types:  # Default to all
        test_types = ["loading", "preprocessing", "datasets", "dataloader", "validation"]
    
    # Run tests
    test_suite = DataProcessingTestSuite(verbose=args.verbose)
    results = test_suite.run_all_tests(test_types)
    test_suite.print_results()
    
    # Exit with appropriate code
    if results["overall"]["success_rate"] == 1.0:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main() 