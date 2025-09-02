#!/usr/bin/env python3
"""
Comprehensive Tests for Quantitative Data Processor
Tests the new quantitative finance optimized data loading approach
"""

import sys
import os
import tempfile
import pandas as pd
import numpy as np
import unittest
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_processing.quantitative_data_processor import (
    QuantitativeDataConfig,
    QuantitativeTimeSeriesSplitter,
    QuantitativeDataCleaner,
    QuantitativeSequenceGenerator,
    QuantitativeFinanceDataset,
    create_quantitative_dataloaders
)


class TestQuantitativeDataProcessor(unittest.TestCase):
    """Test suite for quantitative data processor"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = QuantitativeDataConfig(
            train_start_date="2020-01-01",
            train_end_date="2020-08-31",
            val_start_date="2020-09-01",
            val_end_date="2020-10-31", 
            test_start_date="2020-11-01",
            test_end_date="2020-12-31",
            sequence_length=5,  # Reduced for testing
            prediction_horizon=1,
            batch_size=32,
            min_stock_history_days=10  # Much lower for testing
        )
        
        # Create temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir)
        
        # Generate test data
        self._create_test_data()
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def _create_test_data(self):
        """Create synthetic financial data for testing"""
        # Create data for multiple dates - ensure coverage of all periods
        dates = []
        base_date = datetime(2020, 1, 1)
        for i in range(365):  # Full year of data
            dates.append((base_date + timedelta(days=i)).strftime("%Y-%m-%d"))
        
        # Create 20 stocks for faster testing
        stock_ids = list(range(20))
        
        for date in dates:
            # Create daily data
            daily_data = []
            
            for stock_id in stock_ids:
                # Generate synthetic factor data
                factors = np.random.randn(100)  # 100 factors
                
                # Generate synthetic targets with some correlation
                targets = {
                    'intra30m': np.random.randn() * 0.01,
                    'nextT1d': np.random.randn() * 0.02,
                    'ema1d': np.random.randn() * 0.015
                }
                
                # Create record
                record = {
                    'sid': stock_id,
                    'date': date,
                    'ADV50': np.random.uniform(1000000, 10000000)  # Average dollar volume
                }
                
                # Add factors
                for i in range(100):
                    record[str(i)] = factors[i]
                
                # Add targets
                record.update(targets)
                
                # Add some missing values (5% probability)
                if np.random.random() < 0.05:
                    factor_idx = np.random.randint(0, 100)
                    record[str(factor_idx)] = np.nan
                
                daily_data.append(record)
            
            # Save daily file
            df = pd.DataFrame(daily_data)
            file_path = self.data_dir / f"{date}.parquet"
            df.to_parquet(file_path)
    
    def test_time_series_splitter(self):
        """Test time series splitting using time windows"""
        # Get available dates
        available_dates = [f.stem for f in self.data_dir.glob("*.parquet")]
        
        # Test splitter
        splitter = QuantitativeTimeSeriesSplitter(self.config)
        train_dates, val_dates, test_dates = splitter.split_dates(available_dates)
        
        # Verify splits
        self.assertGreater(len(train_dates), 0, "Training set should not be empty")
        self.assertGreater(len(val_dates), 0, "Validation set should not be empty")
        self.assertGreater(len(test_dates), 0, "Test set should not be empty")
        
        # Verify temporal ordering
        if train_dates and val_dates:
            self.assertLess(max(train_dates), min(val_dates), 
                          "Training should end before validation")
        
        if val_dates and test_dates:
            self.assertLess(max(val_dates), min(test_dates),
                          "Validation should end before test")
        
        # Verify no overlap
        train_set = set(train_dates)
        val_set = set(val_dates)
        test_set = set(test_dates)
        
        self.assertEqual(len(train_set & val_set), 0, "No overlap between train and val")
        self.assertEqual(len(train_set & test_set), 0, "No overlap between train and test")
        self.assertEqual(len(val_set & test_set), 0, "No overlap between val and test")
    
    def test_data_cleaner(self):
        """Test data cleaning functionality"""
        # Load sample data
        sample_file = list(self.data_dir.glob("*.parquet"))[0]
        df = pd.read_parquet(sample_file)
        
        # Test cleaner
        cleaner = QuantitativeDataCleaner(self.config)
        cleaned_df = cleaner.clean_stock_data(df)
        
        # Verify cleaning results
        self.assertGreater(len(cleaned_df), 0, "Some data should remain after cleaning")
        self.assertLessEqual(len(cleaned_df), len(df), "Cleaning should not add data")
        
        # Check for required columns
        self.assertIn('sid', cleaned_df.columns, "Stock ID column should exist")
        self.assertIn('date', cleaned_df.columns, "Date column should exist")
        
        # Check data quality
        self.assertFalse(cleaned_df['sid'].isnull().any(), "No null stock IDs")
        self.assertFalse(cleaned_df['date'].isnull().any(), "No null dates")
    
    def test_sequence_generator(self):
        """Test sequence generation with no-look-ahead policy"""
        # Load and clean sample data
        sample_file = list(self.data_dir.glob("*.parquet"))[0]
        df = pd.read_parquet(sample_file)
        
        cleaner = QuantitativeDataCleaner(self.config)
        cleaned_df = cleaner.clean_stock_data(df)
        
        # Test sequence generator
        generator = QuantitativeSequenceGenerator(self.config)
        sequences = list(generator.create_sequences(cleaned_df, mode='train'))
        
        # Verify sequences
        self.assertGreater(len(sequences), 0, "Should generate some sequences")
        
        for sequence in sequences[:5]:  # Check first 5 sequences
            # Check structure
            self.assertIn('features', sequence, "Sequence should have features")
            self.assertIn('targets', sequence, "Sequence should have targets")
            self.assertIn('stock_id', sequence, "Sequence should have stock_id")
            
            # Check shapes
            features = sequence['features']
            targets = sequence['targets']
            
            self.assertEqual(features.shape[0], self.config.sequence_length,
                           f"Features should have {self.config.sequence_length} timesteps")
            self.assertEqual(features.shape[1], 100, "Features should have 100 factors")
            self.assertEqual(len(targets), 3, "Should have 3 targets")
            
            # Check for no NaN values
            self.assertFalse(np.isnan(features).any(), "Features should not contain NaN")
            self.assertFalse(np.isnan(targets).any(), "Targets should not contain NaN")
    
    def test_quantitative_dataset(self):
        """Test quantitative finance dataset"""
        # Get some data files
        data_files = [str(f) for f in list(self.data_dir.glob("*.parquet"))[:10]]
        
        # Create dataset
        dataset = QuantitativeFinanceDataset(data_files, self.config, mode='train')
        
        # Test dataset
        self.assertGreater(len(dataset), 0, "Dataset should not be empty")
        
        # Test sample access
        sample = dataset[0]
        
        # Verify sample structure
        self.assertIn('features', sample, "Sample should have features")
        self.assertIn('targets', sample, "Sample should have targets")
        self.assertIn('stock_id', sample, "Sample should have stock_id")
        self.assertIn('weight', sample, "Sample should have weight")
        
        # Verify shapes
        features = sample['features']
        targets = sample['targets']
        
        self.assertEqual(features.shape[0], self.config.sequence_length)
        self.assertEqual(features.shape[1], 100)
        self.assertEqual(len(targets), 3)
    
    def test_create_quantitative_dataloaders(self):
        """Test full dataloader creation pipeline"""
        try:
            # Create dataloaders
            train_loader, val_loader, test_loader = create_quantitative_dataloaders(
                str(self.data_dir), self.config
            )
            
            # Verify dataloaders exist
            self.assertIsNotNone(train_loader, "Train loader should be created")
            self.assertIsNotNone(val_loader, "Val loader should be created")
            self.assertIsNotNone(test_loader, "Test loader should be created")
            
            # Test data loading
            train_batch = next(iter(train_loader))
            
            # Verify batch structure
            self.assertIn('features', train_batch, "Batch should have features")
            self.assertIn('targets', train_batch, "Batch should have targets")
            self.assertIn('stock_id', train_batch, "Batch should have stock_id")
            
            # Verify batch shapes
            features = train_batch['features']
            targets = train_batch['targets']
            
            self.assertEqual(len(features.shape), 3, "Features should be 3D: [batch, seq, features]")
            self.assertEqual(len(targets.shape), 2, "Targets should be 2D: [batch, targets]")
            
            # Verify temporal integrity
            batch_size = features.shape[0]
            self.assertLessEqual(batch_size, self.config.batch_size, 
                               "Batch size should not exceed configured size")
            
            print(f" Dataloaders created successfully")
            print(f"  Train batches: {len(train_loader)}")
            print(f"  Val batches: {len(val_loader)}")
            print(f"  Test batches: {len(test_loader)}")
            print(f"  Sample batch shape: {features.shape}")
            
        except Exception as e:
            self.fail(f"Failed to create dataloaders: {e}")
    
    def test_temporal_integrity(self):
        """Test that there's no data leakage in temporal splits"""
        # This is a critical test for financial time series
        available_dates = [f.stem for f in self.data_dir.glob("*.parquet")]
        
        splitter = QuantitativeTimeSeriesSplitter(self.config)
        train_dates, val_dates, test_dates = splitter.split_dates(available_dates)
        
        # Convert to datetime for proper comparison
        train_dt = [datetime.strptime(d, "%Y-%m-%d") for d in train_dates]
        val_dt = [datetime.strptime(d, "%Y-%m-%d") for d in val_dates]
        test_dt = [datetime.strptime(d, "%Y-%m-%d") for d in test_dates]
        
        # Verify strict temporal ordering
        if train_dt and val_dt:
            max_train = max(train_dt)
            min_val = min(val_dt)
            self.assertLess(max_train, min_val, 
                          f"Training data leakage: train ends {max_train}, val starts {min_val}")
        
        if val_dt and test_dt:
            max_val = max(val_dt)
            min_test = min(test_dt)
            self.assertLess(max_val, min_test,
                          f"Validation data leakage: val ends {max_val}, test starts {min_test}")
        
        print(" Temporal integrity verified - no data leakage detected")


def run_quantitative_data_processor_tests():
    """Run all quantitative data processor tests"""
    print("\n" + "=" * 80)
    print("QUANTITATIVE DATA PROCESSOR TESTS")
    print("=" * 80)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestQuantitativeDataProcessor)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 80)
    print("QUANTITATIVE DATA PROCESSOR TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_quantitative_data_processor_tests()
    sys.exit(0 if success else 1)
