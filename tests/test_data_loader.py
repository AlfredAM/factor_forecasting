#!/usr/bin/env python3
"""
Test script to verify data loading and file filtering
"""
import os
from pathlib import Path
from datetime import datetime

# Import the OptimizedStreamingDataLoader class
# Assuming that the data loader reads all parquet files from the given directory
from src.data_processing.optimized_streaming_loader import OptimizedStreamingDataLoader


def main():
    # Set the data directory (should match the server data directory)
    data_dir = '/nas/feature_v2_10s'
    # Create an instance of the loader
    loader = OptimizedStreamingDataLoader(data_dir=data_dir)
    
    # Define date range for filtering based on configuration
    train_start = '2018-01-02'
    train_end   = '2018-12-28'
    
    # Get filtered files
    filtered_files = loader._filter_files_by_date(start_date=train_start, end_date=train_end)
    
    print(f"filterfile{len(filtered_files)}")
    for f in filtered_files:
        print(f)

if __name__ == '__main__':
    main()
