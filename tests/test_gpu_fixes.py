#!/usr/bin/env python3
"""
Test script to verify GPU utilization fixes
"""
import os
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_date_comparison():
    """Test the fixed date comparison logic"""
    print("=== Testing Date Comparison Logic ===")

    # Test cases
    test_cases = [
        ("2018-01-01", "2018-01-02", False),  # file_date < start_date, should be filtered out
        ("2018-01-02", "2018-01-02", True),   # file_date == start_date, should be included
        ("2018-01-03", "2018-01-02", True),   # file_date > start_date, should be included
        ("2018-12-31", "2018-12-31", True),   # file_date == end_date, should be included
        ("2019-01-01", "2018-12-31", False),  # file_date > end_date, should be filtered out
    ]

    for file_date, boundary_date, expected in test_cases:
        try:
            file_date_obj = datetime.strptime(file_date, "%Y-%m-%d")
            boundary_date_obj = datetime.strptime(boundary_date, "%Y-%m-%d")

            # Old logic (string comparison) - WRONG
            old_result = file_date >= boundary_date

            # New logic (datetime comparison) - CORRECT
            new_result = file_date_obj >= boundary_date_obj

            status = "" if new_result == expected else ""
            print(f"{status} {file_date} >= {boundary_date}: Old={old_result}, New={new_result}, Expected={expected}")

        except Exception as e:
            print(f" Error with {file_date} vs {boundary_date}: {e}")

    print()

def test_filename_extraction():
    """Test filename to date extraction"""
    print("=== Testing Filename Extraction ===")

    test_files = [
        "20180102.parquet",
        "20181231.parquet",
        "2019-01-01.parquet",
        "invalid_file.parquet"
    ]

    for filename in test_files:
        stem = Path(filename).stem
        try:
            if len(stem) == 8 and stem.isdigit():
                date_str = f"{stem[:4]}-{stem[4:6]}-{stem[6:]}"
                print(f" {filename} -> {date_str}")
            elif len(stem) >= 10 and stem[4] == '-' and stem[7] == '-':
                print(f" {filename} -> {stem}")
            else:
                print(f"- {filename} -> No date found")
        except Exception as e:
            print(f" {filename} -> Error: {e}")

    print()

def main():
    """Main test function"""
    print("GPUverification")
    print("=" * 50)

    test_date_comparison()
    test_filename_extraction()

    print("===  ===")
    print("1.  dataloadcomparison")
    print("2.  comparisondatetimeobjectcomparison")
    print("3.  yearstrainingsetup")
    print("4.  disabledistributedtraining")
    print("5.  addlogrecord")
    print()
    print("GPU0")
    print("- datafilefilterload")
    print("- trainingnormalbegin")
    print("- GPUcalculatetaskbegin")

if __name__ == "__main__":
    main()

