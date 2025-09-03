#!/usr/bin/env python3
"""
dataload
dataloadimplementationdata
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Iterator
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

# path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_processing.quantitative_data_processor import (
    QuantitativeTimeSeriesSplitter,
    QuantitativeSequenceGenerator,
    QuantitativeDataConfig
)
from src.data_processing.adaptive_memory_manager import create_memory_manager
from src.data_processing.rolling_scaler import validate_time_series_split

logger = logging.getLogger(__name__)

class UnifiedStreamingDataset(IterableDataset):
    """
    datadata
    
    """
    
    def __init__(self,
                 data_dir: str,
                 factor_columns: List[str],
                 target_columns: List[str],
                 sequence_length: int = 20,
                 prediction_horizon: int = 1,
                 data_split: str = 'train',
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 enable_validation: bool = True,
                 memory_manager=None):
        """
        initializedata
        
        Args:
            data_dir: datadirectory
            factor_columns: 
            target_columns: target
            sequence_length: length
            prediction_horizon: predict
            data_split: datatype ('train', 'val', 'test')
            start_date: begin
            end_date: end
            enable_validation: enabledataverification
            memory_manager: memorymanagement
        """
        self.data_dir = Path(data_dir)
        self.factor_columns = factor_columns
        self.target_columns = target_columns
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.data_split = data_split
        self.start_date = start_date
        self.end_date = end_date
        self.enable_validation = enable_validation
        self.memory_manager = memory_manager
        
        # getfilelist
        self.data_files = self._get_data_files()
        
        # verification
        if self.enable_validation:
            self._validate_temporal_integrity()
        
        logger.info(f"datainitializecomplete")
        logger.info(f"  data: {data_split}")
        logger.info(f"  length: {sequence_length}")
        logger.info(f"  predict: {prediction_horizon}")
        logger.info(f"  : {start_date}  {end_date}")
        logger.info(f"  file: {len(self.data_files)}")

    def _get_data_files(self) -> List[Path]:
        """getdatafilelist"""
        all_files = []
        for file_path in self.data_dir.glob("*.parquet"):
            if len(file_path.stem) == 8 and file_path.stem.isdigit():
                file_date = file_path.stem
                
                # filter
                if self.start_date and file_date < self.start_date.replace('-', ''):
                    continue
                if self.end_date and file_date > self.end_date.replace('-', ''):
                    continue
                
                all_files.append(file_path)
        
        return sorted(all_files)

    def _validate_temporal_integrity(self):
        """verification"""
        if len(self.data_files) < 2:
            logger.warning("fileverification")
            return
        
        # checkfile
        dates = []
        for file_path in self.data_files:
            date_str = file_path.stem
            try:
                date_obj = datetime.strptime(date_str, "%Y%m%d")
                dates.append(date_obj)
            except ValueError:
                logger.warning(f"format: {date_str}")
        
        if len(dates) < 2:
            return
        
        # check7
        dates.sort()
        for i in range(1, len(dates)):
            gap = (dates[i] - dates[i-1]).days
            if gap > 7:
                logger.warning(f"discover: {dates[i-1].strftime('%Y-%m-%d')}  {dates[i].strftime('%Y-%m-%d')} ({gap})")
        
        logger.info(f"verificationcompletedata: {dates[0].strftime('%Y-%m-%d')}  {dates[-1].strftime('%Y-%m-%d')}")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """sampledata"""
        for file_path in self.data_files:
            try:
                # file
                df = pd.read_parquet(file_path)
                
                if df.empty:
                    continue
                
                # datacleanup
                df = self._clean_data(df)
                
                # data
                for sequence in self._generate_sequences(df, file_path):
                    yield sequence
                    
            except Exception as e:
                logger.error(f"processfile {file_path} error: {e}")
                continue

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """cleanupdata"""
        # necessary
        required_cols = ['sid'] + self.factor_columns + self.target_columns
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f": {missing_cols}")
            return df
        
        # packageNaN
        df = df.dropna(subset=self.factor_columns + self.target_columns)
        
        # datatype
        for col in self.factor_columns + self.target_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df

    def _generate_sequences(self, df: pd.DataFrame, file_path: Path) -> Iterator[Dict[str, torch.Tensor]]:
        """
        data
        usageverificationimplementation
        """
        # group
        for stock_id, stock_group in df.groupby('sid'):
            stock_group = stock_group.reset_index(drop=True)
            
            # checkdata
            min_required_length = self.sequence_length + self.prediction_horizon
            if len(stock_group) < min_required_length:
                continue
            
            # 
            max_start_idx = len(stock_group) - min_required_length
            
            for i in range(max_start_idx + 1):
                try:
                    # featureshistorydata [i : i+sequence_length]
                    feature_start = i
                    feature_end = i + self.sequence_length
                    feature_data = stock_group[self.factor_columns].iloc[feature_start:feature_end].values
                    
                    # targetdatadata [i+sequence_length+prediction_horizon-1]
                    # targetfeaturesprediction_horizon
                    target_idx = i + self.sequence_length + self.prediction_horizon - 1
                    target_data = stock_group[self.target_columns].iloc[target_idx].values
                    
                    # datacheck
                    if np.isnan(feature_data).any() or np.isnan(target_data).any():
                        continue
                    
                    # createsample
                    sample = {
                        'features': torch.FloatTensor(feature_data),
                        'targets': torch.FloatTensor(target_data),
                        'stock_id': torch.LongTensor([stock_id]),
                        'file_date': file_path.stem,
                        'sequence_start_idx': i,
                        'target_idx': target_idx
                    }
                    
                    yield sample
                    
                except Exception as e:
                    logger.error(f"error ({stock_id}, index{i}): {e}")
                    continue

class UnifiedDataLoaderFactory:
    """dataload"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        initializedataload
        
        Args:
            config: configuredictionary
        """
        self.config = config
        self.data_dir = config['data_dir']
        self.factor_columns = config.get('factor_columns', [str(i) for i in range(100)])
        self.target_columns = config.get('target_columns', ['intra30m', 'nextT1d', 'ema1d'])
        self.sequence_length = config.get('sequence_length', 20)
        self.prediction_horizon = config.get('prediction_horizon', 1)
        self.batch_size = config.get('batch_size', 64)
        self.num_workers = config.get('num_workers', 4)
        
        # creatememorymanagement
        self.memory_manager = create_memory_manager({
            'monitoring_interval': 3.0,
            'critical_threshold': 0.85,
            'warning_threshold': 0.75
        })
        
        logger.info(f"dataloadinitializecomplete")

    def create_time_split_loaders(self, 
                                 train_start: str, train_end: str,
                                 val_start: str, val_end: str,
                                 test_start: str, test_end: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        createdataload
        data
        """
        # verification
        date_ranges = [
            (train_start, train_end, 'train'),
            (val_start, val_end, 'val'),
            (test_start, test_end, 'test')
        ]
        
        # objectverification
        train_end_date = datetime.strptime(train_end, "%Y-%m-%d")
        val_start_date = datetime.strptime(val_start, "%Y-%m-%d")
        val_end_date = datetime.strptime(val_end, "%Y-%m-%d")
        test_start_date = datetime.strptime(test_start, "%Y-%m-%d")
        
        # verification
        if not (train_end_date < val_start_date <= val_end_date < test_start_date):
            raise ValueError("data")
        
        logger.info(" verificationdata")
        
        # createdata
        train_dataset = UnifiedStreamingDataset(
            data_dir=self.data_dir,
            factor_columns=self.factor_columns,
            target_columns=self.target_columns,
            sequence_length=self.sequence_length,
            prediction_horizon=self.prediction_horizon,
            data_split='train',
            start_date=train_start,
            end_date=train_end,
            memory_manager=self.memory_manager
        )
        
        val_dataset = UnifiedStreamingDataset(
            data_dir=self.data_dir,
            factor_columns=self.factor_columns,
            target_columns=self.target_columns,
            sequence_length=self.sequence_length,
            prediction_horizon=self.prediction_horizon,
            data_split='val',
            start_date=val_start,
            end_date=val_end,
            memory_manager=self.memory_manager
        )
        
        test_dataset = UnifiedStreamingDataset(
            data_dir=self.data_dir,
            factor_columns=self.factor_columns,
            target_columns=self.target_columns,
            sequence_length=self.sequence_length,
            prediction_horizon=self.prediction_horizon,
            data_split='test',
            start_date=test_start,
            end_date=test_end,
            memory_manager=self.memory_manager
        )
        
        # createDataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=0,  # IterableDatasetsupportprocess
            pin_memory=torch.cuda.is_available(),
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=False
        )
        
        logger.info(f"dataloadcreatecomplete:")
        logger.info(f"  training: {train_start}  {train_end}")
        logger.info(f"  verification: {val_start}  {val_end}")
        logger.info(f"  test: {test_start}  {test_end}")
        
        return train_loader, val_loader, test_loader

    def create_yearly_split_loaders(self, train_years: List[int], test_year: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        createdataload
        training
        """
        if not train_years or test_year in train_years:
            raise ValueError("trainingtestsetuperror")
        
        if max(train_years) >= test_year:
            raise ValueError("testtraining")
        
        # build
        train_start = f"{min(train_years)}-01-01"
        train_end = f"{max(train_years)}-12-31"
        
        # trainingverificationusage
        val_year = max(train_years)
        val_start = f"{val_year}-07-01"
        val_end = f"{val_year}-12-31"
        
        # adjustmenttrainingend
        if len(train_years) > 1:
            train_end = f"{val_year}-06-30"
        else:
            # 
            train_end = f"{val_year}-06-30"
        
        test_start = f"{test_year}-01-01"
        test_end = f"{test_year}-12-31"
        
        logger.info(f"data:")
        logger.info(f"  training: {train_years}")
        logger.info(f"  test: {test_year}")
        logger.info(f"  training: {train_start}  {train_end}")
        logger.info(f"  verification: {val_start}  {val_end}")
        logger.info(f"  test: {test_start}  {test_end}")
        
        return self.create_time_split_loaders(
            train_start, train_end,
            val_start, val_end,
            test_start, test_end
        )

    def cleanup(self):
        """cleanupresource"""
        if self.memory_manager:
            self.memory_manager.cleanup()


def create_unified_data_loaders(config: Dict[str, Any], 
                               time_split_config: Optional[Dict[str, str]] = None,
                               yearly_split_config: Optional[Dict[str, Any]] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    createdataloadfunction
    
    Args:
        config: foundationconfigure
        time_split_config: configure {'train_start': ..., 'train_end': ..., ...}
        yearly_split_config: configure {'train_years': [...], 'test_year': ...}
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    factory = UnifiedDataLoaderFactory(config)
    
    if time_split_config:
        return factory.create_time_split_loaders(
            time_split_config['train_start'],
            time_split_config['train_end'],
            time_split_config['val_start'],
            time_split_config['val_end'],
            time_split_config['test_start'],
            time_split_config['test_end']
        )
    elif yearly_split_config:
        return factory.create_yearly_split_loaders(
            yearly_split_config['train_years'],
            yearly_split_config['test_year']
        )
    else:
        raise ValueError("configureconfigure")


# compatiblefunction
def create_leak_free_data_loaders(data_dir: str,
                                 factor_columns: List[str],
                                 target_columns: List[str],
                                 sequence_length: int = 20,
                                 prediction_horizon: int = 1,
                                 batch_size: int = 64,
                                 train_start: str = "2018-01-01",
                                 train_end: str = "2018-10-31",
                                 val_start: str = "2018-11-01", 
                                 val_end: str = "2018-12-31",
                                 test_start: str = "2019-01-01",
                                 test_end: str = "2019-03-31") -> Tuple[DataLoader, DataLoader, DataLoader]:
    """createdatadataload"""
    
    config = {
        'data_dir': data_dir,
        'factor_columns': factor_columns,
        'target_columns': target_columns,
        'sequence_length': sequence_length,
        'prediction_horizon': prediction_horizon,
        'batch_size': batch_size
    }
    
    time_split_config = {
        'train_start': train_start,
        'train_end': train_end,
        'val_start': val_start,
        'val_end': val_end,
        'test_start': test_start,
        'test_end': test_end
    }
    
    return create_unified_data_loaders(config, time_split_config=time_split_config)
