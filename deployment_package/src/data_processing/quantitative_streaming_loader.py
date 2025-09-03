#!/usr/bin/env python3
"""
dataload
datapredict
"""
import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
from typing import Dict, List, Tuple, Optional, Iterator, Any
import logging
from datetime import datetime, timedelta
from pathlib import Path
import warnings
from collections import defaultdict

logger = logging.getLogger(__name__)

class QuantitativeStreamingDataset(IterableDataset):
    """
    data
    :
    1. predictdata
    2. processspecial
    3. supportprediction horizon
    4. memoryload
    """
    
    def __init__(self,
                 streaming_loader,
                 factor_columns: List[str],
                 target_columns: List[str],
                 sequence_length: int = 20,
                 prediction_horizon: int = 1,
                 min_stock_history: int = 252,  # 1historydata
                 remove_limit_up_down: bool = True,
                 remove_suspended: bool = True,
                 enable_sequence_shuffle: bool = False,
                 shuffle_buffer_size: int = 10000):
        """
        initializedata
        
        Args:
            streaming_loader: dataload
            factor_columns: 
            target_columns: target  
            sequence_length: length
            prediction_horizon: predict
            min_stock_history: historydata
            remove_limit_up_down: data
            remove_suspended: data
            enable_sequence_shuffle: enableshuffle
            shuffle_buffer_size: shufflesize
        """
        self.streaming_loader = streaming_loader
        self.factor_columns = factor_columns
        self.target_columns = target_columns
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.min_stock_history = min_stock_history
        self.remove_limit_up_down = remove_limit_up_down
        self.remove_suspended = remove_suspended
        self.enable_sequence_shuffle = enable_sequence_shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        
        # datastatistics
        self.stats = {
            'total_sequences': 0,
            'filtered_limit_up_down': 0,
            'filtered_suspended': 0,
            'filtered_insufficient_history': 0,
            'filtered_data_quality': 0
        }
        
        logger.info(f"initializedata:")
        logger.info(f"  length: {sequence_length}")
        logger.info(f"  predict: {prediction_horizon}")
        logger.info(f"  : {len(factor_columns)}")
        logger.info(f"  target: {len(target_columns)}")
        logger.info(f"  : {remove_limit_up_down}")
        logger.info(f"  : {remove_suspended}")
    
    def __iter__(self):
        """"""
        sequence_buffer = []
        
        for df in self.streaming_loader.stream_data():
            if df is None or df.empty:
                continue
            
            # cleanupprocessdata
            df = self._preprocess_financial_data(df)
            
            # 
            for sequence in self._create_quantitative_sequences(df):
                if sequence is not None:
                    if self.enable_sequence_shuffle:
                        sequence_buffer.append(sequence)
                        if len(sequence_buffer) >= self.shuffle_buffer_size:
                            np.random.shuffle(sequence_buffer)
                            for seq in sequence_buffer:
                                yield seq
                            sequence_buffer = []
                    else:
                        yield sequence
        
        # processshuffle buffer
        if sequence_buffer:
            np.random.shuffle(sequence_buffer)
            for seq in sequence_buffer:
                yield seq
    
    def _preprocess_financial_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """processdatadata"""
        original_len = len(df)
        
        # 1. data
        if self.remove_limit_up_down and 'luld' in df.columns:
            # luldmark1=, -1=, 0=
            mask_limit = df['luld'] == 0
            df = df[mask_limit]
            self.stats['filtered_limit_up_down'] += (original_len - len(df))
        
        # 2. data
        if self.remove_suspended:
            # 00decision
            if 'volume' in df.columns:
                mask_suspended = df['volume'] > 0
                df = df[mask_suspended]
                self.stats['filtered_suspended'] += (original_len - len(df))
            elif 'amount' in df.columns:
                mask_suspended = df['amount'] > 0
                df = df[mask_suspended]
                self.stats['filtered_suspended'] += (original_len - len(df))
        
        # 3. datacheck
        # datarecord
        factor_data = df[self.factor_columns]
        missing_ratio = factor_data.isnull().sum(axis=1) / len(self.factor_columns)
        mask_quality = missing_ratio < 0.5  # 50%
        df = df[mask_quality]
        self.stats['filtered_data_quality'] += (len(df) - mask_quality.sum())
        
        # 4. sort
        if 'time' in df.columns:
            df = df.sort_values(['sid', 'time'])
        elif 'date' in df.columns:
            df = df.sort_values(['sid', 'date'])
        
        return df
    
    def _create_quantitative_sequences(self, df: pd.DataFrame) -> Iterator[Dict[str, torch.Tensor]]:
        """createdata"""
        
        # groupprocess
        for stock_id, stock_group in df.groupby('sid'):
            stock_group = stock_group.reset_index(drop=True)
            
            # checkhistorydata
            if len(stock_group) < self.min_stock_history:
                self.stats['filtered_insufficient_history'] += 1
                continue
            
            # 
            # datacreatepredicttarget
            min_required_length = self.sequence_length + self.prediction_horizon
            max_start_idx = len(stock_group) - min_required_length
            
            if max_start_idx < 0:
                continue
            
            for i in range(max_start_idx + 1):
                # featureshistorydata [i : i+sequence_length]
                feature_start = i
                feature_end = i + self.sequence_length
                feature_data = stock_group[self.factor_columns].iloc[feature_start:feature_end].values
                
                # targetdatadata [i+sequence_length+prediction_horizon-1]
                target_idx = i + self.sequence_length + self.prediction_horizon - 1
                target_data = stock_group[self.target_columns].iloc[target_idx].values
                
                # datacheck
                if self._validate_sequence_quality(feature_data, target_data, stock_group, i, target_idx):
                    # getweights
                    weight = self._get_sequence_weight(stock_group, target_idx)
                    
                    # createinfo
                    if 'date' in stock_group.columns:
                        feature_dates = stock_group['date'].iloc[feature_start:feature_end].tolist()
                        target_date = stock_group['date'].iloc[target_idx]
                    else:
                        feature_dates = list(range(feature_start, feature_end))
                        target_date = target_idx
                    
                    sequence = {
                        'features': torch.FloatTensor(feature_data),
                        'targets': torch.FloatTensor(target_data),
                        'stock_id': torch.LongTensor([stock_id]),
                        'weight': torch.FloatTensor([weight]),
                        'feature_dates': feature_dates,
                        'target_date': target_date,
                        'prediction_horizon': self.prediction_horizon
                    }
                    
                    self.stats['total_sequences'] += 1
                    yield sequence
    
    def _validate_sequence_quality(self, feature_data: np.ndarray, target_data: np.ndarray, 
                                 stock_group: pd.DataFrame, feature_start: int, target_idx: int) -> bool:
        """verification"""
        
        # 1. checkNaN
        if np.isnan(feature_data).any() or np.isnan(target_data).any():
            return False
        
        # 2. check
        if 'date' in stock_group.columns:
            feature_dates = pd.to_datetime(stock_group['date'].iloc[feature_start:feature_start+self.sequence_length])
            target_date = pd.to_datetime(stock_group['date'].iloc[target_idx])
            
            # checkfeatures
            date_diff = (feature_dates.max() - feature_dates.min()).days
            expected_days = len(feature_dates) * 1.5  # 50%
            if date_diff > expected_days:
                return False
            
            # checkpredicttarget
            prediction_gap = (target_date - feature_dates.max()).days
            if prediction_gap < self.prediction_horizon or prediction_gap > self.prediction_horizon * 2:
                return False
        
        # 3. checkdata
        feature_std = np.std(feature_data, axis=0)
        if np.any(feature_std == 0):  # 
            return False
        
        # 4. checktarget
        target_abs = np.abs(target_data)
        if np.any(target_abs > 0.2):  # 20%data
            return False
        
        return True
    
    def _get_sequence_weight(self, stock_group: pd.DataFrame, target_idx: int) -> float:
        """getweights"""
        # usage
        if 'ADV50' in stock_group.columns:  # 50
            weight = stock_group['ADV50'].iloc[target_idx]
            if pd.notna(weight) and weight > 0:
                return float(weight)
        
        if 'market_cap' in stock_group.columns:
            weight = stock_group['market_cap'].iloc[target_idx]
            if pd.notna(weight) and weight > 0:
                return float(weight)
        
        # defaultweights
        return 1.0
    
    def get_statistics(self) -> Dict[str, int]:
        """getdataprocessstatisticsinfo"""
        return self.stats.copy()


def create_quantitative_dataloaders(
    data_dir: str,
    factor_columns: List[str],
    target_columns: List[str],
    train_files: List[str],
    val_files: List[str],
    test_files: List[str],
    sequence_length: int = 20,
    prediction_horizon: int = 1,
    batch_size: int = 64,
    memory_config: Optional[Dict[str, Any]] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    createdataload
    
    Args:
        data_dir: datadirectory
        factor_columns: 
        target_columns: target
        train_files: trainingfilelist
        val_files: verificationfilelist 
        test_files: testfilelist
        sequence_length: length
        prediction_horizon: predict
        batch_size: size
        memory_config: memoryconfigure
        
    Returns:
        trainingverificationtestdataload
    """
    from .adaptive_memory_manager import create_memory_manager
    from .optimized_streaming_loader import OptimizedStreamingDataLoader
    
    # creatememorymanagement
    memory_manager = create_memory_manager(memory_config or {})
    
    # createload
    def create_dataloader(files: List[str], shuffle: bool = False) -> DataLoader:
        streaming_loader = OptimizedStreamingDataLoader(
            data_dir=data_dir,
            memory_manager=memory_manager,
            max_workers=4,
            enable_async_loading=True
        )
        # setupfilelist
        streaming_loader.file_list = files
        
        dataset = QuantitativeStreamingDataset(
            streaming_loader=streaming_loader,
            factor_columns=factor_columns,
            target_columns=target_columns,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            enable_sequence_shuffle=shuffle,
            shuffle_buffer_size=10000 if shuffle else 0
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,  # datasupportprocess
            pin_memory=torch.cuda.is_available()
        )
    
    train_loader = create_dataloader(train_files, shuffle=True)
    val_loader = create_dataloader(val_files, shuffle=False)
    test_loader = create_dataloader(test_files, shuffle=False)
    
    logger.info(f"createdataloadcomplete:")
    logger.info(f"  trainingfile: {len(train_files)}")
    logger.info(f"  verificationfile: {len(val_files)}")
    logger.info(f"  testfile: {len(test_files)}")
    logger.info(f"  predict: {prediction_horizon}")
    
    return train_loader, val_loader, test_loader
