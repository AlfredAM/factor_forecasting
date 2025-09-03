#!/usr/bin/env python3
"""
trainingsystem
7implementationdatafiletraintraininference
"""

import os
import sys
import logging
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

class YearlyRollingTrainer:
    """trainingsystem"""
    
    def __init__(self, 
                 data_dir: str,
                 model_factory,
                 output_dir: str,
                 config: Dict[str, Any]):
        """
        initializetraining
        
        Args:
            data_dir: datadirectorypackageparquetfile
            model_factory: modelfunction
            output_dir: outputdirectory
            config: trainingconfigure
        """
        self.data_dir = Path(data_dir)
        self.model_factory = model_factory
        self.output_dir = Path(output_dir)
        self.config = config
        
        # createoutputdirectory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # getdata
        self.available_files = self._discover_data_files()
        self.available_years = self._get_available_years()
        
        # trainingconfigure
        self.min_train_years = config.get('min_train_years', 2)
        self.max_train_years = config.get('max_train_years', 5)
        self.prediction_start_year = config.get('prediction_start_year', None)
        
        # traininghistory
        self.training_history = []
        self.inference_results = {}
        
        logger.info(f"traininginitializecomplete")
        logger.info(f"datadirectory: {self.data_dir}")
        logger.info(f": {self.available_years}")
        logger.info(f"file: {len(self.available_files)}")

    def _discover_data_files(self) -> List[Path]:
        """discoverdatafile"""
        files = []
        for file_path in self.data_dir.glob("*.parquet"):
            if len(file_path.stem) == 8 and file_path.stem.isdigit():
                files.append(file_path)
        
        return sorted(files)

    def _get_available_years(self) -> List[int]:
        """get"""
        years = set()
        for file_path in self.available_files:
            year = int(file_path.stem[:4])
            years.add(year)
        return sorted(list(years))

    def _get_files_for_years(self, years: List[int]) -> List[Path]:
        """getfile"""
        files = []
        for file_path in self.available_files:
            file_year = int(file_path.stem[:4])
            if file_year in years:
                files.append(file_path)
        return sorted(files)

    def _get_files_for_year(self, year: int) -> List[Path]:
        """getfile"""
        return self._get_files_for_years([year])

    def _create_datasets_for_years(self, train_years: List[int], test_year: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """createdata"""
        from src.data_processing.streaming_data_loader import StreamingDataLoader, StreamingDataset
        
        # getfile
        train_files = self._get_files_for_years(train_years)
        test_files = self._get_files_for_year(test_year)
        
        # trainingverificationtrainingverification
        if len(train_years) > 1:
            val_year = train_years[-1]
            train_years_final = train_years[:-1]
            train_files = self._get_files_for_years(train_years_final)
            val_files = self._get_files_for_year(val_year)
        else:
            # trainingdata80%training20%verification
            val_files = train_files[int(len(train_files) * 0.8):]
            train_files = train_files[:int(len(train_files) * 0.8)]
        
        logger.info(f"data:")
        logger.info(f"  trainingfile: {len(train_files)}  (: {train_years_final if len(train_years) > 1 else train_years})")
        logger.info(f"  verificationfile: {len(val_files)}  (: {val_year if len(train_years) > 1 else '' + str(train_years[0])})")
        logger.info(f"  testfile: {len(test_files)}  (: {test_year})")
        
        # createdataload
        streaming_loader = StreamingDataLoader(
            data_dir=str(self.data_dir),
            batch_size=1000,
            cache_size=5,
            max_memory_mb=4096,
            max_workers=4
        )
        
        # target
        factor_columns = [str(i) for i in range(100)]
        target_columns = self.config.get('target_columns', ['intra30m', 'nextT1d', 'ema1d'])
        sequence_length = self.config.get('sequence_length', 20)
        batch_size = self.config.get('batch_size', 64)
        
        # createdata
        train_dataset = StreamingDataset(
            streaming_loader=streaming_loader,
            factor_columns=factor_columns,
            target_columns=target_columns,
            sequence_length=sequence_length,
            start_date=None,
            end_date=None
        )
        
        val_dataset = StreamingDataset(
            streaming_loader=streaming_loader,
            factor_columns=factor_columns,
            target_columns=target_columns,
            sequence_length=sequence_length,
            start_date=None,
            end_date=None
        )
        
        test_dataset = StreamingDataset(
            streaming_loader=streaming_loader,
            factor_columns=factor_columns,
            target_columns=target_columns,
            sequence_length=sequence_length,
            start_date=None,
            end_date=None
        )
        
        # createDataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=0,  # IterableDatasetsupportprocess
            pin_memory=torch.cuda.is_available()
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        
        return train_loader, val_loader, test_loader

    def _train_single_window(self, train_years: List[int], test_year: int) -> Dict[str, Any]:
        """training"""
        logger.info(f"begintraining: {train_years} -> {test_year}")
        
        # createdata
        train_loader, val_loader, test_loader = self._create_datasets_for_years(train_years, test_year)
        
        # createmodel
        model = self.model_factory(self.config)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # createoptimize
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.get('learning_rate', 0.001),
            weight_decay=self.config.get('weight_decay', 1e-5)
        )
        
        # createlossfunction
        from src.training.quantitative_loss import QuantitativeCorrelationLoss
        loss_config = {
            'mse_weight': 0.4,
            'correlation_weight': 1.0,
            'rank_correlation_weight': 0.3,
            'risk_penalty_weight': 0.1,
            'target_correlations': [0.08, 0.05, 0.03]
        }
        criterion = QuantitativeCorrelationLoss(loss_config).to(device)
        
        # trainingconfigure
        epochs = self.config.get('epochs', 50)
        best_val_loss = float('inf')
        best_model_state = None
        training_losses = []
        val_losses = []
        
        # trainingloop
        for epoch in range(epochs):
            # training
            model.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch in train_loader:
                try:
                    features = batch['features'].to(device)
                    targets = batch['targets'].to(device)
                    stock_ids = batch['stock_id'].to(device)
                    
                    optimizer.zero_grad()
                    
                    predictions = model(features, stock_ids.squeeze())
                    target_dict = {col: targets[:, i] for i, col in enumerate(self.config['target_columns'])}
                    
                    loss_dict = criterion(predictions, target_dict)
                    loss = loss_dict['total_loss']
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                    train_batches += 1
                    
                    if train_batches >= 100:  # training
                        break
                        
                except Exception as e:
                    logger.error(f"trainingerror: {e}")
                    continue
            
            avg_train_loss = train_loss / max(train_batches, 1)
            training_losses.append(avg_train_loss)
            
            # verification
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    try:
                        features = batch['features'].to(device)
                        targets = batch['targets'].to(device)
                        stock_ids = batch['stock_id'].to(device)
                        
                        predictions = model(features, stock_ids.squeeze())
                        target_dict = {col: targets[:, i] for i, col in enumerate(self.config['target_columns'])}
                        
                        loss_dict = criterion(predictions, target_dict)
                        loss = loss_dict['total_loss']
                        
                        val_loss += loss.item()
                        val_batches += 1
                        
                        if val_batches >= 50:  # verification
                            break
                            
                    except Exception as e:
                        logger.error(f"verificationerror: {e}")
                        continue
            
            avg_val_loss = val_loss / max(val_batches, 1)
            val_losses.append(avg_val_loss)
            
            # savemodel
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # loadmodeltest
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        # test - predict
        model.eval()
        test_predictions = []
        test_targets = []
        test_stock_ids = []
        
        with torch.no_grad():
            for batch in test_loader:
                try:
                    features = batch['features'].to(device)
                    targets = batch['targets'].to(device)
                    stock_ids = batch['stock_id'].to(device)
                    
                    predictions = model(features, stock_ids.squeeze())
                    
                    # predict
                    for col_idx, col in enumerate(self.config['target_columns']):
                        if col in predictions:
                            test_predictions.append({
                                'target': col,
                                'predictions': predictions[col].cpu().numpy(),
                                'targets': targets[:, col_idx].cpu().numpy(),
                                'stock_ids': stock_ids.cpu().numpy()
                            })
                    
                    if len(test_predictions) >= 100:  # testsample
                        break
                        
                except Exception as e:
                    logger.error(f"testerror: {e}")
                    continue
        
        # calculatetestmetrics
        test_metrics = self._calculate_test_metrics(test_predictions)
        
        # savemodel
        window_id = f"{min(train_years)}-{max(train_years)}_{test_year}"
        model_path = self.output_dir / f"model_{window_id}.pth"
        torch.save({
            'model_state_dict': best_model_state,
            'config': self.config,
            'train_years': train_years,
            'test_year': test_year,
            'training_losses': training_losses,
            'val_losses': val_losses,
            'test_metrics': test_metrics
        }, model_path)
        
        results = {
            'window_id': window_id,
            'train_years': train_years,
            'test_year': test_year,
            'best_val_loss': best_val_loss,
            'test_metrics': test_metrics,
            'model_path': str(model_path),
            'training_losses': training_losses,
            'val_losses': val_losses
        }
        
        logger.info(f" {window_id} trainingcomplete:")
        logger.info(f"  verificationloss: {best_val_loss:.6f}")
        for metric, value in test_metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                logger.info(f"  {metric}: {value:.4f}")
        
        return results

    def _calculate_test_metrics(self, test_predictions: List[Dict]) -> Dict[str, float]:
        """calculatetestmetrics"""
        metrics = {}
        
        for target_col in self.config['target_columns']:
            col_predictions = []
            col_targets = []
            
            for pred_dict in test_predictions:
                if pred_dict['target'] == target_col:
                    col_predictions.extend(pred_dict['predictions'].flatten())
                    col_targets.extend(pred_dict['targets'].flatten())
            
            if len(col_predictions) > 10:
                pred_array = np.array(col_predictions)
                target_array = np.array(col_targets)
                
                # NaN
                mask = ~(np.isnan(pred_array) | np.isnan(target_array))
                if mask.sum() > 10:
                    pred_clean = pred_array[mask]
                    target_clean = target_array[mask]
                    
                    # calculatemetrics
                    ic = np.corrcoef(pred_clean, target_clean)[0, 1]
                    rank_ic = np.corrcoef(np.argsort(pred_clean), np.argsort(target_clean))[0, 1]
                    mse = np.mean((pred_clean - target_clean) ** 2)
                    
                    metrics[f'{target_col}_ic'] = ic
                    metrics[f'{target_col}_rank_ic'] = rank_ic
                    metrics[f'{target_col}_mse'] = mse
                    metrics[f'{target_col}_samples'] = len(pred_clean)
        
        return metrics

    def run_yearly_rolling_training(self) -> Dict[str, Any]:
        """runtraining"""
        logger.info("begintraining...")
        
        if len(self.available_years) < self.min_train_years + 1:
            raise ValueError(f"data {self.min_train_years + 1} data")
        
        # predictbegin
        start_year = self.prediction_start_year or self.available_years[self.min_train_years]
        prediction_years = [year for year in self.available_years if year >= start_year]
        
        if not prediction_years:
            raise ValueError("predict")
        
        logger.info(f"predict: {prediction_years}")
        
        all_results = {}
        
        # Execute rolling training for each prediction year
        for test_year in prediction_years:
            # Determine training years
            available_train_years = [year for year in self.available_years if year < test_year]
            
            if len(available_train_years) < self.min_train_years:
                logger.warning(f"Skipping year {test_year}: insufficient training data")
                continue
            
            # Use recent years as training data
            train_years = available_train_years[-self.max_train_years:]
            
            try:
                # Train this window
                window_results = self._train_single_window(train_years, test_year)
                all_results[test_year] = window_results
                self.training_history.append(window_results)
                
            except Exception as e:
                logger.error(f"Year {test_year} training failed: {e}")
                continue
        
        # Save overall results
        summary_file = self.output_dir / "yearly_rolling_results.json"
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Generate summary report
        self._generate_summary_report(all_results)
        
        logger.info(f"Yearly rolling training completed! Trained {len(all_results)} year windows")
        logger.info(f"Results saved to: {summary_file}")
        
        return all_results

    def _generate_summary_report(self, results: Dict[str, Any]):
        """Generate summary report"""
        report = {
            'summary': {
                'total_windows': len(results),
                'prediction_years': list(results.keys()),
                'avg_metrics_by_target': {}
            },
            'yearly_results': results
        }
        
        # Calculate average metrics for each target
        for target_col in self.config['target_columns']:
            ic_values = []
            rank_ic_values = []
            
            for window_results in results.values():
                metrics = window_results.get('test_metrics', {})
                ic_key = f'{target_col}_ic'
                rank_ic_key = f'{target_col}_rank_ic'
                
                if ic_key in metrics and not np.isnan(metrics[ic_key]):
                    ic_values.append(metrics[ic_key])
                if rank_ic_key in metrics and not np.isnan(metrics[rank_ic_key]):
                    rank_ic_values.append(metrics[rank_ic_key])
            
            if ic_values:
                report['summary']['avg_metrics_by_target'][target_col] = {
                    'avg_ic': np.mean(ic_values),
                    'std_ic': np.std(ic_values),
                    'avg_rank_ic': np.mean(rank_ic_values) if rank_ic_values else np.nan,
                    'num_windows': len(ic_values)
                }
        
        # Save report
        report_file = self.output_dir / "yearly_rolling_summary_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Log to logger
        logger.info("=" * 80)
        logger.info("Yearly Rolling Training Summary Report")
        logger.info("=" * 80)
        logger.info(f"Total training windows: {report['summary']['total_windows']}")
        logger.info(f"Prediction years: {report['summary']['prediction_years']}")
        
        for target_col, metrics in report['summary']['avg_metrics_by_target'].items():
            logger.info(f"{target_col}:")
            logger.info(f"  Average IC: {metrics['avg_ic']:.4f} (±{metrics['std_ic']:.4f})")
            logger.info(f"  Average Rank IC: {metrics['avg_rank_ic']:.4f}")
            logger.info(f"  Valid windows: {metrics['num_windows']}")
        
        logger.info("=" * 80)


def create_yearly_rolling_trainer(data_dir: str, model_factory, output_dir: str, config: Dict[str, Any]) -> YearlyRollingTrainer:
    """Factory function to create yearly rolling trainer"""
    return YearlyRollingTrainer(
        data_dir=data_dir,
        model_factory=model_factory,
        output_dir=output_dir,
        config=config
    )
