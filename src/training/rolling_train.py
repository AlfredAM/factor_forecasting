#!/usr/bin/env python3
"""
Rolling window training and prediction script
Trains on previous years and predicts next year, focusing on correlation optimization
"""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import warnings
# Only suppress specific warnings that are known to be harmless
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, message='.*NumPy.*')

# Add project root to path
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import ModelConfig
from src.data_processing.data_processor import MultiFileDataset
from src.models.models import FactorForecastingModel
from src.training.trainer import FactorForecastingTrainer, MetricsTracker, CorrelationLoss
from src.inference.inference import ModelInference

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rolling_train.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RollingWindowTrainer:
    """Rolling window trainer for time series forecasting"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device(getattr(self.config, 'device', 'cpu'))
        self.results = {}
        
    def get_year_data(self, year: int) -> pd.DataFrame:
        """Load data for a specific year"""
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        
        logger.info(f"Loading data for year {year}: {start_date} to {end_date}")
        
        # Try to load from different possible directories
        data_dirs = [
            getattr(self.config, 'data_dir', None),
            "/nas/feature_v2",
            "/data",
            "/home/ecs-user/data",
            "/home/ecs-user/factor_forecasting"
        ]
        
        for data_dir in data_dirs:
            if data_dir and os.path.exists(data_dir):
                logger.info(f"Found data directory: {data_dir}")
                try:
                    # Look for parquet files in the directory
                    parquet_files = []
                    for file in os.listdir(data_dir):
                        if file.endswith('.parquet'):
                            file_path = os.path.join(data_dir, file)
                            # Check if file contains data for the target year
                            try:
                                df_sample = pd.read_parquet(file_path, nrows=1000)
                                if 'date' in df_sample.columns:
                                    df_sample['date'] = pd.to_datetime(df_sample['date'])
                                    if (df_sample['date'].dt.year == year).any():
                                        parquet_files.append(file_path)
                            except Exception as e:
                                logger.warning(f"Error reading {file_path}: {e}")
                    
                    if parquet_files:
                        logger.info(f"Found {len(parquet_files)} parquet files for year {year}")
                        # Load and combine all files for the year
                        dfs = []
                        for file_path in parquet_files:
                            try:
                                df = pd.read_parquet(file_path)
                                if 'date' in df.columns:
                                    df['date'] = pd.to_datetime(df['date'])
                                    df_year = df[df['date'].dt.year == year]
                                    if not df_year.empty:
                                        dfs.append(df_year)
                            except Exception as e:
                                logger.warning(f"Error loading {file_path}: {e}")
                        
                        if dfs:
                            combined_df = pd.concat(dfs, ignore_index=True)
                            logger.info(f"Loaded {len(combined_df)} records for year {year}")
                            return combined_df
                except Exception as e:
                    logger.warning(f"Error loading from {data_dir}: {e}")
        
        # If no data found, create synthetic data for testing
        logger.warning(f"No data found for year {year}, creating synthetic data")
        return self._create_synthetic_data(year)
    
    def _create_synthetic_data(self, year: int) -> pd.DataFrame:
        """Create synthetic data for testing"""
        dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq='D')
        sids = range(100, 200)
        data = []
        
        for date in dates:
            for sid in sids:
                row = {'date': date, 'sid': sid}
                # Add factor columns (0-99)
                for i in range(100):
                    row[str(i)] = np.random.normal(0, 1)
                # Add target columns
                row['intra30m'] = np.random.normal(0, 1)
                row['nextT1d'] = np.random.normal(0, 1)
                row['ema1d'] = np.random.normal(0, 1)
                row['luld'] = 0  # Non-limit up/down
                data.append(row)
        
        df = pd.DataFrame(data)
        logger.info(f"Created synthetic data: {len(df)} records for year {year}")
        return df
    
    def prepare_data(self, train_years: List[int], test_year: int) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and test data"""
        # Load training data
        train_dfs = []
        for year in train_years:
            df = self.get_year_data(year)
            train_dfs.append(df)
        
        train_df = pd.concat(train_dfs, ignore_index=True)
        logger.info(f"Training data: {len(train_df)} records from years {train_years}")
        
        # Load test data
        test_df = self.get_year_data(test_year)
        logger.info(f"Test data: {len(test_df)} records from year {test_year}")
        
        # Create datasets
        train_dataset = MultiFileDataset(
            dataframes=[train_df],
            config=self.config,
            mode='train'
        )
        
        test_dataset = MultiFileDataset(
            dataframes=[test_df],
            config=self.config,
            mode='test'
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=getattr(self.config, 'batch_size', 64),
            shuffle=True,
            num_workers=getattr(self.config, 'num_workers', 4),
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=getattr(self.config, 'batch_size', 64),
            shuffle=False,
            num_workers=getattr(self.config, 'num_workers', 4),
            pin_memory=True
        )
        
        return train_loader, test_loader
    
    def train_model(self, train_loader: DataLoader, experiment_name: str) -> FactorForecastingModel:
        """Train model for one rolling window with enhanced correlation optimization"""
        # Create model
        model = FactorForecastingModel(self.config).to(self.device)
        
        # Create enhanced loss function
        loss_fn = CorrelationLoss(
            correlation_weight=getattr(self.config, 'correlation_weight', 1.0),
            mse_weight=getattr(self.config, 'mse_weight', 0.1),
            rank_weight=0.1,  # Add rank correlation weight
            target_correlations=getattr(self.config, 'target_correlations', [0.1, 0.05, 0.08])
        )
        
        # Create trainer
        trainer = FactorForecastingTrainer(
            model=model,
            config=self.config.__dict__,
            loss_fn=loss_fn,
            experiment_name=experiment_name
        )
        
        # Train model
        logger.info(f"Starting training for {experiment_name}")
        logger.info(f"Target correlations: {getattr(self.config, 'target_correlations', [0.1, 0.05, 0.08])}")
        trainer.train(train_loader, None, getattr(self.config, 'num_epochs', 100))  # No validation during rolling window training
        
        return model
    
    def evaluate_predictions(self, model: FactorForecastingModel, data_loader: DataLoader, 
                           year: int, split: str = "test") -> Dict[str, float]:
        """Evaluate model predictions and compute correlations for three targets
        Args:
            model: Trained model
            data_loader: DataLoader for evaluation (train or test)
            year: Target year for bookkeeping
            split: 'train' for in-sample, 'test' for out-of-sample
        """
        model.eval()
        all_predictions = {target: [] for target in getattr(self.config, 'target_columns', ['intra30m', 'nextT1d', 'ema1d'])}
        all_targets = {target: [] for target in getattr(self.config, 'target_columns', ['intra30m', 'nextT1d', 'ema1d'])}
        
        with torch.no_grad():
            for batch in data_loader:
                features = batch['features'].to(self.device)
                stock_ids = batch['stock_ids'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                # Get model predictions
                predictions = model(features, stock_ids)
                
                # Collect predictions and targets for each target
                for target_name in getattr(self.config, 'target_columns', ['intra30m', 'nextT1d', 'ema1d']):
                    if target_name in predictions and target_name in targets:
                        all_predictions[target_name].append(predictions[target_name].cpu())
                        all_targets[target_name].append(targets[target_name].cpu())
        
        # Concatenate all predictions and targets
        final_predictions = {}
        final_targets = {}
        
        for target_name in getattr(self.config, 'target_columns', ['intra30m', 'nextT1d', 'ema1d']):
            if all_predictions[target_name]:
                final_predictions[target_name] = torch.cat(all_predictions[target_name], dim=0).numpy()
                final_targets[target_name] = torch.cat(all_targets[target_name], dim=0).numpy()
        
        # Compute correlations for each target
        correlations = {}
        
        for target_name in getattr(self.config, 'target_columns', ['intra30m', 'nextT1d', 'ema1d']):
            if target_name in final_predictions and target_name in final_targets:
                pred = final_predictions[target_name]
                target = final_targets[target_name]
                
                # Remove NaN values
                mask = ~(np.isnan(pred) | np.isnan(target))
                if mask.sum() > 1:
                    pred_clean = pred[mask]
                    target_clean = target[mask]
                    
                    # Pearson correlation
                    correlation = np.corrcoef(pred_clean, target_clean)[0, 1]
                    correlations[f"{target_name}_correlation"] = correlation
                    
                    # Spearman rank correlation
                    from scipy.stats import spearmanr
                    rank_correlation = spearmanr(pred_clean, target_clean)[0]
                    correlations[f"{target_name}_rank_correlation"] = rank_correlation
                    
                    # MSE
                    mse = np.mean((pred_clean - target_clean) ** 2)
                    correlations[f"{target_name}_mse"] = mse
                    
                    # IC (Information Coefficient)
                    ic = np.mean(pred_clean * target_clean)
                    correlations[f"{target_name}_ic"] = ic
                    
                    logger.info(f"{target_name} - Pearson: {correlation:.4f}, "
                              f"Spearman: {rank_correlation:.4f}, MSE: {mse:.4f}, IC: {ic:.4f}")
                else:
                    correlations[f"{target_name}_correlation"] = np.nan
                    correlations[f"{target_name}_rank_correlation"] = np.nan
                    correlations[f"{target_name}_mse"] = np.nan
                    correlations[f"{target_name}_ic"] = np.nan
                    logger.warning(f"Not enough valid data for {target_name}")
        
        # Compute average correlation across all targets
        valid_correlations = [v for k, v in correlations.items() 
                            if 'correlation' in k and not np.isnan(v)]
        if valid_correlations:
            correlations['avg_correlation'] = np.mean(valid_correlations)
            correlations['avg_rank_correlation'] = np.mean([v for k, v in correlations.items() 
                                                          if 'rank_correlation' in k and not np.isnan(v)])
            logger.info(f"Average correlation: {correlations['avg_correlation']:.4f}")
            logger.info(f"Average rank correlation: {correlations['avg_rank_correlation']:.4f}")
        
        # Save predictions for analysis
        results_df = pd.DataFrame()
        for target_name in getattr(self.config, 'target_columns', ['intra30m', 'nextT1d', 'ema1d']):
            if target_name in final_predictions and target_name in final_targets:
                results_df[f'predictions_{target_name}'] = final_predictions[target_name]
                results_df[f'targets_{target_name}'] = final_targets[target_name]
        
        results_dir = f"outputs/rolling_results/{year}/{split}"
        os.makedirs(results_dir, exist_ok=True)
        results_df.to_parquet(f"{results_dir}/predictions.parquet")
        
        # Save correlation summary
        summary_df = pd.DataFrame([correlations])
        summary_df.to_csv(f"{results_dir}/correlations.csv", index=False)
        
        return correlations
    
    def run_rolling_window_training(self):
        """Run rolling window training and prediction"""
        logger.info("Starting rolling window training")
        
        # Define training windows
        training_windows = []
        for pred_year in getattr(self.config, 'prediction_years', [2023]):
            train_years = list(range(2018, pred_year))
            if len(train_years) >= getattr(self.config, 'min_train_years', 2):
                training_windows.append((train_years, pred_year))
        
        logger.info(f"Training windows: {training_windows}")
        
        # Run training for each window
        all_results = {}
        
        for train_years, test_year in training_windows:
            logger.info(f"\n{'='*50}")
            logger.info(f"Training on years {train_years}, predicting year {test_year}")
            logger.info(f"{'='*50}")
            
            try:
                # Prepare data
                train_loader, test_loader = self.prepare_data(train_years, test_year)
                
                # Train model
                experiment_name = f"rolling_{min(train_years)}_{max(train_years)}_predict_{test_year}"
                model = self.train_model(train_loader, experiment_name)
                
                # Evaluate predictions (both in-sample and out-of-sample)
                train_correlations = self.evaluate_predictions(model, train_loader, test_year, split="train")
                test_correlations = self.evaluate_predictions(model, test_loader, test_year, split="test")
                
                # Create models directory if it doesn't exist
                models_dir = Path("models")
                models_dir.mkdir(exist_ok=True)
                model_path = models_dir / f"{experiment_name}_model.pth"
                
                # Store results
                all_results[test_year] = {
                    'train_years': train_years,
                    'train_correlations': train_correlations,
                    'test_correlations': test_correlations,
                    'model_path': str(model_path)
                }
                
                # Save model
                torch.save(model.state_dict(), model_path)
                
                logger.info(f"Completed training for year {test_year}")
                logger.info(f"In-sample correlations: {train_correlations}")
                logger.info(f"Out-of-sample correlations: {test_correlations}")
                
            except Exception as e:
                logger.error(f"Error in training window {train_years} -> {test_year}: {e}")
                all_results[test_year] = {
                    'train_years': train_years,
                    'error': str(e)
                }
        
        # Save overall results
        results_df = pd.DataFrame()
        for test_year, result in all_results.items():
            if 'test_correlations' in result:
                row = {'test_year': test_year, 'train_years': str(result['train_years'])}
                # Prefix keys to distinguish
                row.update({f"train_{k}": v for k, v in result['train_correlations'].items()})
                row.update({f"test_{k}": v for k, v in result['test_correlations'].items()})
                results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
        
        results_df.to_csv("outputs/rolling_results/summary.csv", index=False)
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("ROLLING WINDOW TRAINING SUMMARY")
        logger.info("="*80)
        
        for test_year, result in all_results.items():
            if 'test_correlations' in result:
                logger.info(f"\nYear {test_year} (trained on {result['train_years']}):")
                logger.info("  In-sample (train) correlations:")
                for key, value in result['train_correlations'].items():
                    if 'correlation' in key:
                        logger.info(f"    {key}: {value:.4f}")
                logger.info("  Out-of-sample (test) correlations:")
                for key, value in result['test_correlations'].items():
                    if 'correlation' in key:
                        logger.info(f"    {key}: {value:.4f}")
            else:
                logger.info(f"\nYear {test_year}: ERROR - Unknown error")
        
        return all_results

def main():
    parser = argparse.ArgumentParser(description="Rolling window training for factor forecasting")
    parser.add_argument("--data_dir", type=str, default="/nas/feature_v2_10s",
                       help="Data directory path")
    parser.add_argument("--start_date", type=str, default="2018-01-01",
                       help="Start date for data")
    parser.add_argument("--end_date", type=str, default="2022-12-31",
                       help="End date for data")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for training")
    parser.add_argument("--max_epochs", type=int, default=10,
                       help="Maximum number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--correlation_weight", type=float, default=1.0,
                       help="Weight for correlation loss")
    parser.add_argument("--mse_weight", type=float, default=0.1,
                       help="Weight for MSE loss")
    
    args = parser.parse_args()
    
    # Create configuration
    config = ModelConfig()
    config.data_dir = args.data_dir
    config.start_date = args.start_date
    config.end_date = args.end_date
    config.batch_size = args.batch_size
    config.num_epochs = args.max_epochs
    config.learning_rate = args.learning_rate
    config.correlation_weight = args.correlation_weight
    config.mse_weight = args.mse_weight
    
    # Create trainer and run
    trainer = RollingWindowTrainer(config)
    results = trainer.run_rolling_window_training()
    
    logger.info("Rolling window training completed!")

if __name__ == "__main__":
    main()
