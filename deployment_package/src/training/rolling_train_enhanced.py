#!/usr/bin/env python3
"""
Enhanced Rolling Window Training Script
Implements efficient data loading and model training with rolling windows
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
from typing import Dict, List, Tuple, Optional, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import warnings
from pathlib import Path
import glob
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
# Only suppress specific warnings that are known to be harmless
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, message='.*NumPy.*')

from configs.config import ModelConfig
from src.data_processing.data_processor import MultiFileDataset, MultiFileDataProcessor
from src.models.models import FactorForecastingModel
from src.training.trainer import FactorForecastingTrainer, CorrelationLoss
from src.utils.ic_analysis import ICAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rolling_train_enhanced.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedRollingWindowTrainer:
    """Enhanced rolling window trainer with efficient data loading and caching"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device(getattr(self.config, 'device', 'cpu'))
        self.results = {}
        self.data_cache = {}  # Cache for loaded data
        self.data_processor = MultiFileDataProcessor(config)
        
        # Create output directories
        self.output_dir = Path(self.config.output_dir)
        self.models_dir = self.output_dir / "models"
        self.results_dir = self.output_dir / "rolling_results"
        self.cache_dir = self.output_dir / "cache"
        
        for dir_path in [self.output_dir, self.models_dir, self.results_dir, self.cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_available_years(self) -> List[int]:
        """Get list of available years from data files"""
        data_dir = Path(self.config.data_dir)
        if not data_dir.exists():
            logger.warning(f"Data directory {data_dir} does not exist")
            return []
        
        # Look for parquet files
        parquet_files = list(data_dir.glob("*.parquet"))
        if not parquet_files:
            logger.warning(f"No parquet files found in {data_dir}")
            return []
        
        years = set()
        for file_path in parquet_files:
            try:
                # Try to extract year from filename
                filename = file_path.stem
                
                # Try YYYY-MM-DD format
                if len(filename) == 10 and filename[4] == '-' and filename[7] == '-':
                    year = int(filename[:4])
                    years.add(year)
                # Try YYYYMMDD format
                elif len(filename) == 8:
                    year = int(filename[:4])
                    years.add(year)
                # Try to read from file content
                else:
                    df_sample = pd.read_parquet(file_path, nrows=1000)
                    if 'date' in df_sample.columns:
                        df_sample['date'] = pd.to_datetime(df_sample['date'])
                        years.update(df_sample['date'].dt.year.unique())
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
        
        return sorted(list(years))
    
    def load_year_data(self, year: int, use_cache: bool = True) -> pd.DataFrame:
        """Load data for a specific year with caching"""
        cache_key = f"year_{year}"
        
        # Check cache first
        if use_cache and cache_key in self.data_cache:
            logger.info(f"Using cached data for year {year}")
            return self.data_cache[cache_key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if use_cache and cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                self.data_cache[cache_key] = cached_data
                logger.info(f"Loaded cached data for year {year} from disk")
                return cached_data
            except Exception as e:
                logger.warning(f"Error loading cache for year {year}: {e}")
        
        logger.info(f"Loading data for year {year}")
        start_time = time.time()
        
        # Find all files for the year
        data_dir = Path(self.config.data_dir)
        year_files = []
        
        for file_path in data_dir.glob("*.parquet"):
            try:
                filename = file_path.stem
                
                # Check if file contains data for the target year
                if len(filename) == 10 and filename[4] == '-' and filename[7] == '-':
                    file_year = int(filename[:4])
                    if file_year == year:
                        year_files.append(file_path)
                elif len(filename) == 8:
                    file_year = int(filename[:4])
                    if file_year == year:
                        year_files.append(file_path)
                else:
                    # Check file content
                    df_sample = pd.read_parquet(file_path, nrows=1000)
                    if 'date' in df_sample.columns:
                        df_sample['date'] = pd.to_datetime(df_sample['date'])
                        if (df_sample['date'].dt.year == year).any():
                            year_files.append(file_path)
            except Exception as e:
                logger.warning(f"Error checking {file_path}: {e}")
        
        if not year_files:
            logger.warning(f"No data files found for year {year}, creating synthetic data")
            return self._create_synthetic_data(year)
        
        # Load and combine all files for the year
        dfs = []
        for file_path in year_files:
            try:
                df = pd.read_parquet(file_path)
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df_year = df[df['date'].dt.year == year]
                    if not df_year.empty:
                        dfs.append(df_year)
                        logger.info(f"Loaded {len(df_year)} records from {file_path.name}")
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
        
        if not dfs:
            logger.warning(f"No valid data found for year {year}, creating synthetic data")
            return self._create_synthetic_data(year)
        
        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Clean and preprocess data
        combined_df = self._clean_and_preprocess_data(combined_df)
        
        # Cache the data
        self.data_cache[cache_key] = combined_df
        
        # Save to disk cache
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(combined_df, f)
        except Exception as e:
            logger.warning(f"Error saving cache for year {year}: {e}")
        
        load_time = time.time() - start_time
        logger.info(f"Loaded {len(combined_df)} records for year {year} in {load_time:.2f}s")
        
        return combined_df
    
    def _clean_and_preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the data"""
        original_len = len(df)
        
        # Remove limit up/down data
        if self.config.limit_up_down_column in df.columns:
            df = df[df[self.config.limit_up_down_column] != 1]
            logger.info(f"Removed limit up/down data: {original_len} -> {len(df)} records")
        
        # Handle missing values in factor columns
        factor_cols = self.config.factor_columns
        if factor_cols:
            df.loc[:, factor_cols] = df[factor_cols].ffill().bfill()
        
        # Remove rows with missing target values
        target_cols = self.config.target_columns
        if target_cols:
            df = df.dropna(subset=target_cols)
            logger.info(f"After removing missing targets: {len(df)} records")
        
        # Ensure stock ID column exists
        if self.config.stock_id_column not in df.columns:
            if 'sid' in df.columns:
                df[self.config.stock_id_column] = df['sid']
            else:
                # Create synthetic stock IDs
                df[self.config.stock_id_column] = range(len(df))
        
        return df
    
    def _create_synthetic_data(self, year: int) -> pd.DataFrame:
        """Create synthetic data for testing"""
        logger.info(f"Creating synthetic data for year {year}")
        
        dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq='D')
        sids = range(100, 200)
        data = []
        
        for date in dates:
            for sid in sids:
                row = {'date': date, self.config.stock_id_column: sid}
                
                # Add factor columns
                for i in range(100):
                    row[str(i)] = np.random.normal(0, 1)
                
                # Add target columns
                for target in self.config.target_columns:
                    row[target] = np.random.normal(0, 1)
                
                # Add limit up/down column
                row[self.config.limit_up_down_column] = 0
                
                data.append(row)
        
        df = pd.DataFrame(data)
        logger.info(f"Created synthetic data: {len(df)} records for year {year}")
        return df
    
    def prepare_rolling_data(self, train_years: List[int], test_year: int) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and test data for rolling window"""
        logger.info(f"Preparing data: train on {train_years}, test on {test_year}")
        
        # Load training data
        train_dfs = []
        for year in train_years:
            df = self.load_year_data(year)
            train_dfs.append(df)
        
        train_df = pd.concat(train_dfs, ignore_index=True)
        logger.info(f"Training data: {len(train_df)} records from years {train_years}")
        
        # Load test data
        test_df = self.load_year_data(test_year)
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
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        return train_loader, test_loader
    
    def train_model(self, train_loader: DataLoader, experiment_name: str) -> FactorForecastingModel:
        """Train model for one rolling window"""
        logger.info(f"Starting training for {experiment_name}")
        
        # Create model
        model = FactorForecastingModel(self.config).to(self.device)
        
        # Create loss function
        loss_fn = CorrelationLoss(
            correlation_weight=self.config.correlation_weight,
            mse_weight=self.config.mse_weight,
            target_correlations=self.config.target_correlations
        )
        
        # Create trainer
        trainer = FactorForecastingTrainer(
            model=model,
            config=self.config.to_dict(),
            loss_fn=loss_fn,
            experiment_name=experiment_name
        )
        
        # Train model
        trainer.train(train_loader, None, self.config.num_epochs)
        
        return model
    
    def evaluate_model(self, model: FactorForecastingModel, test_loader: DataLoader, 
                      test_year: int) -> Dict[str, float]:
        """Evaluate model and compute comprehensive metrics"""
        logger.info(f"Evaluating model for year {test_year}")
        
        model.eval()
        all_predictions = {target: [] for target in self.config.target_columns}
        all_targets = {target: [] for target in self.config.target_columns}
        all_stock_ids = []
        
        with torch.no_grad():
            for batch in test_loader:
                features = batch['features'].to(self.device)
                stock_ids = batch['stock_ids'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                # Get predictions
                predictions = model(features, stock_ids)
                
                # Collect data
                for target_name in self.config.target_columns:
                    if target_name in predictions and target_name in targets:
                        all_predictions[target_name].append(predictions[target_name].cpu())
                        all_targets[target_name].append(targets[target_name].cpu())
                
                all_stock_ids.append(stock_ids.cpu())
        
        # Concatenate all data
        final_predictions = {}
        final_targets = {}
        
        for target_name in self.config.target_columns:
            if all_predictions[target_name]:
                final_predictions[target_name] = torch.cat(all_predictions[target_name], dim=0).numpy()
                final_targets[target_name] = torch.cat(all_targets[target_name], dim=0).numpy()
        
        all_stock_ids = torch.cat(all_stock_ids, dim=0).numpy()
        
        # Compute metrics
        metrics = {}
        ic_analyzer = ICAnalyzer()
        
        for target_name in self.config.target_columns:
            if target_name in final_predictions and target_name in final_targets:
                pred = final_predictions[target_name]
                target = final_targets[target_name]
                
                # Remove NaN values
                mask = ~(np.isnan(pred) | np.isnan(target))
                if mask.sum() > 1:
                    pred_clean = pred[mask]
                    target_clean = target[mask]
                    stock_ids_clean = all_stock_ids[mask]
                    
                    # Basic correlations
                    correlation = np.corrcoef(pred_clean, target_clean)[0, 1]
                    metrics[f"{target_name}_correlation"] = correlation
                    
                    # Spearman rank correlation
                    from scipy.stats import spearmanr
                    rank_correlation = spearmanr(pred_clean, target_clean)[0]
                    metrics[f"{target_name}_rank_correlation"] = rank_correlation
                    
                    # MSE and MAE
                    mse = np.mean((pred_clean - target_clean) ** 2)
                    mae = np.mean(np.abs(pred_clean - target_clean))
                    metrics[f"{target_name}_mse"] = mse
                    metrics[f"{target_name}_mae"] = mae
                    
                    # IC analysis
                    ic_metrics = ic_analyzer.compute_ic_metrics(pred_clean, target_clean, stock_ids_clean)
                    for key, value in ic_metrics.items():
                        metrics[f"{target_name}_{key}"] = value
                    
                    logger.info(f"{target_name}: Corr={correlation:.4f}, Rank={rank_correlation:.4f}, "
                              f"MSE={mse:.4f}, MAE={mae:.4f}")
                else:
                    logger.warning(f"Not enough valid data for {target_name}")
                    for metric in ['correlation', 'rank_correlation', 'mse', 'mae', 'ic_mean', 'ic_std']:
                        metrics[f"{target_name}_{metric}"] = np.nan
        
        # Compute average metrics
        valid_correlations = [v for k, v in metrics.items() 
                            if 'correlation' in k and not np.isnan(v)]
        if valid_correlations:
            metrics['avg_correlation'] = np.mean(valid_correlations)
            metrics['avg_rank_correlation'] = np.mean([v for k, v in metrics.items() 
                                                     if 'rank_correlation' in k and not np.isnan(v)])
        
        # Save detailed results
        self._save_detailed_results(final_predictions, final_targets, all_stock_ids, test_year)
        
        return metrics
    
    def _save_detailed_results(self, predictions: Dict, targets: Dict, stock_ids: np.ndarray, test_year: int):
        """Save detailed prediction results"""
        results_dir = self.results_dir / str(test_year)
        results_dir.mkdir(exist_ok=True)
        
        # Save predictions
        results_df = pd.DataFrame()
        for target_name in self.config.target_columns:
            if target_name in predictions and target_name in targets:
                results_df[f'predictions_{target_name}'] = predictions[target_name]
                results_df[f'targets_{target_name}'] = targets[target_name]
        
        results_df['stock_id'] = stock_ids
        results_df.to_parquet(results_dir / "predictions.parquet")
        
        logger.info(f"Saved detailed results to {results_dir}")
    
    def run_rolling_window_training(self):
        """Run complete rolling window training"""
        logger.info("Starting enhanced rolling window training")
        
        # Get available years
        available_years = self.get_available_years()
        if not available_years:
            logger.error("No data years available")
            return {}
        
        logger.info(f"Available years: {available_years}")
        
        # Define training windows
        training_windows = []
        for pred_year in self.config.prediction_years:
            if pred_year in available_years:
                # Use rolling window of previous years
                start_year = max(available_years[0], pred_year - self.config.rolling_window_years)
                train_years = list(range(start_year, pred_year))
                
                if len(train_years) >= self.config.min_train_years:
                    training_windows.append((train_years, pred_year))
        
        logger.info(f"Training windows: {training_windows}")
        
        if not training_windows:
            logger.error("No valid training windows found")
            return {}
        
        # Run training for each window
        all_results = {}
        
        for train_years, test_year in training_windows:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training on years {train_years}, predicting year {test_year}")
            logger.info(f"{'='*60}")
            
            try:
                # Prepare data
                train_loader, test_loader = self.prepare_rolling_data(train_years, test_year)
                
                # Train model
                experiment_name = f"rolling_{min(train_years)}_{max(train_years)}_predict_{test_year}"
                model = self.train_model(train_loader, experiment_name)
                
                # Evaluate model
                metrics = self.evaluate_model(model, test_loader, test_year)
                
                # Save model
                model_path = self.models_dir / f"{experiment_name}_model.pth"
                torch.save(model.state_dict(), model_path)
                
                # Store results
                all_results[test_year] = {
                    'train_years': train_years,
                    'metrics': metrics,
                    'model_path': str(model_path)
                }
                
                logger.info(f"Completed training for year {test_year}")
                logger.info(f"Average correlation: {metrics.get('avg_correlation', 'N/A')}")
                
            except Exception as e:
                logger.error(f"Error in training window {train_years} -> {test_year}: {e}")
                all_results[test_year] = {
                    'train_years': train_years,
                    'error': str(e)
                }
        
        # Save summary results
        self._save_summary_results(all_results)
        
        # Print final summary
        self._print_final_summary(all_results)
        
        return all_results
    
    def _save_summary_results(self, all_results: Dict):
        """Save summary of all results"""
        summary_data = []
        
        for test_year, result in all_results.items():
            if 'metrics' in result:
                row = {
                    'test_year': test_year,
                    'train_years': str(result['train_years']),
                    'model_path': result['model_path']
                }
                row.update(result['metrics'])
                summary_data.append(row)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_path = self.results_dir / "summary.csv"
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"Saved summary results to {summary_path}")
    
    def _print_final_summary(self, all_results: Dict):
        """Print final training summary"""
        logger.info("\n" + "="*80)
        logger.info("ENHANCED ROLLING WINDOW TRAINING SUMMARY")
        logger.info("="*80)
        
        successful_results = {k: v for k, v in all_results.items() if 'metrics' in v}
        failed_results = {k: v for k, v in all_results.items() if 'error' in v}
        
        logger.info(f"\nSuccessful predictions: {len(successful_results)}")
        logger.info(f"Failed predictions: {len(failed_results)}")
        
        if successful_results:
            logger.info("\nDetailed Results:")
            for test_year, result in successful_results.items():
                logger.info(f"\nYear {test_year} (trained on {result['train_years']}):")
                metrics = result['metrics']
                
                # Print correlation metrics
                for target in self.config.target_columns:
                    corr_key = f"{target}_correlation"
                    if corr_key in metrics:
                        logger.info(f"  {target} correlation: {metrics[corr_key]:.4f}")
                
                if 'avg_correlation' in metrics:
                    logger.info(f"  Average correlation: {metrics['avg_correlation']:.4f}")
        
        if failed_results:
            logger.info("\nFailed Results:")
            for test_year, result in failed_results.items():
                logger.info(f"  Year {test_year}: {result['error']}")

def main():
    parser = argparse.ArgumentParser(description="Enhanced rolling window training for factor forecasting")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to configuration YAML file")
    parser.add_argument("--data_dir", type=str, default=None,
                       help="Data directory path")
    parser.add_argument("--prediction_years", type=str, default=None,
                       help="Comma-separated list of years to predict")
    parser.add_argument("--rolling_window_years", type=int, default=None,
                       help="Number of years to use for training")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=None,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=None,
                       help="Learning rate")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        config = ModelConfig.load_config_from_yaml(args.config)
    else:
        config = ModelConfig()
    
    # Override with command line arguments
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.prediction_years:
        config.prediction_years = [int(y) for y in args.prediction_years.split(',')]
    if args.rolling_window_years:
        config.rolling_window_years = args.rolling_window_years
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    
    # Create trainer and run
    trainer = EnhancedRollingWindowTrainer(config)
    results = trainer.run_rolling_window_training()
    
    logger.info("Enhanced rolling window training completed!")

if __name__ == "__main__":
    main() 