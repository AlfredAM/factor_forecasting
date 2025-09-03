#!/usr/bin/env python3
"""
Out-of-Sample Prediction and Correlation Analysis
Uses trained models to make predictions on test data and calculate correlations
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
# Only suppress specific warnings that are known to be harmless
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, message='.*NumPy.*')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.models.models import FactorForecastingModel
from src.data_processing.data_processor import MultiFileDataProcessor
from configs.config import ModelConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OutOfSamplePredictor:
    """Out-of-sample prediction and correlation analysis"""
    
    def __init__(self, model_path: str, config_path: str, device: str = 'auto'):
        """
        Initialize the predictor
        
        Args:
            model_path: Path to the trained model
            config_path: Path to the model configuration
            device: Device to use for inference
        """
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.device = self._get_device(device)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize model
        self.model = self._load_model()
        
        # Initialize data processor
        self.data_processor = MultiFileDataProcessor(self.config)
        
        logger.info(f"Initialized predictor with model: {self.model_path}")
        logger.info(f"Using device: {self.device}")
    
    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate device"""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
        else:
            return torch.device(device)
    
    def _load_config(self) -> ModelConfig:
        """Load model configuration"""
        with open(self.config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Create ModelConfig object
        config = ModelConfig()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        logger.info(f"Loaded configuration: {config}")
        return config
    
    def _load_model(self) -> FactorForecastingModel:
        """Load the trained model"""
        # Create model
        model = FactorForecastingModel(self.config)
        
        # Load trained weights
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        logger.info(f"Loaded model from: {self.model_path}")
        return model
    
    def load_test_data(self, test_data_dir: str) -> pd.DataFrame:
        """Load test data"""
        test_data_path = Path(test_data_dir)
        data_files = list(test_data_path.glob("*.parquet"))
        
        if not data_files:
            raise ValueError(f"No parquet files found in {test_data_dir}")
        
        # Load and concatenate all test data
        dataframes = []
        for file_path in sorted(data_files):
            logger.info(f"Loading test data: {file_path}")
            df = pd.read_parquet(file_path)
            dataframes.append(df)
        
        test_data = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Loaded test data shape: {test_data.shape}")
        
        return test_data
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare data for inference"""
        # Clean and preprocess data directly
        processed_data = self.data_processor._clean_data(data)
        processed_data = self.data_processor._preprocess_data(processed_data)
        
        # Create sequences
        sequences = self.data_processor.build_sequences(processed_data)
        
        # Convert to tensors
        features = torch.tensor(sequences[0], dtype=torch.float32)
        targets = torch.tensor(sequences[1], dtype=torch.float32)
        
        # Create stock_ids tensor (placeholder for now)
        batch_size, seq_len, feature_dim = features.shape
        stock_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
        
        logger.info(f"Prepared data - features: {features.shape}, targets: {targets.shape}")
        
        return features, targets, stock_ids
    
    def predict(self, features: torch.Tensor, stock_ids: torch.Tensor) -> torch.Tensor:
        """Make predictions"""
        with torch.no_grad():
            features = features.to(self.device)
            stock_ids = stock_ids.to(self.device)
            
            predictions = self.model(features, stock_ids)
            
        return predictions.cpu()
    
    def calculate_correlations(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Calculate various correlation metrics"""
        predictions_np = predictions.numpy().flatten()
        targets_np = targets.numpy().flatten()
        
        # Remove NaN values
        valid_mask = ~(np.isnan(predictions_np) | np.isnan(targets_np))
        pred_clean = predictions_np[valid_mask]
        target_clean = targets_np[valid_mask]
        
        if len(pred_clean) == 0:
            logger.warning("No valid data for correlation calculation")
            return {}
        
        # Calculate correlations
        correlations = {}
        
        # Pearson correlation
        try:
            pearson_corr = np.corrcoef(pred_clean, target_clean)[0, 1]
            correlations['pearson_correlation'] = pearson_corr
        except:
            correlations['pearson_correlation'] = np.nan
        
        # Spearman correlation (rank correlation)
        try:
            spearman_corr = np.corrcoef(np.argsort(np.argsort(pred_clean)), 
                                      np.argsort(np.argsort(target_clean)))[0, 1]
            correlations['spearman_correlation'] = spearman_corr
        except:
            correlations['spearman_correlation'] = np.nan
        
        # IC (Information Coefficient) - same as Pearson for single target
        correlations['ic'] = correlations['pearson_correlation']
        
        # Rank IC - same as Spearman
        correlations['rank_ic'] = correlations['spearman_correlation']
        
        # Additional metrics
        try:
            # MSE
            mse = np.mean((pred_clean - target_clean) ** 2)
            correlations['mse'] = mse
            
            # RMSE
            correlations['rmse'] = np.sqrt(mse)
            
            # MAE
            correlations['mae'] = np.mean(np.abs(pred_clean - target_clean))
            
            # R-squared
            ss_res = np.sum((target_clean - pred_clean) ** 2)
            ss_tot = np.sum((target_clean - np.mean(target_clean)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            correlations['r_squared'] = r_squared
            
        except:
            correlations['mse'] = np.nan
            correlations['rmse'] = np.nan
            correlations['mae'] = np.nan
            correlations['r_squared'] = np.nan
        
        logger.info(f"Calculated correlations: {correlations}")
        return correlations
    
    def run_prediction(self, test_data_dir: str, output_dir: str = None) -> Dict[str, float]:
        """Run complete out-of-sample prediction and correlation analysis"""
        if output_dir is None:
            output_dir = project_root / "outputs" / "out_of_sample_results"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Starting out-of-sample prediction analysis")
        
        # Load test data
        test_data = self.load_test_data(test_data_dir)
        
        # Prepare data
        features, targets, stock_ids = self.prepare_data(test_data)
        
        # Make predictions
        logger.info("Making predictions...")
        predictions = self.predict(features, stock_ids)
        
        # Calculate correlations
        logger.info("Calculating correlations...")
        correlations = self.calculate_correlations(predictions, targets)
        
        # Save results
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_path': str(self.model_path),
            'test_data_shape': test_data.shape,
            'predictions_shape': predictions.shape,
            'targets_shape': targets.shape,
            'correlations': correlations,
            'model_config': {
                'hidden_dim': self.config.hidden_dim,
                'num_layers': self.config.num_layers,
                'num_heads': self.config.num_heads,
                'dropout': self.config.dropout,
                'learning_rate': self.config.learning_rate,
                'batch_size': self.config.batch_size,
                'sequence_length': self.config.sequence_length
            }
        }
        
        # Save detailed results
        results_file = output_dir / "out_of_sample_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save predictions and targets for further analysis
        predictions_file = output_dir / "predictions.npy"
        targets_file = output_dir / "targets.npy"
        np.save(predictions_file, predictions.numpy())
        np.save(targets_file, targets.numpy())
        
        # Create summary report
        summary_file = output_dir / "correlation_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("Out-of-Sample Prediction Correlation Analysis\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Test Data: {test_data_dir}\n")
            f.write(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("Correlation Results:\n")
            f.write("-" * 20 + "\n")
            for metric, value in correlations.items():
                f.write(f"{metric}: {value:.6f}\n")
            f.write(f"\nData Summary:\n")
            f.write(f"Test data shape: {test_data.shape}\n")
            f.write(f"Predictions shape: {predictions.shape}\n")
            f.write(f"Targets shape: {targets.shape}\n")
        
        logger.info(f"Results saved to: {output_dir}")
        logger.info(f"Correlation summary: {correlations}")
        
        return correlations

def main():
    """Main function for out-of-sample prediction"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Out-of-sample prediction and correlation analysis")
    parser.add_argument("--model-path", type=str, required=True, 
                       help="Path to the trained model (.pth file)")
    parser.add_argument("--config-path", type=str, required=True,
                       help="Path to the model configuration (.json file)")
    parser.add_argument("--test-data-dir", type=str, 
                       default="data/test_data",
                       help="Directory containing test data files")
    parser.add_argument("--output-dir", type=str,
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda)")
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        predictor = OutOfSamplePredictor(
            model_path=args.model_path,
            config_path=args.config_path,
            device=args.device
        )
        
        # Run prediction
        correlations = predictor.run_prediction(
            test_data_dir=args.test_data_dir,
            output_dir=args.output_dir
        )
        
        print("\n" + "="*50)
        print("OUT-OF-SAMPLE PREDICTION RESULTS")
        print("="*50)
        for metric, value in correlations.items():
            print(f"{metric}: {value:.6f}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise

if __name__ == "__main__":
    main() 