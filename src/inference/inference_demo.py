#!/usr/bin/env python3
"""
Inference demo script
Demonstrates how to use the trained model for real-time prediction
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import time
import logging
from typing import Dict, List
import warnings
# Only suppress specific warnings that are known to be harmless
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, message='.*NumPy.*')

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import config, inference_config
from src.inference.inference import create_inference_engine
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FactorForecastingDemo:
    """Factor prediction demo class"""
    
    def __init__(self, model_path: str, scalers_path: str = None):
        self.model_path = model_path
        self.scalers_path = scalers_path
        
        # Load model and scalers
        self.scalers = self._load_scalers()
        self.inference_engine = create_inference_engine(model_path, config, self.scalers)
        
        logger.info("Inference engine initialized")
    
    def _load_scalers(self) -> Dict:
        """Load scalers"""
        if self.scalers_path and os.path.exists(self.scalers_path):
            with open(self.scalers_path, 'rb') as f:
                return pickle.load(f)
        else:
            logger.warning("Scaler file not found, using default values")
            return {}
    
    def predict_single_stock(self, factors: np.ndarray, sid: str) -> Dict:
        """Single stock prediction"""
        logger.info(f"Predicting stock {sid}")
        
        # Record start time
        start_time = time.time()
        
        # Perform prediction
        result = self.inference_engine.predict(factors, sid)
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Print results
        logger.info(f"Prediction completed, time: {total_time*1000:.2f}ms")
        logger.info(f"Prediction result: {result['prediction']}")
        logger.info(f"Confidence: {result.get('confidence', 0):.3f}")
        
        if 'uncertainty' in result:
            logger.info(f"Uncertainty: {result['uncertainty']}")
        
        return result
    
    def predict_batch_stocks(self, factors_batch: np.ndarray, sids: List[str]) -> List[Dict]:
        """Batch stock prediction"""
        logger.info(f"Batch predicting {len(sids)} stocks")
        
        # Record start time
        start_time = time.time()
        
        # Perform batch prediction
        results = self.inference_engine.batch_predict(factors_batch, sids)
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Print results
        logger.info(f"Batch prediction completed, total time: {total_time*1000:.2f}ms")
        logger.info(f"Average time per stock: {total_time*1000/len(sids):.2f}ms")
        
        for i, (result, sid) in enumerate(zip(results, sids)):
            logger.info(f"Stock {sid}: Prediction={result['prediction']}, Confidence={result.get('confidence', 0):.3f}")
        
        return results
    
    def simulate_real_time_prediction(self, data_path: str, num_samples: int = 10):
        """Simulate real-time prediction"""
        logger.info("Starting real-time prediction simulation...")
        
        # Read data
        df = pd.read_csv(data_path)
        
        # Filter out limit up/down data
        df = df[df[config.luld_column] != 1]
        
        # Randomly select samples
        sample_indices = np.random.choice(len(df), num_samples, replace=False)
        
        # Extract factors and targets
        factor_cols = config.factor_columns
        target_cols = config.target_columns
        
        results = []
        for i, idx in enumerate(sample_indices):
            row = df.iloc[idx]
            sid = row[config.sid_column]
            factors = row[factor_cols].values
            true_targets = row[target_cols].values
            
            logger.info(f"\nSample {i+1}/{num_samples}: Stock {sid}")
            logger.info(f"True targets: {true_targets}")
            
            # Perform prediction
            result = self.predict_single_stock(factors, str(sid))
            
            # Calculate error
            prediction = result['prediction']
            mse = np.mean((prediction - true_targets) ** 2)
            mae = np.mean(np.abs(prediction - true_targets))
            
            logger.info(f"Prediction error - MSE: {mse:.6f}, MAE: {mae:.6f}")
            
            results.append({
                'sid': sid,
                'true_targets': true_targets,
                'prediction': prediction,
                'mse': mse,
                'mae': mae,
                'confidence': result.get('confidence', 0),
                'latency_ms': result.get('latency_ms', 0)
            })
        
        # Print statistics
        self._print_statistics(results)
        
        return results
    
    def _print_statistics(self, results: List[Dict]):
        """Print statistics results"""
        logger.info("\n" + "="*50)
        logger.info("Prediction Statistics")
        logger.info("="*50)
        
        mses = [r['mse'] for r in results]
        maes = [r['mae'] for r in results]
        confidences = [r['confidence'] for r in results]
        latencies = [r['latency_ms'] for r in results]
        
        logger.info(f"Average MSE: {np.mean(mses):.6f} ± {np.std(mses):.6f}")
        logger.info(f"Average MAE: {np.mean(maes):.6f} ± {np.std(maes):.6f}")
        logger.info(f"Average Confidence: {np.mean(confidences):.3f} ± {np.std(confidences):.3f}")
        logger.info(f"Average Latency: {np.mean(latencies):.2f}ms ± {np.std(latencies):.2f}ms")
        logger.info(f"Max Latency: {np.max(latencies):.2f}ms")
        logger.info(f"Min Latency: {np.min(latencies):.2f}ms")
        
        # Performance statistics
        if hasattr(self.inference_engine.predictor, 'get_performance_stats'):
            perf_stats = self.inference_engine.predictor.get_performance_stats()
            if perf_stats:
                logger.info("\nPerformance Statistics:")
                for key, value in perf_stats.items():
                    logger.info(f"  {key}: {value:.2f}")
    
    def test_emergency_handling(self):
        """Test emergency handling"""
        logger.info("\nTesting emergency handling...")
        
        # Create invalid data
        invalid_factors = np.full((100,), np.nan)  # Data with all NaNs
        
        try:
            result = self.predict_single_stock(invalid_factors, "test_sid")
            if result.get('fallback', False):
                logger.info("Emergency handling activated successfully")
            else:
                logger.info("Model handled the invalid data normally")
        except Exception as e:
            logger.info(f"Emergency handling: {e}")
    
    def benchmark_performance(self, num_iterations: int = 100):
        """Benchmark performance"""
        logger.info(f"\nStarting performance benchmark ({num_iterations} iterations)...")
        
        # Create random test data
        test_factors = np.random.randn(num_iterations, len(config.factor_columns))
        test_sids = [f"test_{i}" for i in range(num_iterations)]
        
        # Batch prediction
        start_time = time.time()
        results = self.inference_engine.batch_predict(test_factors, test_sids)
        total_time = time.time() - start_time
        
        # Calculate performance metrics
        avg_time_per_prediction = total_time / num_iterations * 1000  # ms
        predictions_per_second = num_iterations / total_time
        
        logger.info(f"Performance benchmark results:")
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info(f"  Average prediction time: {avg_time_per_prediction:.2f}ms")
        logger.info(f"  Predictions per second: {predictions_per_second:.1f}")
        
        # Check latency requirements
        if avg_time_per_prediction > inference_config.max_latency_ms:
            logger.warning(f"Average latency {avg_time_per_prediction:.2f}ms exceeds threshold {inference_config.max_latency_ms}ms")
        else:
            logger.info("Latency requirements met")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Factor prediction inference demo')
    parser.add_argument('--model_path', type=str, required=True, help='Model file path')
    parser.add_argument('--scalers_path', type=str, help='Scaler file path')
    parser.add_argument('--data_path', type=str, default='head50k.csv', help='Test data path')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of test samples')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    
    args = parser.parse_args()
    
    try:
        # Create demo instance
        demo = FactorForecastingDemo(args.model_path, args.scalers_path)
        
        # Simulate real-time prediction
        demo.simulate_real_time_prediction(args.data_path, args.num_samples)
        
        # Test emergency handling
        demo.test_emergency_handling()
        
        # Benchmark performance
        if args.benchmark:
            demo.benchmark_performance()
        
        logger.info("Inference demo completed!")
        
    except Exception as e:
        logger.error(f"Inference demo failed: {e}")
        raise

if __name__ == "__main__":
    main() 