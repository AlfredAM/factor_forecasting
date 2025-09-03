"""
Inference module
Responsible for model inference, real-time prediction, uncertainty quantification, etc.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import time
import logging
from collections import deque
import pickle
import os
from configs.config import config, inference_config
from src.models.models import create_model
from src.data_processing.data_processor import MultiFileDataProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UncertaintyQuantifier:
    """Uncertainty quantifier"""
    
    def __init__(self, model, config, num_samples: int = 100):
        self.model = model
        self.config = config
        self.num_samples = num_samples
    
    def monte_carlo_dropout(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Monte Carlo Dropout uncertainty quantification"""
        self.model.train()  # Enable dropout
        
        predictions = []
        for _ in range(self.num_samples):
            with torch.no_grad():
                stock_ids = torch.zeros(1, x.size(1), dtype=torch.long).to(x.device)
                pred = self.model(x, stock_ids)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        mean_pred = torch.mean(predictions, dim=0)
        std_pred = torch.std(predictions, dim=0)
        
        self.model.eval()  # Restore evaluation mode
        return mean_pred, std_pred
    
    def ensemble_uncertainty(self, models: List[nn.Module], x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ensemble model uncertainty quantification"""
        predictions = []
        
        for model in models:
            with torch.no_grad():
                stock_ids = torch.zeros(1, x.size(1), dtype=torch.long).to(x.device)
                pred = model(x, stock_ids)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        mean_pred = torch.mean(predictions, dim=0)
        std_pred = torch.std(predictions, dim=0)
        
        return mean_pred, std_pred

class RealTimePredictor:
    """Real-time predictor"""
    
    def __init__(self, model, config, scalers: Dict, max_latency_ms: float = 50.0):
        # Compatible with dict and object attribute access
        if isinstance(config, dict):
            class C(dict): __getattr__ = dict.get
            config = C(config)
        self.model = model
        self.config = config
        self.scalers = scalers
        self.max_latency_ms = max_latency_ms
        
        # Cache recent data
        self.cache = {}
        self.cache_size = inference_config.cache_size
        
        # Performance monitoring
        self.latency_history = deque(maxlen=1000)
        self.prediction_history = deque(maxlen=1000)
    
    def predict_single(self, factors: np.ndarray, sid: str) -> Dict:
        """Single prediction"""
        start_time = time.time()
        
        try:
            # Check cache
            cache_key = f"{sid}_{hash(str(factors))}"
            if cache_key in self.cache:
                result = self.cache[cache_key]
                latency = (time.time() - start_time) * 1000
                self.latency_history.append(latency)
                return result
            
            # Data preprocessing
            processed_factors = self._preprocess_factors(factors)
            
            # Model prediction
            with torch.no_grad():
                x = torch.FloatTensor(processed_factors).unsqueeze(0).to(self.config.device)
                stock_ids = torch.zeros(1, x.size(1), dtype=torch.long).to(self.config.device)
                prediction = self.model(x, stock_ids)
                # Handle dictionary output from model
                if isinstance(prediction, dict):
                    # Take the first target column as the main prediction
                    prediction = prediction[self.config.target_columns[0]]
                prediction = prediction.cpu().numpy()
            
            # Post-processing
            processed_prediction = self._postprocess_prediction(prediction)
            
            # Calculate latency
            latency = (time.time() - start_time) * 1000
            
            # Check latency requirements
            if latency > self.max_latency_ms:
                logger.warning(f"Prediction latency {latency:.2f}ms exceeds threshold {self.max_latency_ms}ms")
            
            # Cache result
            result = {
                'prediction': processed_prediction,
                'latency_ms': latency,
                'timestamp': time.time(),
                'confidence': self._calculate_confidence(processed_prediction)
            }
            
            self.cache[cache_key] = result
            self.latency_history.append(latency)
            self.prediction_history.append(processed_prediction)
            
            # Clear cache
            if len(self.cache) > self.cache_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return self._fallback_prediction()
    
    def predict_batch(self, factors_batch: np.ndarray, sids: List[str]) -> List[Dict]:
        """Batch prediction"""
        start_time = time.time()
        
        try:
            # Data preprocessing
            processed_factors = self._preprocess_factors_batch(factors_batch)
            
            # Model prediction
            with torch.no_grad():
                x = torch.FloatTensor(processed_factors).to(self.config.device)
                stock_ids = torch.zeros(x.size(0), x.size(1), dtype=torch.long).to(self.config.device)
                predictions = self.model(x, stock_ids)
                # Handle dictionary output from model
                if isinstance(predictions, dict):
                    # Take the first target column as the main prediction
                    predictions = predictions[self.config.target_columns[0]]
                predictions = predictions.cpu().numpy()
            
            # Post-processing
            results = []
            for i, (prediction, sid) in enumerate(zip(predictions, sids)):
                processed_prediction = self._postprocess_prediction(prediction)
                
                result = {
                    'prediction': processed_prediction,
                    'sid': sid,
                    'latency_ms': (time.time() - start_time) * 1000 / len(sids),
                    'timestamp': time.time(),
                    'confidence': self._calculate_confidence(processed_prediction)
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            return [self._fallback_prediction() for _ in sids]
    
    def _preprocess_factors(self, factors: np.ndarray) -> np.ndarray:
        """Preprocess factor data"""
        # Standardization
        if 'factor_scaler' in self.scalers:
            factors = self.scalers['factor_scaler'].transform(factors.reshape(1, -1))
        
        return factors.reshape(1, -1)
    
    def _preprocess_factors_batch(self, factors_batch: np.ndarray) -> np.ndarray:
        """Batch preprocess factor data"""
        # Standardization
        if 'factor_scaler' in self.scalers:
            factors_batch = self.scalers['factor_scaler'].transform(factors_batch)
        
        return factors_batch
    
    def _postprocess_prediction(self, prediction: np.ndarray) -> np.ndarray:
        """Post-process prediction results"""
        # Inverse standardization
        if 'target_scaler' in self.scalers:
            prediction = self.scalers['target_scaler'].inverse_transform(prediction.reshape(1, -1))
        
        return prediction.flatten()
    
    def _calculate_confidence(self, prediction: np.ndarray) -> float:
        """Calculate prediction confidence"""
        # Calculate confidence based on prediction variance
        variance = np.var(prediction)
        confidence = 1.0 / (1.0 + variance)
        return min(confidence, 1.0)
    
    def _fallback_prediction(self) -> Dict:
        """Fallback prediction"""
        return {
            'prediction': np.zeros(len(self.config.target_columns)),
            'latency_ms': 0.0,
            'timestamp': time.time(),
            'confidence': 0.0,
            'fallback': True
        }
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.latency_history:
            return {}
        
        latencies = list(self.latency_history)
        return {
            'avg_latency_ms': np.mean(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'max_latency_ms': np.max(latencies),
            'min_latency_ms': np.min(latencies),
            'total_predictions': len(latencies)
        }

class EnsemblePredictor:
    """Ensemble predictor"""
    
    def __init__(self, models: List[nn.Module], config, scalers: Dict):
        # Compatible with dict and object attribute access
        if isinstance(config, dict):
            class C(dict): __getattr__ = dict.get
            config = C(config)
        self.models = models
        self.config = config
        self.scalers = scalers
        self.uncertainty_quantifier = UncertaintyQuantifier(models[0], config)
    
    def predict(self, factors: np.ndarray, sid: str) -> Dict:
        """Ensemble prediction"""
        start_time = time.time()
        
        try:
            # Data preprocessing
            processed_factors = self._preprocess_factors(factors)
            
            # Ensemble prediction
            predictions = []
            for model in self.models:
                with torch.no_grad():
                    x = torch.FloatTensor(processed_factors).unsqueeze(0).to(self.config.device)
                    stock_ids = torch.zeros(1, x.size(1), dtype=torch.long).to(self.config.device)
                    pred = model(x, stock_ids)
                    # Handle dictionary output from model
                    if isinstance(pred, dict):
                        # Take the first target column as the main prediction
                        pred = pred[self.config.target_columns[0]]
                    pred = pred.cpu().numpy()
                    predictions.append(pred)
            
            # Calculate ensemble results
            predictions = np.array(predictions)
            mean_prediction = np.mean(predictions, axis=0)
            std_prediction = np.std(predictions, axis=0)
            
            # Post-processing
            processed_prediction = self._postprocess_prediction(mean_prediction)
            uncertainty = self._postprocess_prediction(std_prediction)
            
            # Calculate confidence
            confidence = self._calculate_ensemble_confidence(mean_prediction, std_prediction)
            
            return {
                'prediction': processed_prediction,
                'uncertainty': uncertainty,
                'confidence': confidence,
                'latency_ms': (time.time() - start_time) * 1000,
                'timestamp': time.time(),
                'ensemble_size': len(self.models)
            }
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            return self._fallback_prediction()
    
    def _preprocess_factors(self, factors: np.ndarray) -> np.ndarray:
        """Preprocess factor data"""
        if 'factor_scaler' in self.scalers:
            factors = self.scalers['factor_scaler'].transform(factors.reshape(1, -1))
        return factors.reshape(1, -1)
    
    def _postprocess_prediction(self, prediction: np.ndarray) -> np.ndarray:
        """Post-process prediction results"""
        if 'target_scaler' in self.scalers:
            prediction = self.scalers['target_scaler'].inverse_transform(prediction.reshape(1, -1))
        return prediction.flatten()
    
    def _calculate_ensemble_confidence(self, mean_pred: np.ndarray, std_pred: np.ndarray) -> float:
        """Calculate ensemble confidence"""
        # Calculate confidence based on mean and standard deviation
        cv = std_pred / (np.abs(mean_pred) + 1e-8)  # Coefficient of variation
        confidence = 1.0 / (1.0 + np.mean(cv))
        return min(confidence, 1.0)
    
    def _fallback_prediction(self) -> Dict:
        """Fallback prediction"""
        return {
            'prediction': np.zeros(len(self.config.target_columns)),
            'uncertainty': np.zeros(len(self.config.target_columns)),
            'confidence': 0.0,
            'latency_ms': 0.0,
            'timestamp': time.time(),
            'fallback': True
        }

class ModelInference:
    """Model inference main class"""
    
    def __init__(self, model_path: str, config, scalers: Dict = None):
        self.config = config
        # Handle both dict and object config formats
        if isinstance(config, dict):
            self.device = torch.device(getattr(config, 'device', 'cpu'))
        else:
            self.device = torch.device(config.device)
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Load scalers
        self.scalers = scalers or {}
        
        # Create predictor
        if inference_config.use_ensemble:
            self.predictor = EnsemblePredictor([self.model], config, self.scalers)
        else:
            self.predictor = RealTimePredictor(self.model, config, self.scalers, inference_config.max_latency_ms)
        
        # Fallback model
        self.fallback_model = None
        if os.path.exists(inference_config.fallback_model):
            self.fallback_model = self._load_model(inference_config.fallback_model)
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Convert config to dict if it's a ModelConfig object
        model_config = checkpoint['config']
        if not isinstance(model_config, dict):
            model_config = model_config.__dict__
        
        # Create model
        model = create_model(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        logger.info(f"Model loaded from {model_path}")
        return model
    
    def predict(self, factors: np.ndarray, sid: str) -> Dict:
        """Prediction interface"""
        # Check confidence threshold
        result = self.predictor.predict(factors, sid)
        
        if result.get('confidence', 0) < inference_config.confidence_threshold:
            logger.warning(f"Prediction confidence {result['confidence']:.3f} below threshold {inference_config.confidence_threshold}")
            
            # Use fallback model
            if self.fallback_model:
                result = self._fallback_predict(factors, sid)
        
        return result
    
    def _fallback_predict(self, factors: np.ndarray, sid: str) -> Dict:
        """Fallback prediction"""
        logger.info("Using fallback model for prediction")
        
        fallback_predictor = RealTimePredictor(self.fallback_model, self.config, self.scalers)
        return fallback_predictor.predict_single(factors, sid)
    
    def batch_predict(self, factors_batch: np.ndarray, sids: List[str]) -> List[Dict]:
        """Batch prediction interface"""
        if hasattr(self.predictor, 'predict_batch'):
            return self.predictor.predict_batch(factors_batch, sids)
        else:
            return [self.predict(factors, sid) for factors, sid in zip(factors_batch, sids)]
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if hasattr(self.predictor, 'get_performance_stats'):
            return self.predictor.get_performance_stats()
        return {}

def create_inference_engine(model_path: str, config, scalers: Dict = None) -> ModelInference:
    """Convenient function to create an inference engine"""
    # Convert config to dict if it's a ModelConfig object
    if not isinstance(config, dict):
        config = config.__dict__
    return ModelInference(model_path, config, scalers) 