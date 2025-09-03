"""
Model factory for creating, loading, and saving factor forecasting models
"""
import torch
import torch.nn as nn
import os
import logging
from typing import Dict, Optional

from .transformer import FactorForecastingModel

logger = logging.getLogger(__name__)


def create_model(config: Dict) -> FactorForecastingModel:
    """
    Create a new factor forecasting model.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized FactorForecastingModel
    """
    logger.info("Creating new factor forecasting model...")
    
    # Validate configuration
    required_keys = ['num_factors', 'num_stocks', 'd_model', 'num_heads', 'num_layers']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    # Create model
    model = FactorForecastingModel(config)
    
    # Log model information
    model_info = model.get_model_info()
    logger.info(f"Model created successfully:")
    logger.info(f"  Total parameters: {model_info['total_parameters']:,}")
    logger.info(f"  Model size: {model_info['model_size_mb']:.2f} MB")
    logger.info(f"  Layers: {model_info['num_layers']}")
    logger.info(f"  Model dimension: {model_info['d_model']}")
    logger.info(f"  Attention heads: {model_info['num_heads']}")
    logger.info(f"  Targets: {model_info['target_columns']}")
    
    return model


def load_model(model_path: str, config: Dict = None, device: str = 'cpu') -> FactorForecastingModel:
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to the saved model file
        config: Configuration dictionary (optional, will use saved config if not provided)
        device: Device to load the model on
        
    Returns:
        Loaded FactorForecastingModel
    """
    logger.info(f"Loading model from {model_path}...")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load saved data
    try:
        saved_data = torch.load(model_path, map_location=device)
        
        # Handle both old format (just state_dict) and new format (dict with config)
        if isinstance(saved_data, dict) and 'state_dict' in saved_data:
            state_dict = saved_data['state_dict']
            saved_config = saved_data.get('config', config)
        else:
            state_dict = saved_data
            saved_config = config
        
        if saved_config is None:
            raise ValueError("No configuration provided and no saved configuration found")
        
        # Create model
        model = create_model(saved_config)
        
        # Load state dict
        model.load_state_dict(state_dict)
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
    
    return model


def save_model(model: FactorForecastingModel, model_path: str):
    """
    Save a trained model to disk.
    
    Args:
        model: FactorForecastingModel to save
        model_path: Path where to save the model
    """
    logger.info(f"Saving model to {model_path}...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save model
    try:
        # Save complete model state including configuration
        save_dict = {
            'state_dict': model.state_dict(),
            'config': model.config,
            'model_info': model.get_model_info()
        }
        torch.save(save_dict, model_path)
        logger.info("Model saved successfully")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise


def get_model_summary(model: FactorForecastingModel) -> Dict:
    """
    Get a summary of model architecture and parameters.
    
    Args:
        model: FactorForecastingModel
        
    Returns:
        Dictionary containing model summary
    """
    model_info = model.get_model_info()
    
    # Count parameters by layer type
    param_counts = {}
    for name, param in model.named_parameters():
        layer_type = name.split('.')[0]
        if layer_type not in param_counts:
            param_counts[layer_type] = 0
        param_counts[layer_type] += param.numel()
    
    summary = {
        'model_info': model_info,
        'parameter_distribution': param_counts,
        'total_trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'total_frozen': sum(p.numel() for p in model.parameters() if not p.requires_grad)
    }
    
    return summary


def print_model_summary(model: FactorForecastingModel):
    """
    Print a formatted summary of the model.
    
    Args:
        model: FactorForecastingModel
    """
    summary = get_model_summary(model)
    
    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    
    # Model info
    model_info = summary['model_info']
    print(f"Total Parameters: {model_info['total_parameters']:,}")
    print(f"Trainable Parameters: {summary['total_trainable']:,}")
    print(f"Frozen Parameters: {summary['total_frozen']:,}")
    print(f"Model Size: {model_info['model_size_mb']:.2f} MB")
    print()
    
    # Architecture
    print("Architecture:")
    print(f"  Layers: {model_info['num_layers']}")
    print(f"  Model Dimension: {model_info['d_model']}")
    print(f"  Attention Heads: {model_info['num_heads']}")
    print(f"  Targets: {', '.join(model_info['target_columns'])}")
    print()
    
    # Parameter distribution
    print("Parameter Distribution:")
    for layer_type, count in summary['parameter_distribution'].items():
        percentage = (count / model_info['total_parameters']) * 100
        print(f"  {layer_type}: {count:,} ({percentage:.1f}%)")
    
    print("=" * 60) 