"""
Configuration Loader for Model Configurations
Provides utilities for loading, validating, and managing model configurations
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Model configuration dataclass"""
    name: str
    type: str
    hidden_size: int
    num_layers: int
    num_heads: int
    dropout: float
    activation: str
    d_ff: int
    max_seq_len: int
    num_factors: int
    num_targets: int
    embedding_dim: int
    use_flash_attention: bool = True
    attention_dropout: float = 0.1
    layer_norm_eps: float = 1e-6

@dataclass
class TrainingConfig:
    """Training configuration dataclass"""
    batch_size: int
    learning_rate: float
    weight_decay: float
    num_epochs: int
    gradient_clip: float
    scheduler_type: str
    warmup_steps: int
    min_lr: float
    early_stopping_patience: int
    early_stopping_min_delta: float
    restore_best_weights: bool
    use_mixed_precision: bool
    gradient_accumulation_steps: int

@dataclass
class DataConfig:
    """Data configuration dataclass"""
    sequence_length: int
    prediction_horizon: int
    min_sequence_length: int
    target_columns: List[str]
    factor_columns: Optional[List[str]]
    stock_id_column: str
    weight_column: str
    train_ratio: float
    val_ratio: float
    test_ratio: float

@dataclass
class LossConfig:
    """Loss function configuration dataclass"""
    type: str
    correlation_weight: float
    mse_weight: float
    rank_correlation_weight: float
    target_correlations: List[float]
    quantile_alpha: float

@dataclass
class OptimizationConfig:
    """Optimization configuration dataclass"""
    optimizer_type: str
    beta1: float
    beta2: float
    eps: float
    label_smoothing: float
    dropout: float

@dataclass
class OutputConfig:
    """Output configuration dataclass"""
    model_save_dir: str
    best_model_name: str
    checkpoint_interval: int
    log_dir: str
    use_wandb: bool
    wandb_project: str
    wandb_run_name: str

@dataclass
class HardwareConfig:
    """Hardware configuration dataclass"""
    device: str
    num_workers: int
    pin_memory: bool

@dataclass
class CompleteConfig:
    """Complete configuration dataclass"""
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    loss: LossConfig
    optimization: OptimizationConfig
    output: OutputConfig
    hardware: HardwareConfig
    training_mode: str
    rolling_window_years: int
    min_train_years: int
    prediction_years: List[int]

class ConfigLoader:
    """Configuration loader class"""
    
    def __init__(self, config_dir: str = "configs/model_configs"):
        self.config_dir = Path(config_dir)
        self.config_cache = {}
    
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file
        
        Args:
            config_name: Name of the configuration file (with or without .yaml extension)
            
        Returns:
            Configuration dictionary
        """
        if not config_name.endswith('.yaml'):
            config_name += '.yaml'
        
        config_path = self.config_dir / config_name
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Check cache first
        if config_name in self.config_cache:
            return self.config_cache[config_name]
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Cache the configuration
            self.config_cache[config_name] = config
            
            logger.info(f"Loaded configuration: {config_name}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading configuration {config_name}: {e}")
            raise
    
    def load_complete_config(self, config_name: str) -> CompleteConfig:
        """
        Load and parse configuration into dataclass structure
        
        Args:
            config_name: Name of the configuration file
            
        Returns:
            CompleteConfig dataclass instance
        """
        config_dict = self.load_config(config_name)
        
        # Parse model configuration
        model_config = ModelConfig(**config_dict['model'])
        
        # Parse training configuration
        training_config = TrainingConfig(**config_dict['training'])
        
        # Parse data configuration
        data_config = DataConfig(**config_dict['data'])
        
        # Parse loss configuration
        loss_config = LossConfig(**config_dict['loss'])
        
        # Parse optimization configuration
        optimization_config = OptimizationConfig(**config_dict['optimization'])
        
        # Parse output configuration
        output_config = OutputConfig(**config_dict['output'])
        
        # Parse hardware configuration
        hardware_config = HardwareConfig(**config_dict['hardware'])
        
        # Create complete configuration
        complete_config = CompleteConfig(
            model=model_config,
            training=training_config,
            data=data_config,
            loss=loss_config,
            optimization=optimization_config,
            output=output_config,
            hardware=hardware_config,
            training_mode=config_dict.get('training_mode', 'rolling_window'),
            rolling_window_years=config_dict.get('rolling_window_years', 1),
            min_train_years=config_dict.get('min_train_years', 1),
            prediction_years=config_dict.get('prediction_years', [2019, 2020, 2021, 2022])
        )
        
        return complete_config
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration parameters
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        required_sections = ['model', 'training', 'data', 'loss', 'optimization', 'output', 'hardware']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate model configuration
        model = config['model']
        if model['hidden_size'] <= 0:
            raise ValueError("hidden_size must be positive")
        if model['num_layers'] <= 0:
            raise ValueError("num_layers must be positive")
        if model['num_heads'] <= 0:
            raise ValueError("num_heads must be positive")
        if not 0 <= model['dropout'] <= 1:
            raise ValueError("dropout must be between 0 and 1")
        
        # Validate training configuration
        training = config['training']
        if training['batch_size'] <= 0:
            raise ValueError("batch_size must be positive")
        if training['learning_rate'] <= 0:
            raise ValueError("learning_rate must be positive")
        if training['num_epochs'] <= 0:
            raise ValueError("num_epochs must be positive")
        
        # Validate data configuration
        data = config['data']
        if data['sequence_length'] <= 0:
            raise ValueError("sequence_length must be positive")
        if not 0 < data['train_ratio'] < 1:
            raise ValueError("train_ratio must be between 0 and 1")
        if not 0 < data['val_ratio'] < 1:
            raise ValueError("val_ratio must be between 0 and 1")
        if not 0 < data['test_ratio'] < 1:
            raise ValueError("test_ratio must be between 0 and 1")
        
        # Check that ratios sum to approximately 1
        total_ratio = data['train_ratio'] + data['val_ratio'] + data['test_ratio']
        if abs(total_ratio - 1.0) > 0.01:
            raise ValueError(f"Data split ratios must sum to 1.0, got {total_ratio}")
        
        logger.info("Configuration validation passed")
        return True
    
    def list_available_configs(self) -> List[str]:
        """
        List all available configuration files
        
        Returns:
            List of configuration file names
        """
        config_files = []
        for file_path in self.config_dir.rglob("*.yaml"):
            # Get relative path from config_dir
            relative_path = file_path.relative_to(self.config_dir)
            config_files.append(str(relative_path))
        
        return sorted(config_files)
    
    def get_config_info(self, config_name: str) -> Dict[str, Any]:
        """
        Get information about a configuration
        
        Args:
            config_name: Name of the configuration file
            
        Returns:
            Dictionary with configuration information
        """
        config = self.load_config(config_name)
        
        info = {
            'name': config_name,
            'model_type': config['model']['type'],
            'model_size': {
                'hidden_size': config['model']['hidden_size'],
                'num_layers': config['model']['num_layers'],
                'num_heads': config['model']['num_heads']
            },
            'training': {
                'batch_size': config['training']['batch_size'],
                'learning_rate': config['training']['learning_rate'],
                'num_epochs': config['training']['num_epochs']
            },
            'data': {
                'sequence_length': config['data']['sequence_length'],
                'target_columns': config['data']['target_columns']
            }
        }
        
        return info
    
    def merge_configs(self, base_config: str, override_config: str) -> Dict[str, Any]:
        """
        Merge two configurations, with override_config taking precedence
        
        Args:
            base_config: Base configuration file name
            override_config: Override configuration file name
            
        Returns:
            Merged configuration dictionary
        """
        base = self.load_config(base_config)
        override = self.load_config(override_config)
        
        def deep_merge(base_dict, override_dict):
            """Recursively merge dictionaries"""
            result = base_dict.copy()
            for key, value in override_dict.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        merged_config = deep_merge(base, override)
        return merged_config
    
    def save_config(self, config: Dict[str, Any], config_name: str) -> None:
        """
        Save configuration to YAML file
        
        Args:
            config: Configuration dictionary
            config_name: Name for the configuration file
        """
        if not config_name.endswith('.yaml'):
            config_name += '.yaml'
        
        config_path = self.config_dir / config_name
        
        # Create directory if it doesn't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            logger.info(f"Saved configuration: {config_name}")
            
        except Exception as e:
            logger.error(f"Error saving configuration {config_name}: {e}")
            raise

# Convenience functions
def load_config(config_name: str) -> Dict[str, Any]:
    """Load configuration from file"""
    loader = ConfigLoader()
    return loader.load_config(config_name)

def load_complete_config(config_name: str) -> CompleteConfig:
    """Load complete configuration as dataclass"""
    loader = ConfigLoader()
    return loader.load_complete_config(config_name)

def list_configs() -> List[str]:
    """List all available configurations"""
    loader = ConfigLoader()
    return loader.list_available_configs()

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration"""
    loader = ConfigLoader()
    return loader.validate_config(config)

if __name__ == "__main__":
    # Example usage
    loader = ConfigLoader()
    
    # List available configurations
    configs = loader.list_available_configs()
    print("Available configurations:")
    for config in configs:
        print(f"  - {config}")
    
    # Load and validate a configuration
    try:
        config = loader.load_config("transformer_base")
        loader.validate_config(config)
        print("\nConfiguration loaded and validated successfully!")
        
        # Get configuration info
        info = loader.get_config_info("transformer_base")
        print(f"\nConfiguration info: {info}")
        
    except Exception as e:
        print(f"Error: {e}") 