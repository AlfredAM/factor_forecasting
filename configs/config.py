"""
Factor Forecasting Model Configuration File
Contains model architecture, training parameters, data processing configurations
"""
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

@dataclass
class ModelConfig:
    """Model configuration class for Factor Forecasting System"""
    
    # ============================================================================
    # Training Mode Configuration
    # ============================================================================
    training_mode: str = "quantitative"  # "quantitative", "rolling_window" or "multi_file"
    
    # ============================================================================
    # Data Configuration - Quantitative Finance Optimized
    # ============================================================================
    # Data directory for multi-file streaming training
    data_dir: str = "./data"
    
    # Fixed time window splits (proper for financial time series)
    train_start_date: str = "2018-01-01"
    train_end_date: str = "2021-12-31"
    val_start_date: str = "2022-01-01"
    val_end_date: str = "2022-06-30"
    test_start_date: str = "2022-07-01"
    test_end_date: str = "2022-12-31"
    
    # Legacy date configuration (for backward compatibility)
    start_date: str = "2018-01-01"
    end_date: str = "2022-12-31"
    
    # Rolling window configuration (legacy)
    rolling_window_years: int = 1  # Number of years to train before predicting next year
    min_train_years: int = 1  # Minimum years required for training
    prediction_years: List[int] = field(default_factory=lambda: [2019, 2020, 2021, 2022])  # Years to predict
    
    # Column configuration
    factor_columns: List[str] = field(default_factory=lambda: [str(i) for i in range(100)])  # 0-99
    target_columns: List[str] = field(default_factory=lambda: ["intra30m", "nextT1d", "ema1d"])
    stock_id_column: str = "sid"
    limit_up_down_column: str = "luld"
    weight_column: str = "ADV50"
    
    # Sequence configuration
    sequence_length: int = 20  # Historical window length (20 trading days)
    prediction_horizon: int = 1  # Prediction steps
    min_sequence_length: int = 5  # Minimum sequence length
    
    # Quantitative finance specific settings
    min_stock_history_days: int = 252  # Minimum 1 year of data per stock
    max_missing_ratio: float = 0.1  # Maximum 10% missing values allowed
    remove_limit_up_down: bool = True  # Remove limit up/down days to avoid bias
    remove_suspended: bool = True  # Remove suspended trading days
    use_fixed_time_windows: bool = True  # Use fixed time windows instead of ratios
    
    # ============================================================================
    # Model Architecture Configuration
    # ============================================================================
    model_type: str = 'transformer'  # 'transformer', 'tcn', 'tcn_attention', 'lstm'
    hidden_size: int = 256
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    activation: str = "gelu"
    
    # TCN-specific configuration
    kernel_size: int = 3  # Kernel size for TCN convolutions
    tcn_channels: List[int] = field(default_factory=lambda: [256, 256, 256, 256, 256, 256])  # Channel sizes for TCN layers
    
    # LSTM-specific configuration
    lstm_hidden_size: int = 256
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.1
    lstm_bidirectional: bool = True
    
    # ============================================================================
    # Training Configuration
    # ============================================================================
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_epochs: int = 100
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    test_split: float = 0.1
    
    # Data split ratios for multi-file mode
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Optimizer configuration
    optimizer_type: str = "adamw"  # "adam", "adamw", "sgd", "rmsprop"
    scheduler_type: str = "cosine"  # "cosine", "step", "exponential", "plateau"
    warmup_steps: int = 1000
    
    # ============================================================================
    # Loss Function Configuration - Quantitative Finance Optimized
    # ============================================================================
    loss_function_type: str = "quantitative_correlation"  # quantitative_correlation, correlation_loss, mse, huber
    quantile_alpha: float = 0.5  # For quantile regression
    
    # Quantitative correlation loss configuration
    correlation_weight: float = 1.0  # Weight for correlation loss
    mse_weight: float = 0.1  # Weight for MSE loss
    rank_correlation_weight: float = 0.2  # Weight for rank correlation loss
    risk_penalty_weight: float = 0.1  # Weight for risk penalty
    target_correlations: List[float] = field(default_factory=lambda: [0.08, 0.05, 0.03])  # Target correlations optimized for finance
    
    # Risk management configuration
    max_leverage: float = 2.0  # Maximum portfolio leverage
    transaction_cost: float = 0.001  # Transaction cost rate
    use_adaptive_loss: bool = True  # Use adaptive loss based on market regime
    volatility_window: int = 20  # Window for volatility regime detection
    regime_sensitivity: float = 0.1  # Sensitivity to regime changes
    
    # ============================================================================
    # Regularization Configuration
    # ============================================================================
    label_smoothing: float = 0.0
    gradient_clip: float = 1.0
    
    # ============================================================================
    # Data Augmentation Configuration
    # ============================================================================
    use_data_augmentation: bool = True
    noise_std: float = 0.01
    mask_probability: float = 0.1
    
    # ============================================================================
    # Model Saving Configuration
    # ============================================================================
    model_save_dir: str = "./outputs/models"
    best_model_name: str = "best_model.pth"
    checkpoint_interval: int = 10
    
    # ============================================================================
    # Logging Configuration
    # ============================================================================
    log_dir: str = "./outputs/logs"
    use_wandb: bool = False
    wandb_project: str = "factor_forecasting"
    wandb_run_name: Optional[str] = None
    
    # ============================================================================
    # Hardware Configuration
    # ============================================================================
    device: str = "cpu"  # "cpu", "cuda", "auto"
    num_workers: int = 2
    pin_memory: bool = True
    
    # ============================================================================
    # Server-specific Configuration
    # ============================================================================
    server_mode: bool = False
    conda_env: str = "factor_forecast"
    output_dir: str = "./outputs"  # Unified output root directory
    
    # ============================================================================
    # Performance Configuration
    # ============================================================================
    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # ============================================================================
    # Validation Configuration
    # ============================================================================
    validation_interval: int = 1
    save_best_only: bool = True
    monitor_metric: str = "val_correlation"
    monitor_mode: str = "max"
    
    def __post_init__(self):
        """Post-initialization processing"""
        # Set default factor columns if None
        if self.factor_columns is None:
            self.factor_columns = [str(i) for i in range(100)]  # 0-99
        
        # Auto-switch local/server paths
        server_env = os.environ.get('SERVER_MODE', None)
        if server_env is not None:
            self.server_mode = server_env.lower() in ['1', 'true', 'yes']
        
        # Set paths based on environment
        if self.server_mode:
            self.data_dir = "/nas/feature_v2_10s"
            self.model_save_dir = "/nas/outputs/models"
            self.log_dir = "/nas/outputs/logs"
            self.output_dir = "/nas/outputs"
        else:
            self.data_dir = "./data"
            self.model_save_dir = "./outputs/models"
            self.log_dir = "./outputs/logs"
            self.output_dir = "./outputs"
        
        # Auto-detect device
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create necessary directories only if not in test mode
        if not os.environ.get('TESTING'):
            os.makedirs(self.model_save_dir, exist_ok=True)
            os.makedirs(self.log_dir, exist_ok=True)
            os.makedirs(self.output_dir, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'training_mode': self.training_mode,
            'data_dir': self.data_dir,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'rolling_window_years': self.rolling_window_years,
            'min_train_years': self.min_train_years,
            'prediction_years': self.prediction_years,
            'factor_columns': self.factor_columns,
            'target_columns': self.target_columns,
            'stock_id_column': self.stock_id_column,
            'limit_up_down_column': self.limit_up_down_column,
            'weight_column': self.weight_column,
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'min_sequence_length': self.min_sequence_length,
            'model_type': self.model_type,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'dropout': self.dropout,
            'activation': self.activation,
            'kernel_size': self.kernel_size,
            'tcn_channels': self.tcn_channels,
            'lstm_hidden_size': self.lstm_hidden_size,
            'lstm_num_layers': self.lstm_num_layers,
            'lstm_dropout': self.lstm_dropout,
            'lstm_bidirectional': self.lstm_bidirectional,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'num_epochs': self.num_epochs,
            'early_stopping_patience': self.early_stopping_patience,
            'validation_split': self.validation_split,
            'test_split': self.test_split,
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'optimizer_type': self.optimizer_type,
            'scheduler_type': self.scheduler_type,
            'warmup_steps': self.warmup_steps,
            'loss_function_type': self.loss_function_type,
            'quantile_alpha': self.quantile_alpha,
            'correlation_weight': self.correlation_weight,
            'mse_weight': self.mse_weight,
            'rank_correlation_weight': self.rank_correlation_weight,
            'target_correlations': self.target_correlations,
            'label_smoothing': self.label_smoothing,
            'gradient_clip': self.gradient_clip,
            'use_data_augmentation': self.use_data_augmentation,
            'noise_std': self.noise_std,
            'mask_probability': self.mask_probability,
            'model_save_dir': self.model_save_dir,
            'best_model_name': self.best_model_name,
            'checkpoint_interval': self.checkpoint_interval,
            'log_dir': self.log_dir,
            'use_wandb': self.use_wandb,
            'wandb_project': self.wandb_project,
            'wandb_run_name': self.wandb_run_name,
            'device': self.device,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'server_mode': self.server_mode,
            'conda_env': self.conda_env,
            'output_dir': self.output_dir,
            'use_mixed_precision': self.use_mixed_precision,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'max_grad_norm': self.max_grad_norm,
            'validation_interval': self.validation_interval,
            'save_best_only': self.save_best_only,
            'monitor_metric': self.monitor_metric,
            'monitor_mode': self.monitor_mode
        }

@dataclass
class InferenceConfig:
    """Inference configuration class for production deployment"""
    
    # Model configuration
    model_path: str = "models/best_model.pth"
    batch_size: int = 128
    use_ensemble: bool = True
    ensemble_size: int = 5
    temperature: float = 1.0  # For uncertainty quantification
    
    # Real-time inference configuration
    real_time_mode: bool = False
    max_latency_ms: float = 50.0
    cache_size: int = 1000
    
    # Emergency handling configuration
    fallback_model: str = "models/fallback_model.pth"
    confidence_threshold: float = 0.8
    max_prediction_std: float = 0.1
    
    # Performance configuration
    num_workers: int = 2
    pin_memory: bool = True
    use_mixed_precision: bool = True
    
    # API configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    
    # Security configuration
    require_api_key: bool = True
    api_key_header: str = "X-API-Key"
    rate_limit_per_minute: int = 1000
    
    def __post_init__(self):
        """Post-initialization processing"""
        # Validate model path
        if not os.path.exists(self.model_path):
            print(f"Warning: Model path does not exist: {self.model_path}")
        
        # Validate fallback model path
        if not os.path.exists(self.fallback_model):
            print(f"Warning: Fallback model path does not exist: {self.fallback_model}")

@dataclass
class DeploymentConfig:
    """Deployment configuration class for different environments"""
    
    # Environment configuration
    environment: str = "development"  # "development", "staging", "production"
    debug_mode: bool = False
    log_level: str = "INFO"
    
    # Resource configuration
    max_memory_gb: float = 8.0
    max_cpu_cores: int = 4
    gpu_memory_gb: float = 8.0
    
    # Monitoring configuration
    enable_metrics: bool = True
    metrics_port: int = 9090
    health_check_interval: int = 30
    
    # Security configuration
    enable_ssl: bool = False
    ssl_cert_path: str = ""
    ssl_key_path: str = ""
    
    # Backup configuration
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    backup_retention_days: int = 30

# Global configuration instances
config = ModelConfig()
inference_config = InferenceConfig()
deployment_config = DeploymentConfig()

def get_default_config() -> dict:
    """Return the default configuration dictionary for server usage"""
    return ModelConfig().__dict__

def load_config_from_yaml(yaml_path: str) -> ModelConfig:
    """Load configuration from YAML file"""
    import yaml
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Extract model config from YAML structure
    if 'model' in config_dict:
        model_config = config_dict['model']
    else:
        model_config = config_dict
    
    return ModelConfig(**model_config)

def save_config_to_yaml(config: ModelConfig, yaml_path: str):
    """Save configuration to YAML file"""
    import yaml
    config_dict = config.to_dict()
    
    # Organize into sections
    yaml_dict = {
        'model': {
            'name': f"{config.model_type}_{config.hidden_size}",
            'type': config.model_type,
            'hidden_size': config.hidden_size,
            'num_layers': config.num_layers,
            'num_heads': config.num_heads,
            'dropout': config.dropout,
            'activation': config.activation
        },
        'training': {
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'weight_decay': config.weight_decay,
            'num_epochs': config.num_epochs,
            'optimizer_type': config.optimizer_type,
            'scheduler_type': config.scheduler_type,
            'warmup_steps': config.warmup_steps,
            'early_stopping_patience': config.early_stopping_patience
        },
        'data': {
            'sequence_length': config.sequence_length,
            'prediction_horizon': config.prediction_horizon,
            'target_columns': config.target_columns,
            'factor_columns': config.factor_columns,
            'train_ratio': config.train_ratio,
            'val_ratio': config.val_ratio,
            'test_ratio': config.test_ratio
        },
        'loss': {
            'type': config.loss_function_type,
            'correlation_weight': config.correlation_weight,
            'mse_weight': config.mse_weight,
            'target_correlations': config.target_correlations
        },
        'output': {
            'model_save_dir': config.model_save_dir,
            'log_dir': config.log_dir,
            'use_wandb': config.use_wandb,
            'wandb_project': config.wandb_project
        },
        'hardware': {
            'device': config.device,
            'num_workers': config.num_workers,
            'pin_memory': config.pin_memory
        }
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_dict, f, default_flow_style=False, indent=2) 