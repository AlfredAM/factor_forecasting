"""
Enhanced main training script for factor forecasting model
Supports multi-file training with rolling window validation and testing
"""
import sys
import os
import argparse
import json
import logging
from datetime import datetime
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import warnings
# Only suppress specific warnings that are known to be harmless
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, message='.*NumPy.*')
import importlib.util

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import get_default_config, config, ModelConfig
from src.data_processing.data_processor import create_training_dataloaders, create_validation_dataloaders
from src.models.models import create_model
from src.training.trainer import create_trainer
from src.training.integrated_training import create_integrated_training_system
from src.utils.ic_analysis import create_ic_analyzer_from_predictions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_experiment(config: dict) -> str:
    """
    Setup experiment directory and save configuration.
    
    Args:
        config: Training configuration
        
    Returns:
        Experiment directory path
    """
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = getattr(config, 'experiment_name', f'factor_forecast_{timestamp}')
    experiment_dir = os.path.join(getattr(config, 'output_dir', 'outputs'), experiment_name)
    
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'ic_analysis'), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'checkpoints'), exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(experiment_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    # Update config with experiment directory
    if isinstance(config, dict):
        config['experiment_dir'] = experiment_dir
        config['checkpoint_dir'] = os.path.join(experiment_dir, 'checkpoints')
        config['log_dir'] = os.path.join(experiment_dir, 'logs')
    else:
        setattr(config, 'experiment_dir', experiment_dir)
        setattr(config, 'checkpoint_dir', os.path.join(experiment_dir, 'checkpoints'))
        setattr(config, 'log_dir', os.path.join(experiment_dir, 'logs'))
    
    logger.info(f"Experiment setup complete: {experiment_dir}")
    return experiment_dir


def validate_data(data_path: str = None, data_dir: str = None, config: dict = None) -> bool:
    """
    Validate input data file or directory.
    
    Args:
        data_path: Path to single data file (for validation/testing)
        data_dir: Path to data directory (for training)
        config: Configuration object
        
    Returns:
        True if data is valid, False otherwise
    """
    try:
        if data_dir:
            # Directory validation for daily parquet files
            from src.data_processing.data_processor import MultiFileDataProcessor
            # Convert dict config to object format
            import types
            config_obj = types.SimpleNamespace(**config)
            processor = MultiFileDataProcessor(config_obj)
            available_dates = processor.get_available_dates()
            if len(available_dates) == 0:
                logger.error("No valid data files found in directory")
                return False
            logger.info(f"Found {len(available_dates)} daily files")
            return True
        elif data_path:
            # Validation file processing
            if not data_path.endswith('.parquet'):
                logger.error("Data file must be in parquet format")
                return False
            
            df = pd.read_parquet(data_path)
            required_factor_cols = [str(i) for i in range(100)]
            required_target_cols = ['intra30m', 'nextT1d', 'ema1d']
            required_cols = ['sid'] + required_factor_cols + required_target_cols
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return False
            
            logger.info(f"Data validation successful: {len(df)} rows")
            return True
        else:
            logger.error("Either data_path or data_dir must be provided")
            return False
            
    except Exception as e:
        logger.error(f"Data validation failed: {str(e)}")
        return False


def prepare_training_data_loaders(config: dict):
    """
    Prepare training data loaders using multi-file mode.
    
    Args:
        config: Training configuration
        
    Returns:
        Tuple of train, validation, and test data loaders
    """
    logger.info("Preparing training data loaders using multi-file mode...")
    
    # Create data loaders
    dataloaders, scalers = create_training_dataloaders(config)
    
    logger.info(f"Training data loaders created:")
    logger.info(f"  Train batches: {len(dataloaders['train'])}")
    logger.info(f"  Validation batches: {len(dataloaders['val'])}")
    logger.info(f"  Test batches: {len(dataloaders['test'])}")
    
    return dataloaders['train'], dataloaders['val'], dataloaders['test'], scalers


def prepare_validation_data_loaders(data_path: str, config: dict):
    """
    Prepare validation data loaders using multi-file approach.
    
    Args:
        data_path: Path to validation data directory or file
        config: Training configuration
        
    Returns:
        Tuple of validation and test data loaders
    """
    logger.info(f"Preparing validation data loaders from: {data_path}")
    
    # Create data loaders
    dataloaders, scalers = create_validation_dataloaders(data_path, config)
    
    logger.info(f"Validation data loaders created:")
    logger.info(f"  Validation batches: {len(dataloaders['val'])}")
    logger.info(f"  Test batches: {len(dataloaders['test'])}")
    
    return dataloaders['val'], dataloaders['test'], scalers


def setup_model(config: dict):
    """
    Setup model and trainer.
    
    Args:
        config: Training configuration
        
    Returns:
        Tuple of model and trainer
    """
    logger.info("Setting up model and trainer...")
    
    # Create model
    model = create_model(config)
    
    # Create trainer
    trainer = create_trainer(model, config)
    
    # Log model information
    model_info = model.get_model_info()
    logger.info(f"Model created successfully:")
    logger.info(f"  Total parameters: {model_info['total_parameters']:,}")
    logger.info(f"  Model size: {model_info['model_size_mb']:.2f} MB")
    
    return model, trainer


def train_model(trainer, train_loader, val_loader, config: dict):
    """
    Train the model.
    
    Args:
        trainer: Model trainer
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        
    Returns:
        Training history
    """
    logger.info("Starting model training...")
    
    num_epochs = getattr(config, 'num_epochs', 100)
    resume_from_checkpoint = getattr(config, 'resume_from_checkpoint', None)
    
    # Train the model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        resume_from_checkpoint=resume_from_checkpoint
    )
    
    logger.info("Model training completed")
    return history


def evaluate_model(trainer, test_loader, config: dict):
    """
    Evaluate the trained model on test data.
    
    Args:
        trainer: Model trainer
        test_loader: Test data loader
        config: Training configuration
        
    Returns:
        Tuple of (metrics, ic_analysis)
    """
    logger.info("Evaluating model on test set...")
    
    # Evaluate model
    metrics = trainer.evaluate(test_loader)
    
    # Perform IC analysis
    ic_analysis = perform_ic_analysis(trainer, test_loader, config)
    
    # Convert numpy types to Python types for JSON serialization
    metrics_serializable = {}
    for target, target_metrics in metrics.items():
        metrics_serializable[target] = {}
        for metric_name, metric_value in target_metrics.items():
            if isinstance(metric_value, np.bool_):
                metrics_serializable[target][metric_name] = bool(metric_value)
            elif isinstance(metric_value, (np.integer, np.int32, np.int64)):
                metrics_serializable[target][metric_name] = int(metric_value)
            elif isinstance(metric_value, (np.floating, np.float32, np.float64)):
                metrics_serializable[target][metric_name] = float(metric_value)
            else:
                metrics_serializable[target][metric_name] = metric_value
    
    # Save evaluation results
    eval_path = os.path.join(getattr(config, 'experiment_dir', 'outputs'), 'logs', 'final_evaluation.json')
    with open(eval_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    
    logger.info("Model evaluation completed")
    return metrics, ic_analysis


def perform_ic_analysis(trainer, test_loader, config: dict):
    """
    Perform IC analysis on test data.
    
    Args:
        trainer: Model trainer
        test_loader: Test data loader
        config: Training configuration
        
    Returns:
        ICAnalyzer: IC analysis results
    """
    logger.info("Starting IC analysis...")
    
    # Initialize predictions and targets
    predictions = {target: [] for target in getattr(config, 'target_columns', [])}
    targets = {target: [] for target in getattr(config, 'target_columns', [])}
    dates = []
    stock_ids = []
    
    trainer.model.eval()
    with torch.no_grad():
        for batch in test_loader:
            # Move data to device
            features = batch['features'].to(trainer.device)
            batch_stock_ids = batch['stock_ids'].to(trainer.device)
            batch_targets = {k: v.to(trainer.device) for k, v in batch['targets'].items()}
            
            # Get predictions
            batch_predictions = trainer.model(features, batch_stock_ids)
            
            # Collect data - ensure proper flattening
            for target in predictions.keys():
                if target in batch_predictions and target in batch_targets:
                    # Flatten the predictions and targets to 1D arrays
                    pred_flat = batch_predictions[target].cpu().numpy().flatten()
                    target_flat = batch_targets[target].cpu().numpy().flatten()
                    
                    # Ensure they have the same length
                    min_len = min(len(pred_flat), len(target_flat))
                    predictions[target].extend(pred_flat[:min_len])
                    targets[target].extend(target_flat[:min_len])
            
            # Collect sequence info
            if 'sequence_info' in batch:
                for info in batch['sequence_info']:
                    dates.extend(info.get('target_dates', []))
                    stock_ids.extend([info.get('stock_id', 0)] * len(info.get('target_dates', [])))
    
    # Convert to numpy arrays and ensure they have the same length
    for target in predictions.keys():
        pred_array = np.array(predictions[target])
        target_array = np.array(targets[target])
        
        # Ensure both arrays have the same length
        min_len = min(len(pred_array), len(target_array))
        predictions[target] = pred_array[:min_len]
        targets[target] = target_array[:min_len]
        
        logger.info(f"IC analysis for {target}: {len(predictions[target])} samples")
    
    # If no dates were collected, create default dates for IC analysis
    if not dates:
        num_samples = len(next(iter(predictions.values())))
        dates = ['2020-01-01'] * num_samples  # Default date for all samples
        logger.info(f"No dates found in batch, using default date for {num_samples} samples")
    
    # Create IC analyzer
    ic_analyzer = create_ic_analyzer_from_predictions(
        predictions=predictions,
        targets=targets,
        dates=dates,
        stock_ids=stock_ids if stock_ids else None
    )
    
    # Generate IC analysis report
    ic_output_dir = os.path.join(getattr(config, 'experiment_dir', 'outputs'), 'ic_analysis')
    os.makedirs(ic_output_dir, exist_ok=True)
    ic_analyzer.generate_report(ic_output_dir)
    
    logger.info("IC analysis completed")
    return ic_analyzer


def save_final_model(trainer, config: dict):
    """
    Save the final trained model.
    
    Args:
        trainer: Model trainer
        config: Training configuration
    """
    logger.info("Saving final model...")
    
    # Save model state
    model_path = os.path.join(getattr(config, 'experiment_dir', 'outputs'), 'models', 'final_model.pth')
    torch.save(trainer.model.state_dict(), model_path)
    
    # Save model configuration
    model_config_path = os.path.join(getattr(config, 'experiment_dir', 'outputs'), 'models', 'model_config.json')
    with open(model_config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    logger.info(f"Final model saved to {model_path}")


def print_training_summary(history: dict, metrics: dict, config: dict, trainer):
    """
    Print training summary.
    
    Args:
        history: Training history
        metrics: Final evaluation metrics
        config: Training configuration
        trainer: Model trainer
    """
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    
    # Training statistics
    print(f"Experiment: {getattr(config, 'experiment_name', 'Unknown')}")
    print(f"Output directory: {getattr(config, 'experiment_dir', 'outputs')}")
    print(f"Total epochs: {len(history['train_loss'])}")
    print(f"Best validation loss: {min(history['val_loss']):.6f}")
    print(f"Final learning rate: {history['learning_rate'][-1]:.2e}")
    
    # Model information
    print(f"Model parameters: {trainer.model.get_model_info()['total_parameters']:,}")
    print(f"Model size: {trainer.model.get_model_info()['model_size_mb']:.2f} MB")
    
    # Final metrics
    print("\nFinal Test Metrics:")
    for target, target_metrics in metrics.items():
        print(f"  {target}:")
        for metric_name, metric_value in target_metrics.items():
            print(f"    {metric_name}: {metric_value:.6f}")
    
    print("="*80)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Factor Forecasting Model Training')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--data_path', type=str, default=None, help='Path to validation data file')
    parser.add_argument('--data_dir', type=str, default=None, help='Path to training data directory')
    parser.add_argument('--start_date', type=str, default=None, help='Start date for training (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None, help='End date for training (YYYY-MM-DD)')
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--num_epochs', type=int, default=None, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--device', type=str, default=None, help='Device (cpu/cuda)')
    parser.add_argument('--use_integrated_training', action='store_true', 
                       help='Use integrated training system with async preloading and incremental learning')
    parser.add_argument('--disable_incremental', action='store_true',
                       help='Disable incremental learning in integrated training')
    parser.add_argument('--disable_async', action='store_true',
                       help='Disable async preloading in integrated training')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        if args.config.endswith('.py'):
            spec = importlib.util.spec_from_file_location('user_config', args.config)
            user_config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(user_config)
            config = user_config.config if hasattr(user_config, 'config') else user_config.get_default_config()
        else:
            with open(args.config, 'r') as f:
                config = json.load(f)
    else:
        config = get_default_config()
    
    # Update config with command line arguments
    if args.data_path:
        setattr(config, 'data_path', args.data_path)
    if args.data_dir:
        setattr(config, 'data_dir', args.data_dir)
    if args.start_date:
        setattr(config, 'start_date', args.start_date)
    
    # Handle integrated training
    if args.use_integrated_training:
        logger.info("Using integrated training system")
        
        # Convert config to ModelConfig if needed
        if not isinstance(config, ModelConfig):
            model_config = ModelConfig()
            for key, value in config.__dict__.items() if hasattr(config, '__dict__') else vars(config).items():
                if hasattr(model_config, key):
                    setattr(model_config, key, value)
            config = model_config
        
        # Override with command line arguments
        if args.num_epochs:
            config.num_epochs = args.num_epochs
        if args.batch_size:
            config.batch_size = args.batch_size
        if args.learning_rate:
            config.learning_rate = args.learning_rate
        
        # Setup device
        device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Create integrated training system
        training_system = create_integrated_training_system(
            config=config,
            device=device,
            use_incremental_learning=not args.disable_incremental,
            use_async_preloader=not args.disable_async
        )
        
        # Get data directory
        data_dir = args.data_dir or getattr(config, 'data_dir', './data')
        
        logger.info(f"Starting integrated training with data from: {data_dir}")
        
        # Run integrated training
        results = training_system.train(
            data_dir=data_dir,
            epochs=args.num_epochs or getattr(config, 'num_epochs', 100),
            validation_split=0.2
        )
        
        logger.info(f"Integrated training completed. Best validation loss: {results['best_val_loss']:.4f}")
        return results
    
    # Continue with standard training if integrated training not selected
    if args.end_date:
        setattr(config, 'end_date', args.end_date)
    if args.experiment_name:
        setattr(config, 'experiment_name', args.experiment_name)
    if args.resume_from_checkpoint:
        setattr(config, 'resume_from_checkpoint', args.resume_from_checkpoint)
    if args.num_epochs:
        setattr(config, 'num_epochs', args.num_epochs)
    if args.batch_size:
        setattr(config, 'batch_size', args.batch_size)
    if args.learning_rate:
        setattr(config, 'learning_rate', args.learning_rate)
    if args.device:
        setattr(config, 'device', args.device)
    
    print("Loaded config:", config)
    print("data_path:", getattr(config, "data_path", None))
    print("data_dir:", getattr(config, "data_dir", None))
    # Setup experiment
    experiment_dir = setup_experiment(config)
    
    try:
        # Validate data
        if not validate_data(data_path=getattr(config, "data_path", None), data_dir=getattr(config, "data_dir", None), config=config):
            logger.error("Data validation failed")
            return
        
        # Prepare training data loaders
        train_loader, val_loader, test_loader, scalers = prepare_training_data_loaders(config)
        
        # Prepare validation data loaders if validation file provided
        val_file_loader = None
        test_file_loader = None
        if args.data_path:
            val_file_loader, test_file_loader, _ = prepare_validation_data_loaders(args.data_path, config)
        
        # Setup model and trainer
        model, trainer = setup_model(config)
        
        # Train model
        history = train_model(trainer, train_loader, val_loader, config)
        
        # Evaluate model
        if test_file_loader:
            # Use validation file test data
            metrics, ic_results = evaluate_model(trainer, test_file_loader, config)
        else:
            # Use multi-file test data
            metrics, ic_results = evaluate_model(trainer, test_loader, config)
        
        # Save final model
        save_final_model(trainer, config)
        
        # Print summary
        print_training_summary(history, metrics, config, trainer)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main() 