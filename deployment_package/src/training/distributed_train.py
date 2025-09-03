"""
Distributed training script for factor forecasting models
Supports DDP multi-GPU training with proper process management
"""
import sys
import os
import argparse
import json
import logging
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from datetime import datetime, timedelta

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configs.config import get_default_config
from src.data_processing.data_pipeline import create_continuous_data_loaders
from src.models.models import create_model
from src.training.trainer import create_trainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_distributed(rank: int, world_size: int, port: str = "12355"):
    """
    Setup distributed training environment.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        port: Port for distributed communication
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    
    # Initialize process group with timeout
    dist.init_process_group(
        backend='nccl', 
        rank=rank, 
        world_size=world_size,
        timeout=timedelta(minutes=30)
    )
    
    # Set device
    torch.cuda.set_device(rank)
    
    logger.info(f"Process {rank}/{world_size} initialized on GPU {rank}")


def cleanup_distributed():
    """Cleanup distributed training environment."""
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        # Be resilient during teardown
        pass


def train_worker(rank: int, world_size: int, config: dict, args):
    """
    Training worker function for each process.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        config: Training configuration
        args: Command line arguments
    """
    try:
        # Setup distributed environment
        setup_distributed(rank, world_size, args.port)
        
        # Set device
        device = torch.device(f'cuda:{rank}')
        config['device'] = str(device)
        
        # Create model
        model = create_model(config)
        model.to(device)
        
        # Wrap model with DDP
        model = DDP(model, device_ids=[rank], output_device=rank)
        
        # Create trainer
        trainer = create_trainer(model, config, rank, world_size)
        
        # Create data loaders with distributed sampling
        train_loader, val_loader, test_loader = create_continuous_data_loaders(config)
        
        # Wrap data loaders with DistributedSampler
        train_sampler = DistributedSampler(
            train_loader.dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        
        val_sampler = DistributedSampler(
            val_loader.dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_loader.dataset,
            batch_size=config['batch_size'],
            sampler=train_sampler,
            num_workers=getattr(config, 'num_workers', 4),
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_loader.dataset,
            batch_size=config['batch_size'],
            sampler=val_sampler,
            num_workers=getattr(config, 'num_workers', 4),
            pin_memory=True,
            drop_last=False
        )
        
        # Log model information on main process
        if rank == 0:
            model_info = model.module.get_model_info()
            logger.info(f"Model created: {model_info['total_parameters']:,} parameters")
            logger.info(f"Training on {world_size} GPUs")
        
        # Train model
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=getattr(config, 'num_epochs', 100),
            resume_from_checkpoint=args.resume_from_checkpoint
        )
        
        # Evaluate model on main process
        if rank == 0:
            logger.info("Evaluating model...")
            metrics = trainer.evaluate(test_loader)
            
            # Log final results
            logger.info("Training completed successfully!")
            for target, target_metrics in metrics.items():
                logger.info(f"{target}:")
                for metric_name, metric_value in target_metrics.items():
                    logger.info(f"  {metric_name}: {metric_value:.6f}")
        
    except Exception as e:
        logger.error(f"Process {rank} failed: {str(e)}")
        raise
    finally:
        # Cleanup distributed environment
        cleanup_distributed()


def main():
    """Main function for distributed training."""
    parser = argparse.ArgumentParser(description='Distributed Factor Forecasting Training')
    parser.add_argument('--data_dir', type=str, default='/nas/feature_v2_10s',
                       help='Directory containing daily parquet files')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--num_gpus', type=int, default=None,
                       help='Number of GPUs to use (default: all available)')
    parser.add_argument('--port', type=str, default='12355',
                       help='Port for distributed communication')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Name for this experiment')
    parser.add_argument('--num_epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Training batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config_path and os.path.exists(args.config_path):
        with open(args.config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {args.config_path}")
    else:
        config = get_default_config()
        logger.info("Using default configuration")
    
    # Override config with command line arguments
    if args.experiment_name:
        config['experiment_name'] = args.experiment_name
    if args.num_epochs:
        config['num_epochs'] = args.num_epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    
    # Set data directory
    config['data_dir'] = args.data_dir
    
    # Determine number of GPUs
    if args.num_gpus is not None:
        world_size = args.num_gpus
    else:
        world_size = torch.cuda.device_count()
    
    if world_size == 0:
        logger.error("No GPUs available for training")
        sys.exit(1)
    
    logger.info(f"Starting distributed training on {world_size} GPUs")
    
    # Set distributed training specific configurations
    config['is_distributed'] = True
    config['world_size'] = world_size
    
    # Adjust batch size for distributed training
    if 'batch_size' in config:
        config['batch_size_per_gpu'] = config['batch_size']
        config['batch_size'] = config['batch_size'] * world_size
        logger.info(f"Total batch size: {config['batch_size']} ({config['batch_size_per_gpu']} per GPU)")
    
    # Adjust learning rate for distributed training
    if 'learning_rate' in config:
        config['learning_rate'] = config['learning_rate'] * world_size
        logger.info(f"Scaled learning rate: {config['learning_rate']}")
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = config.get('experiment_name', f'factor_forecast_{timestamp}')
    experiment_dir = os.path.join(config.get('output_dir', 'outputs'), experiment_name)
    
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'checkpoints'), exist_ok=True)
    
    # Update config with experiment directory
    config['experiment_dir'] = experiment_dir
    config['checkpoint_dir'] = os.path.join(experiment_dir, 'checkpoints')
    config['log_dir'] = os.path.join(experiment_dir, 'logs')
    
    # Save configuration
    config_path = os.path.join(experiment_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    logger.info(f"Experiment directory: {experiment_dir}")
    
    try:
        # Start distributed training
        mp.spawn(
            train_worker,
            args=(world_size, config, args),
            nprocs=world_size,
            join=True
        )
        
        logger.info("Distributed training completed successfully!")
        
    except Exception as e:
        logger.error(f"Distributed training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 