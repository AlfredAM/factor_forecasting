"""
Transformer model for factor forecasting with multi-task learning
Combines stock embeddings, positional encoding, transformer blocks, and task-specific heads
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

from .embeddings import PositionalEncoding, StockEmbedding
from .attention import AttentionBlock

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FactorTransformerModel(nn.Module):
    """
    Main Transformer model for factor forecasting with multi-task learning.
    Combines stock embeddings, positional encoding, transformer blocks, and task-specific heads.
    Optimized for correlation maximization on three targets.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize FactorTransformer model.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        super(FactorTransformerModel, self).__init__()
        
        # Extract configuration parameters
        self.num_factors = getattr(config, 'num_factors', 100)
        self.num_stocks = getattr(config, 'num_stocks', 1000)
        self.d_model = getattr(config, 'd_model', 256)
        self.num_heads = getattr(config, 'num_heads', 8)
        self.num_layers = getattr(config, 'num_layers', 6)
        self.d_ff = getattr(config, 'd_ff', 1024)
        self.dropout = getattr(config, 'dropout', 0.1)
        self.max_seq_len = getattr(config, 'max_seq_len', 50)
        
        # Target columns for multi-task learning
        self.target_columns = getattr(config, 'target_columns', ['intra30m', 'nextT1d', 'ema1d'])
        self.num_targets = len(self.target_columns)
        
        # Factor embedding layer
        self.factor_embedding = nn.Linear(self.num_factors, self.d_model)
        
        # Stock embedding layer
        self.stock_embedding = StockEmbedding(self.num_stocks, self.d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(self.d_model, self.max_seq_len)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            AttentionBlock(self.d_model, self.num_heads, self.d_ff, self.dropout)
            for _ in range(self.num_layers)
        ])
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Enhanced task-specific output heads with correlation optimization
        self.shared_features = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.LayerNorm(self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Individual task heads
        self.task_heads = nn.ModuleDict({
            target: nn.Sequential(
                nn.Linear(self.d_model // 2, self.d_model // 4),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model // 4, 1)
            ) for target in self.target_columns
        })
        
        # Correlation optimization layer
        self.correlation_layer = nn.Sequential(
            nn.Linear(self.d_model // 2, self.d_model // 4),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 4, self.num_targets),
            nn.Tanh()  # Output correlation weights
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(self.dropout)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"Initialized FactorTransformer with {self.num_layers} layers, "
                   f"{self.num_heads} heads, d_model={self.d_model}, targets={self.target_columns}")
    
    def _init_weights(self):
        """Initialize model weights with better initialization for correlation optimization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use Xavier initialization for better gradient flow
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                # Initialize embeddings with smaller variance for stability
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
    
    def create_padding_mask(self, seq_lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """
        Create padding mask for variable length sequences.
        
        Args:
            seq_lengths: Tensor containing actual sequence lengths
            max_len: Maximum sequence length
            
        Returns:
            Padding mask tensor
        """
        batch_size = seq_lengths.size(0)
        mask = torch.arange(max_len).expand(batch_size, max_len) < seq_lengths.unsqueeze(1)
        return mask.unsqueeze(1).unsqueeze(1)  # Shape: (batch_size, 1, 1, max_len)
    
    def forward(self, factors: torch.Tensor, stock_ids: torch.Tensor, 
                seq_lengths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model with enhanced correlation optimization.
        
        Args:
            factors: Factor data tensor of shape (batch_size, seq_len, num_factors)
            stock_ids: Stock ID tensor of shape (batch_size, seq_len)
            seq_lengths: Optional tensor containing actual sequence lengths
            
        Returns:
            Dictionary containing predictions for each target and correlation weights
        """
        batch_size, seq_len, _ = factors.shape
        
        # Create embeddings
        factor_embeddings = self.factor_embedding(factors)  # (batch_size, seq_len, d_model)
        stock_embeddings = self.stock_embedding(stock_ids)  # (batch_size, seq_len, d_model)
        
        # Combine factor and stock embeddings
        combined_embeddings = factor_embeddings + stock_embeddings
        
        # Add positional encoding
        combined_embeddings = combined_embeddings.transpose(0, 1)  # (seq_len, batch_size, d_model)
        combined_embeddings = self.pos_encoding(combined_embeddings)
        combined_embeddings = combined_embeddings.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Apply dropout
        combined_embeddings = self.dropout(combined_embeddings)
        
        # Create attention mask if sequence lengths are provided
        attention_mask = None
        if seq_lengths is not None:
            attention_mask = self.create_padding_mask(seq_lengths, seq_len)
        
        # Pass through transformer blocks
        x = combined_embeddings
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, attention_mask)
        
        # Global average pooling across sequence dimension
        x = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
        x = self.global_pool(x).squeeze(-1)  # (batch_size, d_model)
        
        # Extract shared features
        shared_features = self.shared_features(x)  # (batch_size, d_model // 2)
        
        # Generate predictions for each target
        predictions = {}
        for target_name, head in self.task_heads.items():
            predictions[target_name] = head(shared_features).squeeze(-1)
        
        # Generate correlation optimization weights
        correlation_weights = self.correlation_layer(shared_features)  # (batch_size, num_targets)
        
        # Add correlation weights to output
        predictions['correlation_weights'] = correlation_weights
        
        return predictions


class FactorForecastingModel(nn.Module):
    """
    Wrapper class for the FactorTransformer model with additional utilities.
    Provides a unified interface for training and inference.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize FactorForecastingModel.
        
        Args:
            config: Configuration dictionary
        """
        super(FactorForecastingModel, self).__init__()
        
        self.config = config
        self.transformer = FactorTransformerModel(config)
        
        # Loss functions for each target
        self.loss_functions = {
            'intra30m': nn.MSELoss(),
            'nextT1d': nn.MSELoss(),
            'ema1d': nn.MSELoss()
        }
        
        logger.info("Initialized FactorForecastingModel")
    
    def forward(self, factors: torch.Tensor, stock_ids: torch.Tensor,
                seq_lengths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            factors: Factor data tensor
            stock_ids: Stock ID tensor
            seq_lengths: Optional sequence lengths tensor
            
        Returns:
            Dictionary containing predictions for each target
        """
        return self.transformer(factors, stock_ids, seq_lengths)
    
    def compute_loss(self, predictions: Dict[str, torch.Tensor], 
                    targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute loss for each target.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary containing losses for each target
        """
        losses = {}
        total_loss = 0.0
        
        for target_name in self.config.get('target_columns', ['intra30m', 'nextT1d', 'ema1d']):
            if target_name in predictions and target_name in targets:
                loss = self.loss_functions[target_name](predictions[target_name], targets[target_name])
                losses[target_name] = loss
                total_loss += loss
        
        losses['total'] = total_loss
        return losses
    
    def get_model_info(self) -> Dict:
        """
        Get model information and statistics.
        
        Returns:
            Dictionary containing model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'num_layers': self.config.get('num_layers', 6),
            'd_model': self.config.get('d_model', 256),
            'num_heads': self.config.get('num_heads', 8),
            'target_columns': self.config.get('target_columns', ['intra30m', 'nextT1d', 'ema1d'])
        } 