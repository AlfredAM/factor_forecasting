"""
Advanced TCN+Attention Model with Improved Attention Mechanisms and Regularization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
import logging
from .advanced_attention import (
    AdvancedTCNAttentionBlock, 
    RelativePositionalEncoding,
    GatedLinearUnit,
    StochasticDepth
)

logger = logging.getLogger(__name__)


class StockEmbedding(nn.Module):
    """
    Enhanced stock embedding layer with regularization.
    """
    
    def __init__(self, num_stocks: int, embedding_dim: int, dropout: float = 0.1):
        super(StockEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_stocks, embedding_dim)
        self.dropout = nn.dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        # Initialize embeddings with improved initialization
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
    
    def forward(self, stock_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert stock IDs to embeddings with regularization.
        
        Args:
            stock_ids: Stock ID tensor of shape (batch_size, seq_len)
            
        Returns:
            Stock embeddings of shape (batch_size, seq_len, embedding_dim)
        """
        embeddings = self.embedding(stock_ids)
        embeddings = self.dropout(embeddings)
        embeddings = self.layer_norm(embeddings)
        return embeddings


class AdvancedFactorTCNAttention(nn.Module):
    """
    Advanced Factor TCN-Attention model with improved attention mechanisms and regularization.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize Advanced Factor TCN-Attention model.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        super(AdvancedFactorTCNAttention, self).__init__()
        
        # Extract configuration parameters
        self.input_dim = config.get('input_dim', 100)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.num_layers = config.get('num_layers', 6)
        self.num_heads = config.get('num_heads', 8)
        self.dropout = config.get('dropout', 0.1)
        self.num_stocks = config.get('num_stocks', 1000)
        self.sequence_length = config.get('sequence_length', 10)
        self.target_columns = config.get('target_columns', ['nextT1d'])
        self.kernel_size = config.get('kernel_size', 3)
        
        # Advanced configuration
        self.use_relative_pos = config.get('use_relative_pos', True)
        self.use_multi_scale = config.get('use_multi_scale', True)
        self.use_adaptive = config.get('use_adaptive', True)
        self.use_stochastic_depth = config.get('use_stochastic_depth', True)
        self.use_gated_units = config.get('use_gated_units', True)
        
        # Enhanced stock embedding
        self.stock_embedding = StockEmbedding(self.num_stocks, self.hidden_dim, self.dropout)
        
        # Input projection with gating
        if self.use_gated_units:
            self.input_projection = GatedLinearUnit(self.input_dim, self.hidden_dim, self.dropout)
        else:
            self.input_projection = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.dropout(self.dropout),
                nn.LayerNorm(self.hidden_dim)
            )
        
        # Advanced TCN-Attention blocks with exponential dilation
        self.tcn_attention_blocks = nn.ModuleList([
            AdvancedTCNAttentionBlock(
                d_model=self.hidden_dim,
                num_heads=self.num_heads,
                kernel_size=self.kernel_size,
                dropout=self.dropout,
                dilation=2**i,  # Exponential dilation
                use_relative_pos=self.use_relative_pos,
                use_multi_scale=self.use_multi_scale,
                use_adaptive=self.use_adaptive
            )
            for i in range(self.num_layers)
        ])
        
        # Stochastic depth for regularization
        if self.use_stochastic_depth:
            self.stochastic_depth_layers = nn.ModuleList([
                StochasticDepth(p=0.1 * (i + 1) / self.num_layers)
                for i in range(self.num_layers)
            ])
        
        # Feature fusion layer
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.dropout(self.dropout),
            nn.LayerNorm(self.hidden_dim)
        )
        
        # Output projection with multiple heads
        self.output_projection = nn.ModuleDict({
            target: nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.dropout(self.dropout),
                nn.Linear(self.hidden_dim // 2, 1)
            )
            for target in self.target_columns
        })
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"Initialized AdvancedFactorTCNAttention with {self.num_layers} layers, "
                   f"{self.num_heads} heads, d_model={self.hidden_dim}, targets={self.target_columns}")
    
    def _init_weights(self):
        """Initialize model weights with improved initialization"""
        # Initialize input projection
        if not self.use_gated_units:
            for module in self.input_projection:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        
        # Initialize output projections
        for target_proj in self.output_projection.values():
            for module in target_proj:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        
        # Initialize feature fusion
        for module in self.feature_fusion:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, factors: torch.Tensor, stock_ids: torch.Tensor, 
                seq_lengths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of Advanced Factor TCN-Attention.
        
        Args:
            factors: Factor tensor of shape (batch_size, seq_len, input_dim)
            stock_ids: Stock ID tensor of shape (batch_size, seq_len)
            seq_lengths: Optional tensor of sequence lengths
            
        Returns:
            Dictionary containing predictions for each target
        """
        batch_size, seq_len, _ = factors.shape
        
        # Input projection
        if self.use_gated_units:
            x = self.input_projection(factors)  # (batch_size, seq_len, hidden_dim)
        else:
            x = self.input_projection(factors)
        
        # Add stock embeddings
        stock_embeddings = self.stock_embedding(stock_ids)  # (batch_size, seq_len, hidden_dim)
        x = x + stock_embeddings
        
        # Create attention mask if sequence lengths are provided
        mask = None
        if seq_lengths is not None:
            mask = self._create_padding_mask(seq_lengths, seq_len)
        
        # Store intermediate representations for feature fusion
        intermediate_outputs = []
        
        # Apply TCN-Attention blocks with stochastic depth
        for i, tcn_attention_block in enumerate(self.tcn_attention_blocks):
            # Apply stochastic depth if enabled
            if self.use_stochastic_depth:
                x = self.stochastic_depth_layers[i](x)
            
            # Apply TCN-Attention block
            x = tcn_attention_block(x, mask)
            
            # Store intermediate output for feature fusion
            if i % 2 == 0:  # Store every other layer
                intermediate_outputs.append(x)
        
        # Feature fusion: combine current output with intermediate representations
        if len(intermediate_outputs) > 0:
            # Use the last intermediate output for fusion
            fused_features = torch.cat([x, intermediate_outputs[-1]], dim=-1)
            x = self.feature_fusion(fused_features)
        else:
            # Ensure feature_fusion is always used (to prevent unused parameters in DDP)
            # Use identity fusion when no intermediate outputs
            dummy_features = torch.cat([x, torch.zeros_like(x)], dim=-1)
            x = self.feature_fusion(dummy_features)
        
        # Output projection for each target
        result = {}
        for target_name, target_proj in self.output_projection.items():
            target_output = target_proj(x).squeeze(-1)  # (batch_size, seq_len)
            # Ensure correct output dimensions
            if target_output.dim() == 1:
                target_output = target_output.unsqueeze(0)
            result[target_name] = target_output
        
        return result
    
    def _create_padding_mask(self, seq_lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """
        Create padding mask for variable length sequences.
        
        Args:
            seq_lengths: Tensor of sequence lengths
            max_len: Maximum sequence length
            
        Returns:
            Padding mask tensor
        """
        batch_size = seq_lengths.size(0)
        mask = torch.arange(max_len).expand(batch_size, max_len) < seq_lengths.unsqueeze(1)
        return mask.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, seq_len)


class AdvancedFactorForecastingTCNAttentionModel(nn.Module):
    """
    Advanced Factor Forecasting TCN-Attention model wrapper.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize Advanced Factor Forecasting TCN-Attention model.
        
        Args:
            config: Configuration dictionary
        """
        super(AdvancedFactorForecastingTCNAttentionModel, self).__init__()
        
        self.model = AdvancedFactorTCNAttention(config)
        self.config = config
        
        logger.info("Initialized AdvancedFactorForecastingTCNAttentionModel")
    
    def forward(self, factors: torch.Tensor, stock_ids: torch.Tensor,
                seq_lengths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            factors: Factor tensor
            stock_ids: Stock ID tensor
            seq_lengths: Optional sequence lengths
            
        Returns:
            Model predictions
        """
        return self.model(factors, stock_ids, seq_lengths)
    
    def compute_loss(self, predictions: Dict[str, torch.Tensor], 
                    targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute loss with advanced regularization.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary containing loss values
        """
        losses = {}
        total_loss = 0.0
        
        for target_name in predictions.keys():
            if target_name in targets:
                # MSE loss
                mse_loss = F.mse_loss(predictions[target_name], targets[target_name])
                
                # MAE loss for robustness
                mae_loss = F.l1_loss(predictions[target_name], targets[target_name])
                
                # Huber loss for outlier robustness
                huber_loss = F.smooth_l1_loss(predictions[target_name], targets[target_name])
                
                # Combined loss
                target_loss = 0.5 * mse_loss + 0.3 * mae_loss + 0.2 * huber_loss
                
                losses[f"{target_name}_mse"] = mse_loss
                losses[f"{target_name}_mae"] = mae_loss
                losses[f"{target_name}_huber"] = huber_loss
                losses[f"{target_name}_total"] = target_loss
                
                total_loss += target_loss
        
        # Add L2 regularization
        l2_reg = 0.0
        for param in self.parameters():
            l2_reg += torch.norm(param, p=2)
        
        l2_weight = self.config.get('l2_weight', 1e-5)
        total_loss += l2_weight * l2_reg
        
        losses['total_loss'] = total_loss
        losses['l2_reg'] = l2_reg
        
        return losses
    
    def get_model_info(self) -> Dict:
        """
        Get model information.
        
        Returns:
            Dictionary containing model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'AdvancedFactorForecastingTCNAttention',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'config': self.config,
            'features': {
                'relative_positional_encoding': self.config.get('use_relative_pos', True),
                'multi_scale_attention': self.config.get('use_multi_scale', True),
                'adaptive_attention': self.config.get('use_adaptive', True),
                'stochastic_depth': self.config.get('use_stochastic_depth', True),
                'gated_units': self.config.get('use_gated_units', True)
            }
        }


def create_advanced_model(config: Dict) -> nn.Module:
    """
    Factory function to create advanced TCN-Attention model.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Advanced TCN-Attention model
    """
    return AdvancedFactorForecastingTCNAttentionModel(config) 