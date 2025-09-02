"""
Advanced Attention Mechanisms for Factor Forecasting
Includes improved attention mechanisms and regularization techniques
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class RelativePositionalEncoding(nn.Module):
    """
    Relative positional encoding for better sequence modeling.
    Based on Shaw et al. "Self-Attention with Relative Position Representations"
    """
    
    def __init__(self, d_model: int, max_relative_position: int = 32):
        super(RelativePositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_relative_position = max_relative_position
        
        # Relative position embeddings
        self.relative_attention_bias = nn.Embedding(
            2 * max_relative_position + 1, 
            d_model
        )
        
        # Initialize embeddings
        nn.init.normal_(self.relative_attention_bias.weight, mean=0.0, std=0.02)
    
    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generate relative position embeddings.
        
        Args:
            seq_len: Sequence length
            device: Device to place tensors on
            
        Returns:
            Relative position embeddings with shape (1, seq_len, d_model) for broadcasting
        """
        range_vec = torch.arange(seq_len, device=device)
        range_mat = range_vec.unsqueeze(0).repeat(seq_len, 1)
        distance_mat = range_mat - range_mat.transpose(0, 1)
        
        # Clip distances to max_relative_position
        distance_mat_clipped = torch.clamp(
            distance_mat, 
            -self.max_relative_position, 
            self.max_relative_position
        )
        
        # Shift to non-negative indices
        final_mat = distance_mat_clipped + self.max_relative_position
        
        # Get relative embeddings and take mean over sequence dimension for positional encoding
        # Shape: (seq_len, seq_len, d_model) -> (seq_len, d_model) -> (1, seq_len, d_model)
        rel_emb = self.relative_attention_bias(final_mat)  # (seq_len, seq_len, d_model)
        pos_emb = rel_emb.mean(dim=0).unsqueeze(0)  # (1, seq_len, d_model) for broadcasting
        
        return pos_emb


class GatedLinearUnit(nn.Module):
    """
    Gated Linear Unit for better feature selection and regularization.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super(GatedLinearUnit, self).__init__()
        
        self.fc = nn.Linear(input_dim, hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of GLU.
        
        Args:
            x: Input tensor
            
        Returns:
            Gated output
        """
        x = self.fc(x)
        x = self.dropout(x)
        
        # Split into gates and values
        gates, values = x.chunk(2, dim=-1)
        
        # Apply gating mechanism
        return F.sigmoid(gates) * values


class MultiScaleAttention(nn.Module):
    """
    Multi-scale attention mechanism that attends to different temporal scales.
    """
    
    def __init__(self, d_model: int, num_heads: int, scales: List[int] = [1, 2, 4], dropout: float = 0.1):
        super(MultiScaleAttention, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.scales = scales
        self.d_k = d_model // num_heads
        
        # Multi-scale attention heads
        self.scale_attentions = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
            for _ in scales
        ])
        
        # Scale fusion
        self.scale_fusion = nn.Linear(d_model * len(scales), d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of multi-scale attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Multi-scale attention output
        """
        batch_size, seq_len, d_model = x.shape
        scale_outputs = []
        
        for i, (scale, attention) in enumerate(zip(self.scales, self.scale_attentions)):
            # Downsample for larger scales
            if scale > 1:
                # Average pooling for downsampling
                x_downsampled = F.avg_pool1d(
                    x.transpose(1, 2), 
                    kernel_size=scale, 
                    stride=scale
                ).transpose(1, 2)
                
                # Apply attention on downsampled sequence
                attn_out, _ = attention(x_downsampled, x_downsampled, x_downsampled, attn_mask=mask)
                
                # Upsample back to original sequence length
                attn_out = F.interpolate(
                    attn_out.transpose(1, 2), 
                    size=seq_len, 
                    mode='linear', 
                    align_corners=False
                ).transpose(1, 2)
            else:
                # No downsampling for scale=1
                attn_out, _ = attention(x, x, x, attn_mask=mask)
            
            scale_outputs.append(attn_out)
        
        # Concatenate and fuse multi-scale outputs
        multi_scale_output = torch.cat(scale_outputs, dim=-1)
        fused_output = self.scale_fusion(multi_scale_output)
        fused_output = self.dropout(fused_output)
        
        # Residual connection and layer normalization
        output = self.layer_norm(x + fused_output)
        
        return output


class AdaptiveAttention(nn.Module):
    """
    Adaptive attention mechanism that dynamically adjusts attention weights.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super(AdaptiveAttention, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Standard attention components
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Adaptive components
        self.attention_gate = nn.Linear(d_model, num_heads)
        self.context_gate = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of adaptive attention.
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            mask: Optional attention mask
            
        Returns:
            Attention output and attention weights
        """
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Adaptive attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Compute attention gate
        attention_gate = torch.sigmoid(self.attention_gate(query.mean(dim=1)))  # (batch_size, num_heads)
        attention_gate = attention_gate.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, num_heads)
        
        # Apply adaptive gating
        attention_weights = attention_weights * attention_gate
        
        attention_weights = self.dropout(attention_weights)
        attention_output = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        # Context gating
        context_gate = torch.sigmoid(self.context_gate(attention_output))
        attention_output = attention_output * context_gate
        
        # Output projection
        output = self.w_o(attention_output)
        output = self.dropout(output)
        
        # Residual connection and layer normalization
        output = self.layer_norm(query + output)
        
        return output, attention_weights


class StochasticDepth(nn.Module):
    """
    Stochastic depth for regularization during training.
    """
    
    def __init__(self, p: float = 0.1):
        super(StochasticDepth, self).__init__()
        self.p = p
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with stochastic depth.
        
        Args:
            x: Input tensor
            
        Returns:
            Output with potential dropout
        """
        if not self.training or self.p == 0:
            return x
        
        # Randomly drop the entire residual branch
        if torch.rand(1) < self.p:
            return torch.zeros_like(x)
        
        return x


class AdvancedTCNAttentionBlock(nn.Module):
    """
    Advanced TCN-Attention block with improved attention mechanisms and regularization.
    """
    
    def __init__(self, d_model: int, num_heads: int, kernel_size: int = 3, 
                 dropout: float = 0.1, dilation: int = 1, use_relative_pos: bool = True,
                 use_multi_scale: bool = True, use_adaptive: bool = True):
        super(AdvancedTCNAttentionBlock, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_relative_pos = use_relative_pos
        self.use_multi_scale = use_multi_scale
        self.use_adaptive = use_adaptive
        
        # TCN components with improved regularization
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size, 
                               padding=(kernel_size-1) * dilation, 
                               dilation=dilation)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size, 
                               padding=(kernel_size-1) * dilation, 
                               dilation=dilation)
        
        # Chomp layers
        self.chomp1 = Chomp1d((kernel_size-1) * dilation)
        self.chomp2 = Chomp1d((kernel_size-1) * dilation)
        
        # Advanced attention mechanisms
        if use_multi_scale:
            self.attention = MultiScaleAttention(d_model, num_heads, dropout=dropout)
        elif use_adaptive:
            self.attention = AdaptiveAttention(d_model, num_heads, dropout=dropout)
        else:
            self.attention = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        
        # Relative positional encoding
        if use_relative_pos:
            self.relative_pos_encoding = RelativePositionalEncoding(d_model)
        
        # Gated components
        self.gated_tcn = GatedLinearUnit(d_model, d_model, dropout)
        self.gated_attention = GatedLinearUnit(d_model, d_model, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout and regularization
        self.dropout = nn.Dropout(dropout)
        self.stochastic_depth = StochasticDepth(p=0.1)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, d_model)
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights with improved initialization"""
        # Xavier initialization for convolutions
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        
        # Kaiming initialization for linear layers
        nn.init.kaiming_normal_(self.output_projection.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.output_projection.bias)
        
        # Initialize batch norm if used
        if hasattr(self, 'bn1'):
            nn.init.constant_(self.bn1.weight, 1)
            nn.init.constant_(self.bn1.bias, 0)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of advanced TCN-Attention block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        residual = x
        
        # TCN branch with gating
        x_tcn = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
        x_tcn = F.relu(self.conv1(x_tcn))
        x_tcn = self.chomp1(x_tcn)
        x_tcn = self.dropout(x_tcn)
        x_tcn = F.relu(self.conv2(x_tcn))
        x_tcn = self.chomp2(x_tcn)
        x_tcn = self.dropout(x_tcn)
        x_tcn = x_tcn.transpose(1, 2)  # (batch_size, seq_len, d_model)
        
        # Apply gated TCN
        x_tcn = self.gated_tcn(x_tcn)
        
        # Residual connection for TCN with stochastic depth
        x = self.norm1(residual + self.stochastic_depth(x_tcn))
        
        # Attention branch with relative positional encoding
        if self.use_relative_pos:
            rel_pos_emb = self.relative_pos_encoding(seq_len, x.device)
            x = x + rel_pos_emb
        
        if self.use_multi_scale:
            x_attn = self.attention(x, mask)
        elif self.use_adaptive:
            x_attn, _ = self.attention(x, x, x, mask)
        else:
            x_attn, _ = self.attention(x, x, x, mask)
        
        # Apply gated attention
        x_attn = self.gated_attention(x_attn)
        x_attn = self.dropout(x_attn)
        
        # Residual connection for attention with stochastic depth
        x = self.norm2(x + self.stochastic_depth(x_attn))
        
        # Output projection
        x_out = self.output_projection(x)
        x_out = self.dropout(x_out)
        
        # Final residual connection with stochastic depth
        x = self.norm3(x + self.stochastic_depth(x_out))
        
        return x


class Chomp1d(nn.Module):
    """
    Chomp1d layer for removing extra padding from convolutions.
    """
    
    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Remove extra padding from the end of the sequence.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor with padding removed
        """
        return x[:, :, :-self.chomp_size].contiguous()


class MultiHeadAttention(nn.Module):
    """
    Standard multi-head attention mechanism.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-head attention.
        """
        batch_size = query.size(0)
        
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        output = output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        output = self.w_o(output)
        
        return output, attention_weights 