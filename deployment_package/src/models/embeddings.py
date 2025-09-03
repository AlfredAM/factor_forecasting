"""
Embedding layers for factor forecasting models
Includes positional encoding and stock embeddings
"""
import torch
import torch.nn as nn
import math
import logging

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer model to provide sequence position information.
    Implements the sinusoidal positional encoding as described in "Attention Is All You Need".
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Dimension of the model embeddings
            max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:x.size(0), :]


class StockEmbedding(nn.Module):
    """
    Stock embedding layer to learn stock-specific representations.
    Maps stock IDs to learnable embeddings.
    """
    
    def __init__(self, num_stocks: int, embedding_dim: int):
        """
        Initialize stock embedding layer.
        
        Args:
            num_stocks: Number of unique stocks
            embedding_dim: Dimension of stock embeddings
        """
        super(StockEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_stocks, embedding_dim)
        self.embedding_dim = embedding_dim
        
        # Initialize embeddings with small random values
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)
    
    def forward(self, stock_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert stock IDs to embeddings.
        
        Args:
            stock_ids: Stock ID tensor of shape (batch_size, seq_len)
            
        Returns:
            Stock embeddings of shape (batch_size, seq_len, embedding_dim)
        """
        return self.embedding(stock_ids) 