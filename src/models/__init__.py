"""
Factor Forecasting Models Module
Contains Transformer models, attention mechanisms, embeddings, etc.
"""

from .transformer import FactorTransformerModel, FactorForecastingModel
from .attention import MultiHeadAttention, AttentionBlock
from .embeddings import PositionalEncoding, StockEmbedding
from .model_factory import create_model, load_model, save_model

# Advanced models
from .advanced_attention import (
    RelativePositionalEncoding,
    GatedLinearUnit,
    MultiScaleAttention,
    AdaptiveAttention,
    StochasticDepth,
    AdvancedTCNAttentionBlock
)
from .advanced_tcn_attention import (
    AdvancedFactorTCNAttention,
    AdvancedFactorForecastingTCNAttentionModel,
    create_advanced_model
)

__all__ = [
    'FactorTransformerModel',
    'FactorForecastingModel', 
    'MultiHeadAttention',
    'AttentionBlock',
    'PositionalEncoding',
    'StockEmbedding',
    'create_model',
    'load_model',
    'save_model',
    # Advanced models
    'RelativePositionalEncoding',
    'GatedLinearUnit',
    'MultiScaleAttention',
    'AdaptiveAttention',
    'StochasticDepth',
    'AdvancedTCNAttentionBlock',
    'AdvancedFactorTCNAttention',
    'AdvancedFactorForecastingTCNAttentionModel',
    'create_advanced_model'
]
