#!/usr/bin/env python3
"""
Quantitative Finance Optimized Loss Functions
Designed specifically for financial time series prediction with risk management
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class QuantitativeCorrelationLoss(nn.Module):
    """
    Advanced correlation loss function optimized for quantitative finance
    Includes risk-adjusted objectives and regime-aware optimization
    """
    
    def __init__(self, 
                 correlation_weight: float = 1.0,
                 mse_weight: float = 0.1,
                 rank_weight: float = 0.2,
                 risk_penalty_weight: float = 0.1,
                 target_correlations: List[float] = None,
                 max_leverage: float = 2.0,
                 transaction_cost: float = 0.001):
        super().__init__()
        
        self.correlation_weight = correlation_weight
        self.mse_weight = mse_weight
        self.rank_weight = rank_weight
        self.risk_penalty_weight = risk_penalty_weight
        self.max_leverage = max_leverage
        self.transaction_cost = transaction_cost
        
        # Target correlations for different time horizons
        # intra30m: short-term (higher volatility tolerance)
        # nextT1d: medium-term (balanced)
        # ema1d: long-term (lower volatility preference)
        self.target_correlations = target_correlations or [0.08, 0.05, 0.03]
        
        # Risk adjustment factors for different targets
        self.risk_factors = [1.2, 1.0, 0.8]  # Higher penalty for short-term volatility
        
        self.mse_loss = nn.MSELoss()
        self.huber_loss = nn.HuberLoss(delta=0.1)  # Robust to outliers
        
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor],
                weights: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute quantitative finance optimized loss
        
        Args:
            predictions: Model predictions for each target
            targets: Ground truth targets
            weights: Sample weights (e.g., market cap weights)
            
        Returns:
            Dictionary containing individual loss components and total loss
        """
        target_names = ['intra30m', 'nextT1d', 'ema1d']
        
        # Initialize loss components
        mse_losses = {}
        correlation_losses = {}
        rank_losses = {}
        risk_penalties = {}
        
        total_loss = 0.0
        
        for i, target_name in enumerate(target_names):
            if target_name in predictions and target_name in targets:
                pred = predictions[target_name]
                target = targets[target_name]
                
                # Ensure compatible shapes
                if pred.shape != target.shape:
                    pred = pred.view_as(target)
                
                # Apply sample weights if provided
                if weights is not None:
                    sample_weights = weights.view_as(pred)
                else:
                    sample_weights = torch.ones_like(pred)
                
                # Remove invalid values
                valid_mask = ~(torch.isnan(pred) | torch.isnan(target) | torch.isnan(sample_weights))
                
                if valid_mask.sum() > 10:  # Minimum samples for reliable statistics
                    pred_clean = pred[valid_mask]
                    target_clean = target[valid_mask]
                    weights_clean = sample_weights[valid_mask]
                    
                    # 1. MSE Loss with Huber robustness
                    mse_loss = self._compute_robust_mse(pred_clean, target_clean, weights_clean)
                    mse_losses[target_name] = mse_loss
                    
                    # 2. Correlation Loss (IC optimization)
                    corr_loss = self._compute_correlation_loss(pred_clean, target_clean, weights_clean, i)
                    correlation_losses[target_name] = corr_loss
                    
                    # 3. Rank Correlation Loss (RankIC optimization)
                    rank_loss = self._compute_rank_loss(pred_clean, target_clean, weights_clean)
                    rank_losses[target_name] = rank_loss
                    
                    # 4. Risk Penalty (volatility and extreme values)
                    risk_penalty = self._compute_risk_penalty(pred_clean, target_clean, i)
                    risk_penalties[target_name] = risk_penalty
                    
                    # Combine losses with target-specific weights
                    target_loss = (
                        self.mse_weight * mse_loss +
                        self.correlation_weight * corr_loss +
                        self.rank_weight * rank_loss +
                        self.risk_penalty_weight * self.risk_factors[i] * risk_penalty
                    )
                    
                    total_loss += target_loss
                    
                else:
                    # Handle insufficient data gracefully
                    logger.warning(f"Insufficient valid data for {target_name}: {valid_mask.sum()} samples")
                    mse_losses[target_name] = torch.tensor(0.0)
                    correlation_losses[target_name] = torch.tensor(0.0)
                    rank_losses[target_name] = torch.tensor(0.0)
                    risk_penalties[target_name] = torch.tensor(0.0)
        
        # Return detailed loss breakdown for monitoring
        return {
            'total_loss': total_loss,
            'mse_losses': mse_losses,
            'correlation_losses': correlation_losses,
            'rank_losses': rank_losses,
            'risk_penalties': risk_penalties
        }
    
    def _compute_robust_mse(self, pred: torch.Tensor, target: torch.Tensor, 
                           weights: torch.Tensor) -> torch.Tensor:
        """Compute weighted robust MSE using Huber loss"""
        # Use Huber loss for robustness to outliers
        huber_loss = self.huber_loss(pred, target)
        
        # Apply weights
        weighted_loss = huber_loss * weights
        return weighted_loss.mean()
    
    def _compute_correlation_loss(self, pred: torch.Tensor, target: torch.Tensor, 
                                weights: torch.Tensor, target_idx: int) -> torch.Tensor:
        """Compute weighted correlation loss with target IC optimization"""
        # Compute weighted correlation
        pred_weighted = pred * torch.sqrt(weights)
        target_weighted = target * torch.sqrt(weights)
        
        # Compute correlation
        pred_mean = pred_weighted.mean()
        target_mean = target_weighted.mean()
        
        pred_centered = pred_weighted - pred_mean
        target_centered = target_weighted - target_mean
        
        correlation = (pred_centered * target_centered).mean() / (
            pred_centered.std() * target_centered.std() + 1e-8
        )
        
        # Target correlation for this specific target
        target_corr = self.target_correlations[target_idx]
        
        # Penalize deviation from target correlation
        # Use L1 loss for correlation deviation
        correlation_loss = torch.abs(correlation - target_corr)
        
        return correlation_loss
    
    def _compute_rank_loss(self, pred: torch.Tensor, target: torch.Tensor, 
                          weights: torch.Tensor) -> torch.Tensor:
        """Compute weighted rank correlation loss (Spearman-like)"""
        # Convert to ranks
        pred_ranks = torch.argsort(torch.argsort(pred)).float()
        target_ranks = torch.argsort(torch.argsort(target)).float()
        
        # Apply weights to ranks
        pred_ranks_weighted = pred_ranks * torch.sqrt(weights)
        target_ranks_weighted = target_ranks * torch.sqrt(weights)
        
        # Compute rank correlation
        pred_ranks_mean = pred_ranks_weighted.mean()
        target_ranks_mean = target_ranks_weighted.mean()
        
        pred_ranks_centered = pred_ranks_weighted - pred_ranks_mean
        target_ranks_centered = target_ranks_weighted - target_ranks_mean
        
        rank_correlation = (pred_ranks_centered * target_ranks_centered).mean() / (
            pred_ranks_centered.std() * target_ranks_centered.std() + 1e-8
        )
        
        # Maximize rank correlation (minimize negative)
        return -rank_correlation
    
    def _compute_risk_penalty(self, pred: torch.Tensor, target: torch.Tensor, 
                             target_idx: int) -> torch.Tensor:
        """Compute risk penalty for extreme predictions and high volatility"""
        # 1. Volatility penalty - penalize high prediction volatility
        pred_volatility = pred.std()
        volatility_penalty = torch.clamp(pred_volatility - 0.1, min=0.0) ** 2
        
        # 2. Extreme value penalty - penalize predictions beyond reasonable bounds
        # Financial returns typically within [-20%, +20%] for daily returns
        extreme_bound = 0.2
        extreme_penalty = torch.mean(torch.clamp(torch.abs(pred) - extreme_bound, min=0.0) ** 2)
        
        # 3. Leverage penalty - prevent over-concentration
        # Compute effective leverage from predictions
        pred_abs_sum = torch.abs(pred).sum() + 1e-8
        leverage = torch.abs(pred).max() / (pred_abs_sum / len(pred))
        leverage_penalty = torch.clamp(leverage - self.max_leverage, min=0.0) ** 2
        
        # 4. Transaction cost penalty - penalize high turnover
        if len(pred) > 1:
            # Approximate turnover as prediction changes
            turnover = torch.mean(torch.abs(pred[1:] - pred[:-1]))
            transaction_penalty = turnover * self.transaction_cost
        else:
            transaction_penalty = torch.tensor(0.0)
        
        # Combine risk penalties
        total_risk_penalty = (volatility_penalty + extreme_penalty + 
                            leverage_penalty + transaction_penalty)
        
        return total_risk_penalty


class AdaptiveQuantitativeLoss(nn.Module):
    """
    Adaptive loss function that adjusts based on market regime and volatility
    """
    
    def __init__(self, base_loss: QuantitativeCorrelationLoss,
                 volatility_window: int = 20,
                 regime_sensitivity: float = 0.1):
        super().__init__()
        
        self.base_loss = base_loss
        self.volatility_window = volatility_window
        self.regime_sensitivity = regime_sensitivity
        
        # Track historical volatilities for regime detection
        self.volatility_history = []
        
    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                weights: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute adaptive loss with regime awareness
        """
        # Compute base loss
        loss_dict = self.base_loss(predictions, targets, weights)
        
        # Detect market regime based on target volatility
        regime_factor = self._detect_market_regime(targets)
        
        # Adjust loss weights based on regime
        if regime_factor > 1.5:  # High volatility regime
            # Increase risk penalty weight
            adjusted_loss = (loss_dict['total_loss'] + 
                           0.5 * sum(loss_dict['risk_penalties'].values()))
        elif regime_factor < 0.7:  # Low volatility regime
            # Focus more on correlation optimization
            adjusted_loss = (loss_dict['total_loss'] + 
                           0.3 * sum(loss_dict['correlation_losses'].values()))
        else:  # Normal regime
            adjusted_loss = loss_dict['total_loss']
        
        # Update loss dictionary
        loss_dict['total_loss'] = adjusted_loss
        loss_dict['regime_factor'] = regime_factor
        
        return loss_dict
    
    def _detect_market_regime(self, targets: Dict[str, torch.Tensor]) -> float:
        """
        Detect market regime based on recent volatility
        
        Returns:
            Regime factor: >1.5 (high vol), 0.7-1.5 (normal), <0.7 (low vol)
        """
        # Compute current volatility from targets
        current_volatilities = []
        for target_name, target_values in targets.items():
            if len(target_values) > 5:
                vol = target_values.std().item()
                current_volatilities.append(vol)
        
        if current_volatilities:
            current_vol = np.mean(current_volatilities)
            
            # Update volatility history
            self.volatility_history.append(current_vol)
            if len(self.volatility_history) > self.volatility_window:
                self.volatility_history.pop(0)
            
            # Compute regime factor
            if len(self.volatility_history) > 5:
                historical_vol = np.mean(self.volatility_history[:-1])
                regime_factor = current_vol / (historical_vol + 1e-8)
            else:
                regime_factor = 1.0
        else:
            regime_factor = 1.0
        
        return regime_factor


def create_quantitative_loss_function(config) -> nn.Module:
    """
    Factory function to create quantitative finance optimized loss function
    
    Args:
        config: Configuration object with loss parameters
        
    Returns:
        Configured loss function
    """
    # Extract loss configuration
    correlation_weight = getattr(config, 'correlation_weight', 1.0)
    mse_weight = getattr(config, 'mse_weight', 0.1)
    rank_weight = getattr(config, 'rank_correlation_weight', 0.2)
    risk_penalty_weight = getattr(config, 'risk_penalty_weight', 0.1)
    target_correlations = getattr(config, 'target_correlations', [0.08, 0.05, 0.03])
    max_leverage = getattr(config, 'max_leverage', 2.0)
    transaction_cost = getattr(config, 'transaction_cost', 0.001)
    
    # Create base loss
    base_loss = QuantitativeCorrelationLoss(
        correlation_weight=correlation_weight,
        mse_weight=mse_weight,
        rank_weight=rank_weight,
        risk_penalty_weight=risk_penalty_weight,
        target_correlations=target_correlations,
        max_leverage=max_leverage,
        transaction_cost=transaction_cost
    )
    
    # Wrap in adaptive loss if enabled
    use_adaptive = getattr(config, 'use_adaptive_loss', True)
    if use_adaptive:
        volatility_window = getattr(config, 'volatility_window', 20)
        regime_sensitivity = getattr(config, 'regime_sensitivity', 0.1)
        
        loss_function = AdaptiveQuantitativeLoss(
            base_loss=base_loss,
            volatility_window=volatility_window,
            regime_sensitivity=regime_sensitivity
        )
    else:
        loss_function = base_loss
    
    logger.info(f"Created quantitative loss function:")
    logger.info(f"  Correlation weight: {correlation_weight}")
    logger.info(f"  MSE weight: {mse_weight}")
    logger.info(f"  Rank weight: {rank_weight}")
    logger.info(f"  Risk penalty weight: {risk_penalty_weight}")
    logger.info(f"  Target correlations: {target_correlations}")
    logger.info(f"  Adaptive: {use_adaptive}")
    
    return loss_function


if __name__ == "__main__":
    # Test the quantitative loss function
    import torch
    
    # Create test data
    batch_size = 100
    predictions = {
        'intra30m': torch.randn(batch_size) * 0.02,
        'nextT1d': torch.randn(batch_size) * 0.015,
        'ema1d': torch.randn(batch_size) * 0.01
    }
    
    targets = {
        'intra30m': torch.randn(batch_size) * 0.025,
        'nextT1d': torch.randn(batch_size) * 0.018,
        'ema1d': torch.randn(batch_size) * 0.012
    }
    
    weights = torch.rand(batch_size)
    
    # Test base loss
    class TestConfig:
        correlation_weight = 1.0
        mse_weight = 0.1
        rank_correlation_weight = 0.2
        risk_penalty_weight = 0.1
        target_correlations = [0.08, 0.05, 0.03]
        use_adaptive_loss = True
    
    config = TestConfig()
    loss_fn = create_quantitative_loss_function(config)
    
    # Compute loss
    loss_dict = loss_fn(predictions, targets, weights)
    
    print("Test Results:")
    print(f"Total Loss: {loss_dict['total_loss']:.6f}")
    for target in ['intra30m', 'nextT1d', 'ema1d']:
        if target in loss_dict['mse_losses']:
            print(f"{target}:")
            print(f"  MSE Loss: {loss_dict['mse_losses'][target]:.6f}")
            print(f"  Correlation Loss: {loss_dict['correlation_losses'][target]:.6f}")
            print(f"  Rank Loss: {loss_dict['rank_losses'][target]:.6f}")
            print(f"  Risk Penalty: {loss_dict['risk_penalties'][target]:.6f}")
    
    if 'regime_factor' in loss_dict:
        print(f"Regime Factor: {loss_dict['regime_factor']:.3f}")
    
    print(" Quantitative loss function test completed successfully!")
