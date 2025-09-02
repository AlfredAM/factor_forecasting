#!/usr/bin/env python3
"""
Quantitative Finance Metrics and Performance Evaluation
Professional metrics for factor model evaluation in quantitative finance
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class QuantitativeMetrics:
    """Container for quantitative finance metrics"""
    
    # Basic prediction metrics
    correlation: float
    rank_ic: float
    mse: float
    mae: float
    r_squared: float
    
    # IC statistics
    ic_mean: float
    ic_std: float
    ic_ir: float  # Information Ratio
    ic_t_stat: float
    ic_p_value: float
    
    # Risk metrics
    volatility: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Trading metrics
    hit_rate: float
    directional_accuracy: float
    turnover: float
    transaction_costs: float
    
    # Cross-sectional metrics
    cross_sectional_ic: float
    long_short_spread: float
    quintile_spread: float
    
    # Stability metrics
    ic_stability: float
    regime_consistency: float
    decay_rate: float


class QuantitativePerformanceAnalyzer:
    """
    Professional performance analyzer for quantitative finance models
    Implements industry standard metrics used in factor model evaluation
    """
    
    def __init__(self, 
                 target_names: List[str] = None,
                 benchmark_return: float = 0.0,
                 risk_free_rate: float = 0.02,
                 transaction_cost: float = 0.001):
        """
        Initialize performance analyzer
        
        Args:
            target_names: List of target column names
            benchmark_return: Benchmark return for excess return calculation
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            transaction_cost: Transaction cost rate
        """
        self.target_names = target_names or ['intra30m', 'nextT1d', 'ema1d']
        self.benchmark_return = benchmark_return
        self.risk_free_rate = risk_free_rate
        self.transaction_cost = transaction_cost
        
        # Storage for analysis
        self.predictions = {}
        self.targets = {}
        self.dates = []
        self.stock_ids = []
        self.weights = []
        
    def add_predictions(self, 
                       predictions: Dict[str, np.ndarray],
                       targets: Dict[str, np.ndarray],
                       dates: Optional[List[str]] = None,
                       stock_ids: Optional[List[str]] = None,
                       weights: Optional[np.ndarray] = None):
        """
        Add prediction data for analysis
        
        Args:
            predictions: Dictionary of predictions by target
            targets: Dictionary of actual values by target
            dates: List of dates for time series analysis
            stock_ids: List of stock identifiers
            weights: Sample weights (e.g., market cap weights)
        """
        for target in self.target_names:
            if target in predictions and target in targets:
                if target not in self.predictions:
                    self.predictions[target] = []
                    self.targets[target] = []
                
                self.predictions[target].extend(predictions[target])
                self.targets[target].extend(targets[target])
        
        if dates:
            self.dates.extend(dates)
        if stock_ids:
            self.stock_ids.extend(stock_ids)
        if weights is not None:
            self.weights.extend(weights)
    
    def compute_comprehensive_metrics(self) -> Dict[str, QuantitativeMetrics]:
        """
        Compute comprehensive quantitative finance metrics
        
        Returns:
            Dictionary of metrics for each target
        """
        results = {}
        
        for target in self.target_names:
            if target in self.predictions and len(self.predictions[target]) > 0:
                pred = np.array(self.predictions[target])
                true = np.array(self.targets[target])
                
                # Remove NaN values
                valid_mask = ~(np.isnan(pred) | np.isnan(true))
                if valid_mask.sum() < 10:
                    logger.warning(f"Insufficient valid data for {target}")
                    continue
                
                pred_clean = pred[valid_mask]
                true_clean = true[valid_mask]
                
                # Get corresponding weights if available
                if self.weights:
                    weights_clean = np.array(self.weights)[valid_mask]
                else:
                    weights_clean = np.ones_like(pred_clean)
                
                # Get corresponding dates and stock_ids if available
                dates_clean = None
                stock_ids_clean = None
                if self.dates:
                    dates_clean = np.array(self.dates)[valid_mask]
                if self.stock_ids:
                    stock_ids_clean = np.array(self.stock_ids)[valid_mask]
                
                # Compute all metrics
                metrics = self._compute_target_metrics(
                    pred_clean, true_clean, weights_clean,
                    dates_clean, stock_ids_clean, target
                )
                
                results[target] = metrics
        
        return results
    
    def _compute_target_metrics(self, 
                               pred: np.ndarray, 
                               true: np.ndarray,
                               weights: np.ndarray,
                               dates: Optional[np.ndarray],
                               stock_ids: Optional[np.ndarray],
                               target: str) -> QuantitativeMetrics:
        """Compute comprehensive metrics for a single target"""
        
        # Basic prediction metrics
        correlation = self._compute_weighted_correlation(pred, true, weights)
        rank_ic = self._compute_rank_ic(pred, true, weights)
        mse = mean_squared_error(true, pred, sample_weight=weights)
        mae = mean_absolute_error(true, pred, sample_weight=weights)
        r_squared = r2_score(true, pred, sample_weight=weights)
        
        # IC statistics
        ic_stats = self._compute_ic_statistics(pred, true, weights, dates, stock_ids)
        
        # Risk metrics
        risk_metrics = self._compute_risk_metrics(pred, true, weights)
        
        # Trading metrics
        trading_metrics = self._compute_trading_metrics(pred, true, weights)
        
        # Cross-sectional metrics
        cross_metrics = self._compute_cross_sectional_metrics(pred, true, weights, dates, stock_ids)
        
        # Stability metrics
        stability_metrics = self._compute_stability_metrics(pred, true, weights, dates)
        
        return QuantitativeMetrics(
            # Basic metrics
            correlation=correlation,
            rank_ic=rank_ic,
            mse=mse,
            mae=mae,
            r_squared=r_squared,
            
            # IC statistics
            ic_mean=ic_stats['ic_mean'],
            ic_std=ic_stats['ic_std'],
            ic_ir=ic_stats['ic_ir'],
            ic_t_stat=ic_stats['ic_t_stat'],
            ic_p_value=ic_stats['ic_p_value'],
            
            # Risk metrics
            volatility=risk_metrics['volatility'],
            max_drawdown=risk_metrics['max_drawdown'],
            sharpe_ratio=risk_metrics['sharpe_ratio'],
            sortino_ratio=risk_metrics['sortino_ratio'],
            calmar_ratio=risk_metrics['calmar_ratio'],
            
            # Trading metrics
            hit_rate=trading_metrics['hit_rate'],
            directional_accuracy=trading_metrics['directional_accuracy'],
            turnover=trading_metrics['turnover'],
            transaction_costs=trading_metrics['transaction_costs'],
            
            # Cross-sectional metrics
            cross_sectional_ic=cross_metrics['cross_sectional_ic'],
            long_short_spread=cross_metrics['long_short_spread'],
            quintile_spread=cross_metrics['quintile_spread'],
            
            # Stability metrics
            ic_stability=stability_metrics['ic_stability'],
            regime_consistency=stability_metrics['regime_consistency'],
            decay_rate=stability_metrics['decay_rate']
        )
    
    def _compute_weighted_correlation(self, pred: np.ndarray, true: np.ndarray, 
                                    weights: np.ndarray) -> float:
        """Compute weighted Pearson correlation"""
        if len(pred) < 2:
            return 0.0
        
        # Weighted correlation formula
        w_sum = np.sum(weights)
        pred_mean = np.sum(pred * weights) / w_sum
        true_mean = np.sum(true * weights) / w_sum
        
        pred_centered = pred - pred_mean
        true_centered = true - true_mean
        
        numerator = np.sum(weights * pred_centered * true_centered)
        pred_var = np.sum(weights * pred_centered ** 2)
        true_var = np.sum(weights * true_centered ** 2)
        
        if pred_var <= 0 or true_var <= 0:
            return 0.0
        
        correlation = numerator / np.sqrt(pred_var * true_var)
        return correlation
    
    def _compute_rank_ic(self, pred: np.ndarray, true: np.ndarray, 
                        weights: np.ndarray) -> float:
        """Compute weighted rank correlation (Spearman-like)"""
        if len(pred) < 2:
            return 0.0
        
        try:
            # Convert to ranks
            pred_ranks = stats.rankdata(pred)
            true_ranks = stats.rankdata(true)
            
            # Compute weighted correlation of ranks
            rank_ic = self._compute_weighted_correlation(pred_ranks, true_ranks, weights)
            return rank_ic
        except:
            return 0.0
    
    def _compute_ic_statistics(self, pred: np.ndarray, true: np.ndarray,
                              weights: np.ndarray, dates: Optional[np.ndarray],
                              stock_ids: Optional[np.ndarray]) -> Dict[str, float]:
        """Compute IC statistics including time series properties"""
        
        if dates is None or stock_ids is None:
            # Fallback to overall IC
            ic_values = [self._compute_weighted_correlation(pred, true, weights)]
        else:
            # Compute daily IC values
            ic_values = []
            unique_dates = np.unique(dates)
            
            for date in unique_dates:
                date_mask = dates == date
                if date_mask.sum() > 5:  # Minimum cross-section size
                    date_pred = pred[date_mask]
                    date_true = true[date_mask]
                    date_weights = weights[date_mask]
                    
                    daily_ic = self._compute_weighted_correlation(date_pred, date_true, date_weights)
                    if not np.isnan(daily_ic):
                        ic_values.append(daily_ic)
        
        if len(ic_values) == 0:
            return {
                'ic_mean': 0.0, 'ic_std': 0.0, 'ic_ir': 0.0,
                'ic_t_stat': 0.0, 'ic_p_value': 1.0
            }
        
        ic_values = np.array(ic_values)
        ic_mean = np.mean(ic_values)
        ic_std = np.std(ic_values)
        ic_ir = ic_mean / (ic_std + 1e-8)  # Information Ratio
        
        # T-test for IC significance
        if len(ic_values) > 1:
            ic_t_stat, ic_p_value = stats.ttest_1samp(ic_values, 0.0)
        else:
            ic_t_stat, ic_p_value = 0.0, 1.0
        
        return {
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'ic_ir': ic_ir,
            'ic_t_stat': ic_t_stat,
            'ic_p_value': ic_p_value
        }
    
    def _compute_risk_metrics(self, pred: np.ndarray, true: np.ndarray,
                             weights: np.ndarray) -> Dict[str, float]:
        """Compute risk-adjusted performance metrics"""
        
        # Simulate portfolio returns based on predictions
        # Simple long-short strategy
        normalized_pred = pred / (np.abs(pred).sum() + 1e-8)
        portfolio_returns = normalized_pred * true
        
        if len(portfolio_returns) < 2:
            return {
                'volatility': 0.0, 'max_drawdown': 0.0, 'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0, 'calmar_ratio': 0.0
            }
        
        # Annualized volatility (assuming daily returns)
        volatility = np.std(portfolio_returns) * np.sqrt(252)
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.abs(np.min(drawdowns))
        
        # Sharpe ratio
        excess_returns = portfolio_returns - self.risk_free_rate / 252
        sharpe_ratio = np.mean(excess_returns) / (np.std(portfolio_returns) + 1e-8) * np.sqrt(252)
        
        # Sortino ratio (downside deviation)
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns) * np.sqrt(252)
            sortino_ratio = np.mean(excess_returns) * np.sqrt(252) / (downside_std + 1e-8)
        else:
            sortino_ratio = sharpe_ratio
        
        # Calmar ratio
        annual_return = np.mean(portfolio_returns) * 252
        calmar_ratio = annual_return / (max_drawdown + 1e-8)
        
        return {
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio
        }
    
    def _compute_trading_metrics(self, pred: np.ndarray, true: np.ndarray,
                                weights: np.ndarray) -> Dict[str, float]:
        """Compute trading-related performance metrics"""
        
        # Hit rate (probability of positive returns when predicted positive)
        positive_pred_mask = pred > 0
        if positive_pred_mask.sum() > 0:
            hit_rate = np.mean(true[positive_pred_mask] > 0)
        else:
            hit_rate = 0.0
        
        # Directional accuracy
        pred_direction = np.sign(pred)
        true_direction = np.sign(true)
        directional_accuracy = np.mean(pred_direction == true_direction)
        
        # Turnover (portfolio change rate)
        if len(pred) > 1:
            normalized_pred = pred / (np.abs(pred).sum() + 1e-8)
            position_changes = np.abs(normalized_pred[1:] - normalized_pred[:-1])
            turnover = np.mean(position_changes)
        else:
            turnover = 0.0
        
        # Transaction costs
        transaction_costs = turnover * self.transaction_cost
        
        return {
            'hit_rate': hit_rate,
            'directional_accuracy': directional_accuracy,
            'turnover': turnover,
            'transaction_costs': transaction_costs
        }
    
    def _compute_cross_sectional_metrics(self, pred: np.ndarray, true: np.ndarray,
                                       weights: np.ndarray, dates: Optional[np.ndarray],
                                       stock_ids: Optional[np.ndarray]) -> Dict[str, float]:
        """Compute cross-sectional analysis metrics"""
        
        # Cross-sectional IC (if we have date information)
        if dates is not None and stock_ids is not None:
            cross_sectional_ic = self._compute_weighted_correlation(pred, true, weights)
        else:
            cross_sectional_ic = self._compute_weighted_correlation(pred, true, weights)
        
        # Quintile analysis
        n_quintiles = 5
        quintile_size = len(pred) // n_quintiles
        
        if quintile_size > 0:
            # Sort by predictions and compute quintile returns
            sorted_indices = np.argsort(pred)
            quintile_returns = []
            
            for q in range(n_quintiles):
                start_idx = q * quintile_size
                end_idx = start_idx + quintile_size if q < n_quintiles - 1 else len(pred)
                quintile_indices = sorted_indices[start_idx:end_idx]
                
                quintile_return = np.mean(true[quintile_indices])
                quintile_returns.append(quintile_return)
            
            # Long-short spread (top quintile - bottom quintile)
            long_short_spread = quintile_returns[-1] - quintile_returns[0]
            
            # Quintile spread (measure of monotonicity)
            quintile_spread = np.std(quintile_returns)
        else:
            long_short_spread = 0.0
            quintile_spread = 0.0
        
        return {
            'cross_sectional_ic': cross_sectional_ic,
            'long_short_spread': long_short_spread,
            'quintile_spread': quintile_spread
        }
    
    def _compute_stability_metrics(self, pred: np.ndarray, true: np.ndarray,
                                  weights: np.ndarray, dates: Optional[np.ndarray]) -> Dict[str, float]:
        """Compute stability and consistency metrics"""
        
        if dates is None or len(np.unique(dates)) < 3:
            return {
                'ic_stability': 0.0,
                'regime_consistency': 0.0,
                'decay_rate': 0.0
            }
        
        # Compute rolling IC stability
        unique_dates = sorted(np.unique(dates))
        rolling_ics = []
        window_size = min(20, len(unique_dates) // 3)
        
        for i in range(window_size, len(unique_dates)):
            window_dates = unique_dates[i-window_size:i]
            window_mask = np.isin(dates, window_dates)
            
            if window_mask.sum() > 10:
                window_pred = pred[window_mask]
                window_true = true[window_mask]
                window_weights = weights[window_mask]
                
                window_ic = self._compute_weighted_correlation(window_pred, window_true, window_weights)
                if not np.isnan(window_ic):
                    rolling_ics.append(window_ic)
        
        if len(rolling_ics) > 1:
            ic_stability = 1.0 - (np.std(rolling_ics) / (np.abs(np.mean(rolling_ics)) + 1e-8))
            ic_stability = max(0.0, ic_stability)  # Ensure non-negative
        else:
            ic_stability = 0.0
        
        # Regime consistency (correlation stability across different periods)
        if len(rolling_ics) > 5:
            # Split into two halves and compare
            mid_point = len(rolling_ics) // 2
            first_half = rolling_ics[:mid_point]
            second_half = rolling_ics[mid_point:]
            
            if len(first_half) > 1 and len(second_half) > 1:
                regime_consistency = abs(np.corrcoef(
                    range(len(first_half)), first_half)[0, 1] - 
                    np.corrcoef(range(len(second_half)), second_half)[0, 1])
                regime_consistency = 1.0 - regime_consistency  # Convert to consistency measure
            else:
                regime_consistency = 0.0
        else:
            regime_consistency = 0.0
        
        # Decay rate (how quickly IC deteriorates)
        if len(rolling_ics) > 10:
            time_series = np.arange(len(rolling_ics))
            correlation_with_time = np.corrcoef(time_series, rolling_ics)[0, 1]
            decay_rate = -correlation_with_time  # Negative correlation indicates decay
            decay_rate = max(0.0, decay_rate)  # Ensure non-negative
        else:
            decay_rate = 0.0
        
        return {
            'ic_stability': ic_stability,
            'regime_consistency': regime_consistency,
            'decay_rate': decay_rate
        }
    
    def generate_performance_report(self, metrics: Dict[str, QuantitativeMetrics]) -> str:
        """Generate a comprehensive performance report"""
        
        report = []
        report.append("=" * 80)
        report.append("QUANTITATIVE FINANCE PERFORMANCE REPORT")
        report.append("=" * 80)
        
        for target, metric in metrics.items():
            report.append(f"\n{target.upper()} TARGET ANALYSIS")
            report.append("-" * 50)
            
            # Basic metrics
            report.append(f"Correlation (IC):        {metric.correlation:.4f}")
            report.append(f"Rank IC:                 {metric.rank_ic:.4f}")
            report.append(f"MSE:                     {metric.mse:.6f}")
            report.append(f"MAE:                     {metric.mae:.6f}")
            report.append(f"R²:                      {metric.r_squared:.4f}")
            
            # IC statistics
            report.append(f"\nIC Statistics:")
            report.append(f"  Mean IC:               {metric.ic_mean:.4f}")
            report.append(f"  IC Std:                {metric.ic_std:.4f}")
            report.append(f"  Information Ratio:     {metric.ic_ir:.4f}")
            report.append(f"  IC T-stat:             {metric.ic_t_stat:.4f}")
            report.append(f"  IC P-value:            {metric.ic_p_value:.4f}")
            
            # Risk metrics
            report.append(f"\nRisk Metrics:")
            report.append(f"  Volatility:            {metric.volatility:.4f}")
            report.append(f"  Max Drawdown:          {metric.max_drawdown:.4f}")
            report.append(f"  Sharpe Ratio:          {metric.sharpe_ratio:.4f}")
            report.append(f"  Sortino Ratio:         {metric.sortino_ratio:.4f}")
            report.append(f"  Calmar Ratio:          {metric.calmar_ratio:.4f}")
            
            # Trading metrics
            report.append(f"\nTrading Metrics:")
            report.append(f"  Hit Rate:              {metric.hit_rate:.4f}")
            report.append(f"  Directional Accuracy:  {metric.directional_accuracy:.4f}")
            report.append(f"  Turnover:              {metric.turnover:.4f}")
            report.append(f"  Transaction Costs:     {metric.transaction_costs:.4f}")
            
            # Cross-sectional metrics
            report.append(f"\nCross-sectional Metrics:")
            report.append(f"  Cross-sectional IC:    {metric.cross_sectional_ic:.4f}")
            report.append(f"  Long-Short Spread:     {metric.long_short_spread:.4f}")
            report.append(f"  Quintile Spread:       {metric.quintile_spread:.4f}")
            
            # Stability metrics
            report.append(f"\nStability Metrics:")
            report.append(f"  IC Stability:          {metric.ic_stability:.4f}")
            report.append(f"  Regime Consistency:    {metric.regime_consistency:.4f}")
            report.append(f"  Decay Rate:            {metric.decay_rate:.4f}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)


if __name__ == "__main__":
    # Test the quantitative metrics analyzer
    np.random.seed(42)
    
    # Generate test data
    n_samples = 1000
    n_dates = 50
    n_stocks = 20
    
    dates = [f"2023-01-{i+1:02d}" for i in range(n_dates)] * (n_samples // n_dates)
    dates = dates[:n_samples]
    
    stock_ids = [f"STOCK_{i}" for i in range(n_stocks)] * (n_samples // n_stocks)
    stock_ids = stock_ids[:n_samples]
    
    # Generate correlated predictions and targets
    true_factor = np.random.randn(n_samples)
    noise = np.random.randn(n_samples) * 0.5
    
    predictions = {
        'intra30m': true_factor * 0.3 + noise * 0.02,
        'nextT1d': true_factor * 0.2 + noise * 0.015,
        'ema1d': true_factor * 0.15 + noise * 0.01
    }
    
    targets = {
        'intra30m': true_factor * 0.25 + np.random.randn(n_samples) * 0.025,
        'nextT1d': true_factor * 0.18 + np.random.randn(n_samples) * 0.018,
        'ema1d': true_factor * 0.12 + np.random.randn(n_samples) * 0.012
    }
    
    weights = np.random.uniform(0.5, 2.0, n_samples)
    
    # Test analyzer
    analyzer = QuantitativePerformanceAnalyzer()
    analyzer.add_predictions(predictions, targets, dates, stock_ids, weights)
    
    metrics = analyzer.compute_comprehensive_metrics()
    report = analyzer.generate_performance_report(metrics)
    
    print(report)
    print("\n Quantitative metrics analyzer test completed successfully!")
