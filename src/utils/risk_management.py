#!/usr/bin/env python3
"""
Risk Management Module for Quantitative Finance
Implements comprehensive risk controls and monitoring for factor models
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import warnings

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Risk limits configuration"""
    max_leverage: float = 2.0
    max_concentration: float = 0.1  # Maximum weight per stock
    max_sector_exposure: float = 0.3  # Maximum sector exposure
    max_volatility: float = 0.2  # Maximum portfolio volatility
    max_drawdown: float = 0.15  # Maximum allowed drawdown
    min_ic: float = 0.01  # Minimum acceptable IC
    max_turnover: float = 2.0  # Maximum daily turnover
    vola_target: float = 0.15  # Target portfolio volatility


@dataclass
class RiskMetrics:
    """Risk metrics container"""
    leverage: float
    concentration: float
    sector_exposure: Dict[str, float]
    volatility: float
    var_95: float  # Value at Risk 95%
    var_99: float  # Value at Risk 99%
    cvar_95: float  # Conditional Value at Risk 95%
    max_drawdown: float
    sharpe_ratio: float
    tracking_error: float
    information_ratio: float
    beta: float
    correlation_breakdown: bool  # Whether correlation structure has broken down
    regime_change: bool  # Whether market regime has changed


class RiskManager:
    """
    Professional risk management system for quantitative factor models
    Implements real-time risk monitoring and portfolio constraints
    """
    
    def __init__(self, risk_limits: RiskLimits = None):
        """
        Initialize risk manager
        
        Args:
            risk_limits: Risk limits configuration
        """
        self.risk_limits = risk_limits or RiskLimits()
        
        # Risk monitoring state
        self.position_history = []
        self.return_history = []
        self.volatility_history = []
        self.ic_history = []
        self.drawdown_history = []
        
        # Market regime tracking
        self.regime_indicators = []
        self.correlation_matrix_history = []
        
        # Alert system
        self.active_alerts = []
        
    def check_position_limits(self, positions: np.ndarray, 
                            stock_metadata: Optional[Dict] = None) -> Dict[str, bool]:
        """
        Check position-based risk limits
        
        Args:
            positions: Array of stock positions (weights)
            stock_metadata: Optional metadata with sector information
            
        Returns:
            Dictionary of limit check results
        """
        checks = {}
        
        # Leverage check
        leverage = np.sum(np.abs(positions))
        checks['leverage_ok'] = leverage <= self.risk_limits.max_leverage
        
        # Concentration check
        max_position = np.max(np.abs(positions))
        checks['concentration_ok'] = max_position <= self.risk_limits.max_concentration
        
        # Sector exposure check
        if stock_metadata and 'sectors' in stock_metadata:
            sector_exposures = self._compute_sector_exposures(positions, stock_metadata['sectors'])
            max_sector_exposure = max(sector_exposures.values()) if sector_exposures else 0.0
            checks['sector_exposure_ok'] = max_sector_exposure <= self.risk_limits.max_sector_exposure
        else:
            checks['sector_exposure_ok'] = True
        
        # Long-short balance check
        long_exposure = np.sum(positions[positions > 0])
        short_exposure = np.abs(np.sum(positions[positions < 0]))
        balance_ratio = abs(long_exposure - short_exposure) / (long_exposure + short_exposure + 1e-8)
        checks['balance_ok'] = balance_ratio <= 0.2  # Allow max 20% imbalance
        
        return checks
    
    def check_performance_limits(self, returns: np.ndarray,
                               predictions: np.ndarray,
                               targets: np.ndarray) -> Dict[str, bool]:
        """
        Check performance-based risk limits
        
        Args:
            returns: Portfolio returns
            predictions: Model predictions
            targets: Actual targets
            
        Returns:
            Dictionary of performance check results
        """
        checks = {}
        
        if len(returns) == 0:
            return {'all_ok': True}
        
        # Volatility check
        if len(returns) > 5:
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            checks['volatility_ok'] = volatility <= self.risk_limits.max_volatility
        else:
            checks['volatility_ok'] = True
        
        # Drawdown check
        if len(returns) > 1:
            cumulative_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = np.abs(np.min(drawdowns))
            checks['drawdown_ok'] = max_drawdown <= self.risk_limits.max_drawdown
        else:
            checks['drawdown_ok'] = True
        
        # IC check
        if len(predictions) > 10 and len(targets) > 10:
            # Remove NaN values
            valid_mask = ~(np.isnan(predictions) | np.isnan(targets))
            if valid_mask.sum() > 10:
                pred_clean = predictions[valid_mask]
                target_clean = targets[valid_mask]
                ic = np.corrcoef(pred_clean, target_clean)[0, 1]
                checks['ic_ok'] = not np.isnan(ic) and ic >= self.risk_limits.min_ic
            else:
                checks['ic_ok'] = False
        else:
            checks['ic_ok'] = True
        
        return checks
    
    def compute_risk_metrics(self, positions: np.ndarray,
                           returns: np.ndarray,
                           benchmark_returns: Optional[np.ndarray] = None,
                           factor_exposures: Optional[np.ndarray] = None) -> RiskMetrics:
        """
        Compute comprehensive risk metrics
        
        Args:
            positions: Current portfolio positions
            returns: Historical portfolio returns
            benchmark_returns: Benchmark returns for tracking error
            factor_exposures: Factor exposures for risk attribution
            
        Returns:
            Comprehensive risk metrics
        """
        # Basic risk metrics
        leverage = np.sum(np.abs(positions))
        concentration = np.max(np.abs(positions))
        
        # Volatility and VaR
        if len(returns) > 5:
            volatility = np.std(returns) * np.sqrt(252)
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            
            # Conditional VaR (Expected Shortfall)
            cvar_95 = np.mean(returns[returns <= var_95])
        else:
            volatility = 0.0
            var_95 = 0.0
            var_99 = 0.0
            cvar_95 = 0.0
        
        # Drawdown analysis
        if len(returns) > 1:
            cumulative_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = np.abs(np.min(drawdowns))
        else:
            max_drawdown = 0.0
        
        # Performance metrics
        if len(returns) > 5:
            excess_returns = returns - 0.02/252  # Assume 2% risk-free rate
            sharpe_ratio = np.mean(excess_returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Tracking error and information ratio
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            active_returns = returns - benchmark_returns
            tracking_error = np.std(active_returns) * np.sqrt(252)
            information_ratio = np.mean(active_returns) / (np.std(active_returns) + 1e-8) * np.sqrt(252)
            
            # Beta calculation
            if len(returns) > 10:
                beta = np.cov(returns, benchmark_returns)[0, 1] / (np.var(benchmark_returns) + 1e-8)
            else:
                beta = 1.0
        else:
            tracking_error = 0.0
            information_ratio = 0.0
            beta = 1.0
        
        # Market regime detection
        correlation_breakdown = self._detect_correlation_breakdown(returns)
        regime_change = self._detect_regime_change(returns)
        
        return RiskMetrics(
            leverage=leverage,
            concentration=concentration,
            sector_exposure={},  # Would need sector data to populate
            volatility=volatility,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            tracking_error=tracking_error,
            information_ratio=information_ratio,
            beta=beta,
            correlation_breakdown=correlation_breakdown,
            regime_change=regime_change
        )
    
    def apply_risk_constraints(self, raw_positions: np.ndarray,
                             stock_metadata: Optional[Dict] = None,
                             previous_positions: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply risk constraints to raw model positions
        
        Args:
            raw_positions: Raw positions from model
            stock_metadata: Metadata for risk constraints
            previous_positions: Previous positions for turnover control
            
        Returns:
            Risk-adjusted positions
        """
        positions = raw_positions.copy()
        
        # 1. Leverage constraint
        current_leverage = np.sum(np.abs(positions))
        if current_leverage > self.risk_limits.max_leverage:
            positions = positions * (self.risk_limits.max_leverage / current_leverage)
            logger.info(f"Applied leverage constraint: {current_leverage:.3f} -> {self.risk_limits.max_leverage:.3f}")
        
        # 2. Position size constraint
        max_position = np.max(np.abs(positions))
        if max_position > self.risk_limits.max_concentration:
            # Scale down large positions
            excess_mask = np.abs(positions) > self.risk_limits.max_concentration
            positions[excess_mask] = np.sign(positions[excess_mask]) * self.risk_limits.max_concentration
            logger.info(f"Applied concentration constraint: max position {max_position:.3f} -> {self.risk_limits.max_concentration:.3f}")
        
        # 3. Sector exposure constraint
        if stock_metadata and 'sectors' in stock_metadata:
            positions = self._apply_sector_constraints(positions, stock_metadata['sectors'])
        
        # 4. Turnover constraint
        if previous_positions is not None:
            positions = self._apply_turnover_constraint(positions, previous_positions)
        
        # 5. Long-short balance
        positions = self._balance_long_short(positions)
        
        # 6. Volatility targeting
        if len(self.volatility_history) > 5:
            target_vol_ratio = self.risk_limits.vola_target / (np.mean(self.volatility_history[-20:]) + 1e-8)
            target_vol_ratio = np.clip(target_vol_ratio, 0.5, 2.0)  # Limit adjustment
            positions = positions * target_vol_ratio
        
        return positions
    
    def _compute_sector_exposures(self, positions: np.ndarray, 
                                 sectors: List[str]) -> Dict[str, float]:
        """Compute sector exposures"""
        sector_exposures = {}
        unique_sectors = set(sectors)
        
        for sector in unique_sectors:
            sector_mask = np.array([s == sector for s in sectors])
            sector_exposure = np.sum(np.abs(positions[sector_mask]))
            sector_exposures[sector] = sector_exposure
        
        return sector_exposures
    
    def _apply_sector_constraints(self, positions: np.ndarray,
                                 sectors: List[str]) -> np.ndarray:
        """Apply sector exposure constraints"""
        adjusted_positions = positions.copy()
        sector_exposures = self._compute_sector_exposures(positions, sectors)
        
        for sector, exposure in sector_exposures.items():
            if exposure > self.risk_limits.max_sector_exposure:
                # Scale down positions in this sector
                sector_mask = np.array([s == sector for s in sectors])
                scale_factor = self.risk_limits.max_sector_exposure / exposure
                adjusted_positions[sector_mask] *= scale_factor
                logger.info(f"Applied sector constraint for {sector}: {exposure:.3f} -> {self.risk_limits.max_sector_exposure:.3f}")
        
        return adjusted_positions
    
    def _apply_turnover_constraint(self, current_positions: np.ndarray,
                                  previous_positions: np.ndarray) -> np.ndarray:
        """Apply turnover constraint"""
        position_changes = np.abs(current_positions - previous_positions)
        turnover = np.sum(position_changes)
        
        if turnover > self.risk_limits.max_turnover:
            # Blend current and previous positions to reduce turnover
            blend_ratio = self.risk_limits.max_turnover / turnover
            adjusted_positions = (previous_positions * (1 - blend_ratio) + 
                                current_positions * blend_ratio)
            logger.info(f"Applied turnover constraint: {turnover:.3f} -> {self.risk_limits.max_turnover:.3f}")
            return adjusted_positions
        
        return current_positions
    
    def _balance_long_short(self, positions: np.ndarray) -> np.ndarray:
        """Balance long and short exposures"""
        long_exposure = np.sum(positions[positions > 0])
        short_exposure = np.abs(np.sum(positions[positions < 0]))
        
        # Target equal dollar exposure
        total_exposure = long_exposure + short_exposure
        target_exposure = total_exposure / 2
        
        if long_exposure > 0:
            long_scale = target_exposure / long_exposure
            positions[positions > 0] *= long_scale
        
        if short_exposure > 0:
            short_scale = target_exposure / short_exposure
            positions[positions < 0] *= short_scale
        
        return positions
    
    def _detect_correlation_breakdown(self, returns: np.ndarray) -> bool:
        """Detect if factor correlation structure has broken down"""
        if len(returns) < 50:
            return False
        
        # Compare recent correlation with historical
        recent_period = 20
        historical_period = len(returns) - recent_period
        
        if historical_period < 20:
            return False
        
        recent_vol = np.std(returns[-recent_period:])
        historical_vol = np.std(returns[:historical_period])
        
        # Flag if volatility has increased significantly
        vol_ratio = recent_vol / (historical_vol + 1e-8)
        
        return vol_ratio > 2.0  # Volatility doubled
    
    def _detect_regime_change(self, returns: np.ndarray) -> bool:
        """Detect market regime change"""
        if len(returns) < 40:
            return False
        
        # Use rolling correlation with lagged returns as regime indicator
        window = 20
        correlations = []
        
        for i in range(window, len(returns)-1):
            recent = returns[i-window:i]
            future = returns[i:i+window]
            
            if len(recent) == len(future) and len(recent) > 5:
                corr = np.corrcoef(recent, future)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
        
        if len(correlations) < 5:
            return False
        
        # Check if recent correlations are significantly different
        recent_corr = np.mean(correlations[-5:])
        historical_corr = np.mean(correlations[:-5])
        
        return abs(recent_corr - historical_corr) > 0.3
    
    def update_monitoring_state(self, positions: np.ndarray,
                               returns: np.ndarray,
                               predictions: np.ndarray,
                               targets: np.ndarray):
        """Update risk monitoring state"""
        
        # Update position history
        self.position_history.append(positions.copy())
        if len(self.position_history) > 100:  # Keep last 100 observations
            self.position_history.pop(0)
        
        # Update return history
        if len(returns) > 0:
            self.return_history.extend(returns)
            if len(self.return_history) > 252:  # Keep last year
                self.return_history = self.return_history[-252:]
            
            # Update volatility
            if len(self.return_history) > 5:
                current_vol = np.std(self.return_history[-20:]) * np.sqrt(252)
                self.volatility_history.append(current_vol)
                if len(self.volatility_history) > 50:
                    self.volatility_history.pop(0)
        
        # Update IC history
        if len(predictions) > 0 and len(targets) > 0:
            valid_mask = ~(np.isnan(predictions) | np.isnan(targets))
            if valid_mask.sum() > 10:
                pred_clean = predictions[valid_mask]
                target_clean = targets[valid_mask]
                ic = np.corrcoef(pred_clean, target_clean)[0, 1]
                if not np.isnan(ic):
                    self.ic_history.append(ic)
                    if len(self.ic_history) > 50:
                        self.ic_history.pop(0)
    
    def generate_risk_alerts(self, risk_metrics: RiskMetrics,
                           limit_checks: Dict[str, bool]) -> List[str]:
        """Generate risk alerts based on current state"""
        alerts = []
        
        # Leverage alerts
        if not limit_checks.get('leverage_ok', True):
            alerts.append(f"LEVERAGE EXCEEDED: {risk_metrics.leverage:.3f} > {self.risk_limits.max_leverage:.3f}")
        
        # Concentration alerts
        if not limit_checks.get('concentration_ok', True):
            alerts.append(f"CONCENTRATION EXCEEDED: {risk_metrics.concentration:.3f} > {self.risk_limits.max_concentration:.3f}")
        
        # Volatility alerts
        if risk_metrics.volatility > self.risk_limits.max_volatility:
            alerts.append(f"VOLATILITY HIGH: {risk_metrics.volatility:.3f} > {self.risk_limits.max_volatility:.3f}")
        
        # Drawdown alerts
        if risk_metrics.max_drawdown > self.risk_limits.max_drawdown:
            alerts.append(f"DRAWDOWN EXCEEDED: {risk_metrics.max_drawdown:.3f} > {self.risk_limits.max_drawdown:.3f}")
        
        # Performance alerts
        if not limit_checks.get('ic_ok', True):
            alerts.append(f"IC BELOW THRESHOLD: Current IC < {self.risk_limits.min_ic:.3f}")
        
        # Market regime alerts
        if risk_metrics.correlation_breakdown:
            alerts.append("CORRELATION BREAKDOWN DETECTED: Factor relationships may have changed")
        
        if risk_metrics.regime_change:
            alerts.append("REGIME CHANGE DETECTED: Market conditions may have shifted")
        
        # VaR alerts
        if risk_metrics.var_99 < -0.05:  # 5% daily loss
            alerts.append(f"HIGH VAR: 99% VaR = {risk_metrics.var_99:.3f}")
        
        return alerts
    
    def generate_risk_report(self, risk_metrics: RiskMetrics,
                           limit_checks: Dict[str, bool]) -> str:
        """Generate comprehensive risk report"""
        
        report = []
        report.append("=" * 60)
        report.append("RISK MANAGEMENT REPORT")
        report.append("=" * 60)
        
        # Position risk metrics
        report.append("\nPOSITION RISK METRICS:")
        report.append(f"Leverage:              {risk_metrics.leverage:.3f} / {self.risk_limits.max_leverage:.3f}")
        report.append(f"Concentration:         {risk_metrics.concentration:.3f} / {self.risk_limits.max_concentration:.3f}")
        
        # Performance risk metrics
        report.append("\nPERFORMANCE RISK METRICS:")
        report.append(f"Volatility:            {risk_metrics.volatility:.3f} / {self.risk_limits.max_volatility:.3f}")
        report.append(f"Max Drawdown:          {risk_metrics.max_drawdown:.3f} / {self.risk_limits.max_drawdown:.3f}")
        report.append(f"Sharpe Ratio:          {risk_metrics.sharpe_ratio:.3f}")
        report.append(f"Information Ratio:     {risk_metrics.information_ratio:.3f}")
        
        # VaR metrics
        report.append("\nVALUE AT RISK METRICS:")
        report.append(f"VaR 95%:               {risk_metrics.var_95:.4f}")
        report.append(f"VaR 99%:               {risk_metrics.var_99:.4f}")
        report.append(f"CVaR 95%:              {risk_metrics.cvar_95:.4f}")
        
        # Market risk indicators
        report.append("\nMARKET RISK INDICATORS:")
        report.append(f"Beta:                  {risk_metrics.beta:.3f}")
        report.append(f"Tracking Error:        {risk_metrics.tracking_error:.3f}")
        report.append(f"Correlation Breakdown: {'YES' if risk_metrics.correlation_breakdown else 'NO'}")
        report.append(f"Regime Change:         {'YES' if risk_metrics.regime_change else 'NO'}")
        
        # Limit check status
        report.append("\nLIMIT CHECK STATUS:")
        for check_name, status in limit_checks.items():
            status_str = "PASS" if status else "FAIL"
            report.append(f"{check_name.replace('_', ' ').title():20s} {status_str}")
        
        # Active alerts
        alerts = self.generate_risk_alerts(risk_metrics, limit_checks)
        if alerts:
            report.append("\nACTIVE ALERTS:")
            for alert in alerts:
                report.append(f"WARNING: {alert}")
        else:
            report.append("\nNO ACTIVE ALERTS")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


if __name__ == "__main__":
    # Test the risk management system
    np.random.seed(42)
    
    # Create test data
    n_stocks = 100
    n_days = 50
    
    # Generate positions
    positions = np.random.randn(n_stocks) * 0.02
    positions = positions / np.sum(np.abs(positions)) * 1.5  # 1.5x leverage
    
    # Generate returns
    returns = np.random.randn(n_days) * 0.01
    returns[20:25] = -0.03  # Simulate drawdown period
    
    # Generate predictions and targets
    predictions = np.random.randn(1000) * 0.02
    true_factor = np.random.randn(1000)
    targets = true_factor * 0.1 + np.random.randn(1000) * 0.015
    predictions = true_factor * 0.08 + np.random.randn(1000) * 0.012  # Correlated
    
    # Test risk manager
    risk_limits = RiskLimits(
        max_leverage=2.0,
        max_concentration=0.05,
        max_volatility=0.15,
        max_drawdown=0.10
    )
    
    risk_manager = RiskManager(risk_limits)
    
    # Check position limits
    position_checks = risk_manager.check_position_limits(positions)
    print("Position Limit Checks:", position_checks)
    
    # Check performance limits  
    performance_checks = risk_manager.check_performance_limits(returns, predictions, targets)
    print("Performance Limit Checks:", performance_checks)
    
    # Compute risk metrics
    risk_metrics = risk_manager.compute_risk_metrics(positions, returns)
    
    # Apply risk constraints
    adjusted_positions = risk_manager.apply_risk_constraints(positions)
    print(f"Original leverage: {np.sum(np.abs(positions)):.3f}")
    print(f"Adjusted leverage: {np.sum(np.abs(adjusted_positions)):.3f}")
    
    # Generate risk report
    all_checks = {**position_checks, **performance_checks}
    risk_report = risk_manager.generate_risk_report(risk_metrics, all_checks)
    print("\n" + risk_report)
    
    print("\n Risk management system test completed successfully!")
