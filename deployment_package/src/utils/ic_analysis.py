"""
IC Analysis and Backtesting Module for Factor Forecasting
Provides comprehensive IC calculations, backtesting metrics, and visualizations
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
# Only suppress specific warnings that are known to be harmless
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, message='.*NumPy.*')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ICAnalyzer:
    """
    Comprehensive IC analysis for factor forecasting models.
    Calculates various IC metrics, performs statistical tests, and creates visualizations.
    """
    
    def __init__(self, predictions: Dict[str, np.ndarray], 
                 targets: Dict[str, np.ndarray],
                 dates: List[str] = None,
                 stock_ids: List[int] = None,
                 market_caps: Optional[np.ndarray] = None,
                 industries: Optional[List[str]] = None,
                 betas: Optional[np.ndarray] = None):
        """
        Initialize IC analyzer.
        
        Args:
            predictions: Dictionary of predictions for each target
            targets: Dictionary of actual targets
            dates: List of dates for time series analysis
            stock_ids: List of stock IDs
            market_caps: Market capitalization data for grouping
            industries: Industry classification data
            betas: Beta values for risk grouping
        """
        self.predictions = predictions
        self.targets = targets
        self.dates = dates
        self.stock_ids = stock_ids
        self.market_caps = market_caps
        self.industries = industries
        self.betas = betas
        
        # Validate inputs
        self._validate_inputs()
        
        # Calculate basic IC metrics
        self.ic_metrics = self._calculate_basic_ic()
        
        logger.info("IC Analyzer initialized successfully")
    
    def _validate_inputs(self):
        """Validate input data."""
        if not self.predictions or not self.targets:
            raise ValueError("Predictions and targets must be provided")
        
        for target in self.predictions.keys():
            if target not in self.targets:
                raise ValueError(f"Target {target} not found in targets")
            
            if len(self.predictions[target]) != len(self.targets[target]):
                raise ValueError(f"Length mismatch for target {target}")
        
        if self.dates and len(self.dates) != len(self.predictions[list(self.predictions.keys())[0]]):
            raise ValueError("Dates length mismatch")
    
    def _calculate_basic_ic(self) -> Dict[str, Dict[str, float]]:
        """Calculate basic IC metrics for each target."""
        ic_metrics = {}
        
        for target in self.predictions.keys():
            pred = self.predictions[target]
            true = self.targets[target]
            
            # Pearson correlation (IC)
            ic = np.corrcoef(pred, true)[0, 1]
            if np.isnan(ic):
                ic = 0.0
            
            # Rank IC
            rank_ic = np.corrcoef(np.argsort(np.argsort(pred)), np.argsort(np.argsort(true)))[0, 1]
            if np.isnan(rank_ic):
                rank_ic = 0.0
            
            # IC significance test
            n = len(pred)
            if n > 3:
                ic_se = np.sqrt((1 - ic**2) / (n - 2))
                ic_t_stat = ic / ic_se if ic_se > 0 else 0
                ic_p_value = 2 * (1 - abs(stats.t.cdf(ic_t_stat, n - 2)))
            else:
                ic_t_stat = 0
                ic_p_value = 1.0
            
            ic_metrics[target] = {
                'ic': ic,
                'rank_ic': rank_ic,
                'ic_t_stat': ic_t_stat,
                'ic_p_value': ic_p_value,
                'ic_significant': ic_p_value < 0.05,
                'sample_size': n
            }
        
        return ic_metrics
    
    def calculate_daily_ic(self) -> Dict[str, pd.DataFrame]:
        """
        Calculate daily IC values for time series analysis.
        
        Returns:
            Dictionary of daily IC DataFrames for each target
        """
        if not self.dates:
            raise ValueError("Dates must be provided for daily IC calculation")
        
        daily_ic = {}
        
        for target in self.predictions.keys():
            pred = self.predictions[target]
            true = self.targets[target]
            
            # Group by date and calculate IC
            df = pd.DataFrame({
                'date': self.dates,
                'prediction': pred,
                'target': true
            })
            
            daily_ic_values = []
            daily_dates = []
            
            for date in sorted(set(self.dates)):
                date_data = df[df['date'] == date]
                if len(date_data) > 10:  # Minimum sample size for IC calculation
                    ic = np.corrcoef(date_data['prediction'], date_data['target'])[0, 1]
                    rank_ic = np.corrcoef(
                        np.argsort(np.argsort(date_data['prediction'])), 
                        np.argsort(np.argsort(date_data['target']))
                    )[0, 1]
                    
                    daily_ic_values.append({
                        'date': date,
                        'ic': ic if not np.isnan(ic) else 0.0,
                        'rank_ic': rank_ic if not np.isnan(rank_ic) else 0.0,
                        'sample_size': len(date_data)
                    })
                    daily_dates.append(date)
            
            daily_ic[target] = pd.DataFrame(daily_ic_values)
        
        return daily_ic
    
    def calculate_grouped_ic(self, group_type: str = 'market_cap') -> Dict[str, Dict[str, float]]:
        """
        Calculate IC for different groups (market cap, industry, beta).
        
        Args:
            group_type: Type of grouping ('market_cap', 'industry', 'beta')
            
        Returns:
            Dictionary of IC metrics for each group
        """
        grouped_ic = {}
        
        if group_type == 'market_cap' and self.market_caps is not None:
            # Market cap quintiles
            quintiles = pd.qcut(self.market_caps, 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
            groups = quintiles
        elif group_type == 'industry' and self.industries is not None:
            groups = self.industries
        elif group_type == 'beta' and self.betas is not None:
            # Beta terciles
            terciles = pd.qcut(self.betas, 3, labels=['Low', 'Mid', 'High'])
            groups = terciles
        else:
            raise ValueError(f"Group type {group_type} not supported or data not available")
        
        for target in self.predictions.keys():
            pred = self.predictions[target]
            true = self.targets[target]
            
            target_grouped_ic = {}
            
            for group in set(groups):
                if pd.isna(group):
                    continue
                
                mask = groups == group
                if np.sum(mask) > 10:  # Minimum sample size
                    group_pred = pred[mask]
                    group_true = true[mask]
                    
                    ic = np.corrcoef(group_pred, group_true)[0, 1]
                    rank_ic = np.corrcoef(
                        np.argsort(np.argsort(group_pred)), 
                        np.argsort(np.argsort(group_true))
                    )[0, 1]
                    
                    target_grouped_ic[str(group)] = {
                        'ic': ic if not np.isnan(ic) else 0.0,
                        'rank_ic': rank_ic if not np.isnan(rank_ic) else 0.0,
                        'sample_size': len(group_pred)
                    }
            
            grouped_ic[target] = target_grouped_ic
        
        return grouped_ic
    
    def calculate_rolling_ic(self, window: int = 20) -> Dict[str, pd.DataFrame]:
        """
        Calculate rolling IC for time series analysis.
        
        Args:
            window: Rolling window size
            
        Returns:
            Dictionary of rolling IC DataFrames for each target
        """
        if not self.dates:
            raise ValueError("Dates must be provided for rolling IC calculation")
        
        rolling_ic = {}
        
        for target in self.predictions.keys():
            pred = self.predictions[target]
            true = self.targets[target]
            
            # Create time series DataFrame
            df = pd.DataFrame({
                'date': pd.to_datetime(self.dates),
                'prediction': pred,
                'target': true
            }).sort_values('date')
            
            # Calculate rolling IC
            rolling_ic_values = []
            
            for i in range(window, len(df)):
                window_data = df.iloc[i-window:i]
                if len(window_data) >= window:
                    ic = np.corrcoef(window_data['prediction'], window_data['target'])[0, 1]
                    rank_ic = np.corrcoef(
                        np.argsort(np.argsort(window_data['prediction'])), 
                        np.argsort(np.argsort(window_data['target']))
                    )[0, 1]
                    
                    rolling_ic_values.append({
                        'date': window_data.iloc[-1]['date'],
                        'ic': ic if not np.isnan(ic) else 0.0,
                        'rank_ic': rank_ic if not np.isnan(rank_ic) else 0.0,
                        'window_size': window
                    })
            
            rolling_ic[target] = pd.DataFrame(rolling_ic_values)
        
        return rolling_ic
    
    def calculate_backtest_metrics(self, transaction_cost: float = 0.001) -> Dict[str, Dict[str, float]]:
        """
        Calculate backtesting metrics including returns, turnover, and fees.
        
        Args:
            transaction_cost: Transaction cost as a fraction
            
        Returns:
            Dictionary of backtesting metrics for each target
        """
        backtest_metrics = {}
        
        for target in self.predictions.keys():
            pred = self.predictions[target]
            true = self.targets[target]
            
            # Calculate portfolio weights (simple long-short strategy)
            weights = pred / (np.abs(pred).sum() + 1e-8)
            
            # Calculate returns
            returns = weights * true
            
            # Calculate cumulative returns
            cumulative_returns = np.cumprod(1 + returns)
            
            # Calculate turnover
            if len(weights) > 1:
                turnover = np.mean(np.abs(weights[1:] - weights[:-1]))
            else:
                turnover = 0.0
            
            # Calculate transaction costs
            total_cost = turnover * transaction_cost
            
            # Calculate net returns
            net_returns = returns - total_cost
            net_cumulative_returns = np.cumprod(1 + net_returns)
            
            # Calculate metrics
            total_return = cumulative_returns[-1] - 1 if len(cumulative_returns) > 0 else 0
            net_total_return = net_cumulative_returns[-1] - 1 if len(net_cumulative_returns) > 0 else 0
            
            volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252) if len(returns) > 1 else 0
            
            max_drawdown = self._calculate_max_drawdown(cumulative_returns)
            
            backtest_metrics[target] = {
                'total_return': total_return,
                'net_total_return': net_total_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'turnover': turnover,
                'transaction_cost': total_cost,
                'information_ratio': np.mean(returns) / (np.std(returns - true) + 1e-8) if len(returns) > 1 else 0
            }
        
        return backtest_metrics
    
    def _calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(cumulative_returns) == 0:
            return 0.0
        
        peak = cumulative_returns[0]
        max_dd = 0.0
        
        for value in cumulative_returns:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def plot_ic_time_series(self, save_path: str = None):
        """
        Plot IC time series for all targets.
        
        Args:
            save_path: Path to save the plot
        """
        daily_ic = self.calculate_daily_ic()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot daily IC
        for i, (target, ic_df) in enumerate(daily_ic.items()):
            if len(ic_df) > 0:
                ax = axes[i // 2, i % 2]
                ax.plot(ic_df['date'], ic_df['ic'], label='IC', alpha=0.7)
                ax.plot(ic_df['date'], ic_df['rank_ic'], label='Rank IC', alpha=0.7)
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax.set_title(f'{target} - Daily IC')
                ax.set_xlabel('Date')
                ax.set_ylabel('IC Value')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Rotate x-axis labels
                plt.setp(ax.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"IC time series plot saved to {save_path}")
        
        plt.show()
    
    def plot_cumulative_returns(self, save_path: str = None):
        """
        Plot cumulative returns for backtesting.
        
        Args:
            save_path: Path to save the plot
        """
        backtest_metrics = self.calculate_backtest_metrics()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for i, (target, metrics) in enumerate(backtest_metrics.items()):
            if i >= 4:  # Limit to 4 subplots
                break
            
            ax = axes[i // 2, i % 2]
            
            # Calculate cumulative returns for plotting
            pred = self.predictions[target]
            true = self.targets[target]
            weights = pred / (np.abs(pred).sum() + 1e-8)
            returns = weights * true
            cumulative_returns = np.cumprod(1 + returns)
            
            # Plot cumulative returns
            dates = range(len(cumulative_returns))
            ax.plot(dates, cumulative_returns, label=f'{target} (Return: {metrics["total_return"]:.2%})')
            ax.set_title(f'{target} - Cumulative Returns')
            ax.set_xlabel('Time')
            ax.set_ylabel('Cumulative Return')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Cumulative returns plot saved to {save_path}")
        
        plt.show()
    
    def plot_grouped_ic(self, group_type: str = 'market_cap', save_path: str = None):
        """
        Plot grouped IC analysis.
        
        Args:
            group_type: Type of grouping
            save_path: Path to save the plot
        """
        grouped_ic = self.calculate_grouped_ic(group_type)
        
        fig, axes = plt.subplots(1, len(grouped_ic), figsize=(5*len(grouped_ic), 6))
        if len(grouped_ic) == 1:
            axes = [axes]
        
        for i, (target, group_metrics) in enumerate(grouped_ic.items()):
            ax = axes[i]
            
            groups = list(group_metrics.keys())
            ic_values = [group_metrics[g]['ic'] for g in groups]
            rank_ic_values = [group_metrics[g]['rank_ic'] for g in groups]
            
            x = np.arange(len(groups))
            width = 0.35
            
            ax.bar(x - width/2, ic_values, width, label='IC', alpha=0.7)
            ax.bar(x + width/2, rank_ic_values, width, label='Rank IC', alpha=0.7)
            
            ax.set_title(f'{target} - {group_type.title()} Grouped IC')
            ax.set_xlabel('Group')
            ax.set_ylabel('IC Value')
            ax.set_xticks(x)
            ax.set_xticklabels(groups)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Grouped IC plot saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, output_dir: str = "ic_analysis"):
        """
        Generate comprehensive IC analysis report.
        
        Args:
            output_dir: Directory to save the report
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate all metrics
        daily_ic = self.calculate_daily_ic()
        
        # Only calculate grouped IC if market cap data is available
        grouped_ic = {}
        try:
            grouped_ic = self.calculate_grouped_ic('market_cap')
        except ValueError as e:
            logger.info(f"Skipping grouped IC analysis: {e}")
        
        backtest_metrics = self.calculate_backtest_metrics()
        
        # Generate plots
        self.plot_ic_time_series(os.path.join(output_dir, 'ic_time_series.png'))
        self.plot_cumulative_returns(os.path.join(output_dir, 'cumulative_returns.png'))
        
        # Only plot grouped IC if data is available
        if grouped_ic:
            self.plot_grouped_ic('market_cap', os.path.join(output_dir, 'grouped_ic.png'))
        
        # Save metrics to files
        for target in self.predictions.keys():
            # Save daily IC
            if target in daily_ic:
                daily_ic[target].to_csv(os.path.join(output_dir, f'{target}_daily_ic.csv'), index=False)
            
            # Save grouped IC if available
            if target in grouped_ic:
                grouped_df = pd.DataFrame(grouped_ic[target]).T
                grouped_df.to_csv(os.path.join(output_dir, f'{target}_grouped_ic.csv'))
        
        # Save backtest metrics
        backtest_df = pd.DataFrame(backtest_metrics).T
        backtest_df.to_csv(os.path.join(output_dir, 'backtest_metrics.csv'))
        
        # Generate summary report
        self._generate_summary_report(output_dir)
        
        logger.info(f"IC analysis report generated in {output_dir}")
    
    def _generate_summary_report(self, output_dir: str):
        """Generate summary report."""
        import os
        report_lines = []
        report_lines.append("Factor Forecasting IC Analysis Report")
        report_lines.append("=" * 50)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Basic IC metrics
        report_lines.append("Basic IC Metrics:")
        report_lines.append("-" * 20)
        for target, metrics in self.ic_metrics.items():
            report_lines.append(f"{target}:")
            report_lines.append(f"  IC: {metrics['ic']:.4f}")
            report_lines.append(f"  Rank IC: {metrics['rank_ic']:.4f}")
            report_lines.append(f"  Significance: {'Yes' if metrics['ic_significant'] else 'No'}")
            report_lines.append(f"  Sample Size: {metrics['sample_size']}")
            report_lines.append("")
        
        # Backtest metrics
        backtest_metrics = self.calculate_backtest_metrics()
        report_lines.append("Backtest Metrics:")
        report_lines.append("-" * 20)
        for target, metrics in backtest_metrics.items():
            report_lines.append(f"{target}:")
            report_lines.append(f"  Total Return: {metrics['total_return']:.2%}")
            report_lines.append(f"  Net Return: {metrics['net_total_return']:.2%}")
            report_lines.append(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            report_lines.append(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
            report_lines.append(f"  Turnover: {metrics['turnover']:.2%}")
            report_lines.append("")
        
        # Save report
        with open(os.path.join(output_dir, 'summary_report.txt'), 'w') as f:
            f.write('\n'.join(report_lines))


def create_ic_analyzer_from_predictions(predictions: Dict[str, np.ndarray], 
                                      targets: Dict[str, np.ndarray],
                                      dates: List[str] = None,
                                      **kwargs) -> ICAnalyzer:
    """
    Factory function to create IC analyzer from predictions and targets.
    
    Args:
        predictions: Model predictions
        targets: Actual targets
        dates: Dates for time series analysis
        **kwargs: Additional arguments for ICAnalyzer
        
    Returns:
        ICAnalyzer instance
    """
    return ICAnalyzer(predictions, targets, dates, **kwargs)


if __name__ == "__main__":
    # Test the IC analyzer
    np.random.seed(42)
    
    # Generate sample data
    n_samples = 1000
    dates = [f"2023-{i//30+1:02d}-{i%30+1:02d}" for i in range(n_samples)]
    
    # Sample predictions and targets
    predictions = {
        'intra30m': np.random.randn(n_samples),
        'nextT1d': np.random.randn(n_samples),
        'ema1d': np.random.randn(n_samples)
    }
    
    targets = {
        'intra30m': predictions['intra30m'] * 0.3 + np.random.randn(n_samples) * 0.7,
        'nextT1d': predictions['nextT1d'] * 0.4 + np.random.randn(n_samples) * 0.6,
        'ema1d': predictions['ema1d'] * 0.5 + np.random.randn(n_samples) * 0.5
    }
    
    # Sample market cap data
    market_caps = np.random.lognormal(10, 1, n_samples)
    
    # Create analyzer
    analyzer = ICAnalyzer(
        predictions=predictions,
        targets=targets,
        dates=dates,
        market_caps=market_caps
    )
    
    # Generate report
    analyzer.generate_report("test_ic_analysis")
    
    print("IC Analyzer test completed successfully!") 