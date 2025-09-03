#!/usr/bin/env python3
"""
Scheduled IC Correlation Reporting System
Feature 6 Complete Implementation: Periodic feedback of in-sample and out-of-sample prediction-target correlation
"""

import time
import threading
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
import torch

logger = logging.getLogger(__name__)

class ICCorrelationReporter:
    """IC Correlation Reporter - Generates detailed correlation analysis on schedule"""
    
    def __init__(self, 
                 output_dir: str,
                 target_columns: List[str],
                 report_interval: int = 7200,  # 2 hours
                 history_size: int = 1000):
        """
        Initialize IC Reporter
        
        Args:
            output_dir: Output directory
            target_columns: Target column names
            report_interval: Report interval in seconds
            history_size: Number of historical samples to save
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.target_columns = target_columns
        self.report_interval = report_interval
        self.history_size = history_size
        
        # Data storage
        self.in_sample_predictions = {col: deque(maxlen=history_size) for col in target_columns}
        self.in_sample_targets = {col: deque(maxlen=history_size) for col in target_columns}
        self.out_sample_predictions = {col: deque(maxlen=history_size) for col in target_columns}
        self.out_sample_targets = {col: deque(maxlen=history_size) for col in target_columns}
        
        # Time tracking
        self.last_report_time = 0
        self.start_time = time.time()
        
        # Report history
        self.report_history = []
        
        # Thread control
        self.reporting_thread = None
        self.stop_reporting = False
        
        logger.info(f"IC Correlation Reporter initialized")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Report interval: {report_interval}s ({report_interval/3600:.1f}h)")
        logger.info(f"Target columns: {target_columns}")

    def add_in_sample_data(self, predictions: Dict[str, np.ndarray], targets: Dict[str, np.ndarray]):
        """Add in-sample data"""
        for col in self.target_columns:
            if col in predictions and col in targets:
                self.in_sample_predictions[col].extend(predictions[col].flatten())
                self.in_sample_targets[col].extend(targets[col].flatten())

    def add_out_sample_data(self, predictions: Dict[str, np.ndarray], targets: Dict[str, np.ndarray]):
        """Add out-of-sample data"""
        for col in self.target_columns:
            if col in predictions and col in targets:
                self.out_sample_predictions[col].extend(predictions[col].flatten())
                self.out_sample_targets[col].extend(targets[col].flatten())

    def should_generate_report(self) -> bool:
        """checkreport"""
        current_time = time.time()
        return (current_time - self.last_report_time) >= self.report_interval

    def calculate_ic_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """calculateICmetrics"""
        if len(predictions) == 0 or len(targets) == 0:
            return {}
        
        # NaN
        mask = ~(np.isnan(predictions) | np.isnan(targets))
        if mask.sum() < 10:  # 10sample
            return {}
        
        pred_clean = predictions[mask]
        target_clean = targets[mask]
        
        try:
            # ICmetrics
            ic = np.corrcoef(pred_clean, target_clean)[0, 1]
            rank_ic = np.corrcoef(np.argsort(pred_clean), np.argsort(target_clean))[0, 1]
            
            # metrics
            mse = np.mean((pred_clean - target_clean) ** 2)
            mae = np.mean(np.abs(pred_clean - target_clean))
            
            # ICstatisticst-test
            n = len(pred_clean)
            ic_t_stat = ic * np.sqrt((n - 2) / (1 - ic**2)) if abs(ic) < 0.999 else np.inf
            
            # ICtop/bottom quintiles
            pred_quintiles = np.quantile(pred_clean, [0.2, 0.8])
            top_mask = pred_clean >= pred_quintiles[1]
            bottom_mask = pred_clean <= pred_quintiles[0]
            
            top_ic = np.corrcoef(pred_clean[top_mask], target_clean[top_mask])[0, 1] if top_mask.sum() > 5 else np.nan
            bottom_ic = np.corrcoef(pred_clean[bottom_mask], target_clean[bottom_mask])[0, 1] if bottom_mask.sum() > 5 else np.nan
            
            return {
                'ic': ic,
                'rank_ic': rank_ic,
                'mse': mse,
                'mae': mae,
                'samples': n,
                'ic_t_stat': ic_t_stat,
                'top_quintile_ic': top_ic,
                'bottom_quintile_ic': bottom_ic
            }
            
        except Exception as e:
            logger.error(f"ICcalculateerror: {e}")
            return {}

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """ICreport"""
        current_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = {
            'timestamp': timestamp,
            'elapsed_time_hours': (current_time - self.start_time) / 3600,
            'in_sample_metrics': {},
            'out_sample_metrics': {},
            'cross_sample_comparison': {}
        }
        
        # calculatein-samplemetrics
        for col in self.target_columns:
            if len(self.in_sample_predictions[col]) > 0:
                pred_array = np.array(self.in_sample_predictions[col])
                target_array = np.array(self.in_sample_targets[col])
                metrics = self.calculate_ic_metrics(pred_array, target_array)
                if metrics:
                    report['in_sample_metrics'][col] = metrics
        
        # calculateout-of-samplemetrics
        for col in self.target_columns:
            if len(self.out_sample_predictions[col]) > 0:
                pred_array = np.array(self.out_sample_predictions[col])
                target_array = np.array(self.out_sample_targets[col])
                metrics = self.calculate_ic_metrics(pred_array, target_array)
                if metrics:
                    report['out_sample_metrics'][col] = metrics
        
        # comparison
        for col in self.target_columns:
            if col in report['in_sample_metrics'] and col in report['out_sample_metrics']:
                in_ic = report['in_sample_metrics'][col]['ic']
                out_ic = report['out_sample_metrics'][col]['ic']
                
                report['cross_sample_comparison'][col] = {
                    'ic_degradation': in_ic - out_ic,
                    'ic_retention_ratio': out_ic / in_ic if abs(in_ic) > 1e-8 else np.nan,
                    'overfitting_risk': 'High' if (in_ic - out_ic) > 0.05 else 'Low'
                }
        
        # update
        self.last_report_time = current_time
        self.report_history.append(report)
        
        return report

    def save_report(self, report: Dict[str, Any]):
        """savereportfile"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # savereport
        report_file = self.output_dir / f"ic_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # updatereportlink
        latest_file = self.output_dir / "latest_ic_report.json"
        with open(latest_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # savehistory
        summary_file = self.output_dir / "ic_report_history.json"
        with open(summary_file, 'w') as f:
            json.dump(self.report_history[-50:], f, indent=2, default=str)  # Keep last 50 reports
        
        logger.info(f"IC report saved: {report_file}")

    def log_report(self, report: Dict[str, Any]):
        """Log report to log"""
        logger.info("=" * 80)
        logger.info(f"IC Correlation Report - {report['timestamp']}")
        logger.info(f"Runtime: {report['elapsed_time_hours']:.2f} hours")
        logger.info("=" * 80)
        
        # In-sample metrics
        logger.info("In-Sample Metrics:")
        for col, metrics in report['in_sample_metrics'].items():
            logger.info(f"  {col}:")
            logger.info(f"    IC: {metrics.get('ic', 'N/A'):.4f}")
            logger.info(f"    Rank IC: {metrics.get('rank_ic', 'N/A'):.4f}")
            logger.info(f"    Samples: {metrics.get('samples', 0):,}")
        
        # Out-of-sample metrics
        logger.info("Out-of-Sample Metrics:")
        for col, metrics in report['out_sample_metrics'].items():
            logger.info(f"  {col}:")
            logger.info(f"    IC: {metrics.get('ic', 'N/A'):.4f}")
            logger.info(f"    Rank IC: {metrics.get('rank_ic', 'N/A'):.4f}")
            logger.info(f"    Samples: {metrics.get('samples', 0):,}")
        
        # Cross comparison
        logger.info("Cross-Sample Comparison:")
        for col, comparison in report['cross_sample_comparison'].items():
            logger.info(f"  {col}:")
            logger.info(f"    IC Degradation: {comparison.get('ic_degradation', 'N/A'):.4f}")
            logger.info(f"    IC Retention Ratio: {comparison.get('ic_retention_ratio', 'N/A'):.2%}")
            logger.info(f"    Overfitting Risk: {comparison.get('overfitting_risk', 'N/A')}")
        
        logger.info("=" * 80)

    def start_automatic_reporting(self):
        """Start automatic reporting thread"""
        if self.reporting_thread and self.reporting_thread.is_alive():
            logger.warning("Automatic reporting already running")
            return
        
        self.stop_reporting = False
        self.reporting_thread = threading.Thread(target=self._reporting_loop, daemon=True)
        self.reporting_thread.start()
        logger.info("Automatic IC reporting started")

    def stop_automatic_reporting(self):
        """Stop automatic reporting"""
        self.stop_reporting = True
        if self.reporting_thread and self.reporting_thread.is_alive():
            self.reporting_thread.join(timeout=5)
        logger.info("Automatic IC reporting stopped")

    def _reporting_loop(self):
        """Reporting loop"""
        while not self.stop_reporting:
            try:
                if self.should_generate_report():
                    report = self.generate_comprehensive_report()
                    self.log_report(report)
                    self.save_report(report)
                
                # Wait before checking again
                time.sleep(min(60, self.report_interval // 10))  # Wait at most 1 minute
                
            except Exception as e:
                logger.error(f"Automatic reporting error: {e}")
                time.sleep(60)  # Wait 1 minute after error

    def force_report(self) -> Dict[str, Any]:
        """Force generate report"""
        report = self.generate_comprehensive_report()
        self.log_report(report)
        self.save_report(report)
        return report

    def get_latest_metrics(self) -> Dict[str, float]:
        """Get latest simplified metrics"""
        metrics = {}
        
        for col in self.target_columns:
            # In-sample
            if len(self.in_sample_predictions[col]) > 0:
                pred = np.array(self.in_sample_predictions[col])
                target = np.array(self.in_sample_targets[col])
                mask = ~(np.isnan(pred) | np.isnan(target))
                if mask.sum() > 10:
                    try:
                        ic = np.corrcoef(pred[mask], target[mask])[0, 1]
                        # 处理NaN和Inf值
                        if np.isnan(ic) or np.isinf(ic):
                            ic = 0.0
                        metrics[f'{col}_in_sample_ic'] = ic
                    except Exception as e:
                        logger.warning(f"IC calculation failed for {col}: {e}")
                        metrics[f'{col}_in_sample_ic'] = 0.0
            
            # Out-of-sample
            if len(self.out_sample_predictions[col]) > 0:
                pred = np.array(self.out_sample_predictions[col])
                target = np.array(self.out_sample_targets[col])
                mask = ~(np.isnan(pred) | np.isnan(target))
                if mask.sum() > 10:
                    try:
                        ic = np.corrcoef(pred[mask], target[mask])[0, 1]
                        # 处理NaN和Inf值
                        if np.isnan(ic) or np.isinf(ic):
                            ic = 0.0
                        metrics[f'{col}_out_sample_ic'] = ic
                    except Exception as e:
                        logger.warning(f"IC calculation failed for {col}: {e}")
                        metrics[f'{col}_out_sample_ic'] = 0.0
        
        return metrics

    def cleanup(self):
        """cleanupresource"""
        self.stop_automatic_reporting()


def create_ic_reporter(output_dir: str, target_columns: List[str], report_interval: int = 7200) -> ICCorrelationReporter:
    """createICreportfunction"""
    return ICCorrelationReporter(
        output_dir=output_dir,
        target_columns=target_columns,
        report_interval=report_interval
    )
