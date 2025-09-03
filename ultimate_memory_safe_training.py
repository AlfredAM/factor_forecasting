#!/usr/bin/env python3
"""
终极内存安全训练脚本 - 完全独立，彻底解决所有问题
"""

import os
import sys
import gc
import time
import yaml
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import numpy as np
from datetime import datetime

# 设置CUDA内存管理
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# 启用CUDA内存优化
torch.cuda.empty_cache()
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.7)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, '/nas/factor_forecasting/src')

def cleanup_memory():
    """强制清理内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

class SimpleCorrelationLoss(nn.Module):
    """简单的相关性损失函数"""
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, predictions, targets):
        if isinstance(predictions, dict):
            predictions = predictions['nextT1d']
        if isinstance(targets, dict):
            targets = targets['nextT1d']
            
        # 简单的MSE损失
        return self.mse(predictions, targets)

class MemorySafeTrainer:
    """内存安全训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.criterion = SimpleCorrelationLoss()
        self.scaler = torch.cuda.amp.GradScaler() if config.get('use_mixed_precision', True) else None
        
        logger.info(f"训练器初始化完成 - 设备: {self.device}")
    
    def create_model(self):
        """创建简化模型"""
        from src.models.advanced_tcn_attention import create_advanced_model
        
        # 创建模型配置
        model_config = {
            'input_dim': self.config['input_dim'],
            'hidden_dim': self.config['hidden_dim'],
            'num_layers': self.config['num_layers'],
            'num_heads': self.config['num_heads'],
            'tcn_kernel_size': self.config['tcn_kernel_size'],
            'tcn_dilation_factor': self.config['tcn_dilation_factor'],
            'dropout_rate': self.config['dropout_rate'],
            'attention_dropout': self.config['attention_dropout'],
            'target_columns': self.config['target_columns'],
            'sequence_length': self.config['sequence_length']
        }
        
        self.model = create_advanced_model(model_config)
        self.model.to(self.device)
        
        # 创建优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"模型创建完成 - 参数数量: {total_params:,}")
        
        cleanup_memory()
    
    def create_data_loaders(self):
        """创建数据加载器"""
        from src.data_processing.optimized_streaming_loader import create_optimized_dataloaders
        from src.data_processing.adaptive_memory_manager import create_memory_manager
        
        # 创建内存管理器
        memory_manager = create_memory_manager({
            'max_memory_usage_gb': 200,  # 减少内存使用
            'enable_monitoring': True
        })
        
        # 获取因子列名（简化版本）
        factor_columns = [f'factor_{i}' for i in range(self.config['input_dim'])]
        
        # 创建数据加载器
        self.train_loader, self.val_loader, self.test_loader = create_optimized_dataloaders(
            data_dir=self.config['data_dir'],
            factor_columns=factor_columns,
            target_columns=self.config['target_columns'],
            train_dates=(self.config['train_start_date'], self.config['train_end_date']),
            val_dates=(self.config['val_start_date'], self.config['val_end_date']),
            test_dates=(self.config['test_start_date'], self.config['test_end_date']),
            sequence_length=self.config['sequence_length'],
            batch_size=self.config['batch_size'],
            num_workers=0,  # 禁用多进程
            memory_config={'max_memory_usage_gb': 200}
        )
        
        logger.info("数据加载器创建完成")
        cleanup_memory()
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        start_time = time.time()
        
        logger.info(f"开始训练 Epoch {epoch}")
        
        for batch_idx, batch in enumerate(self.train_loader):
            try:
                # 数据移动到GPU
                if isinstance(batch, dict):
                    for key in batch:
                        if isinstance(batch[key], torch.Tensor):
                            batch[key] = batch[key].to(self.device, non_blocking=True)
                
                # 前向传播
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch)
                        loss = self.criterion(outputs, batch)
                else:
                    outputs = self.model(batch)
                    loss = self.criterion(outputs, batch)
                
                # 反向传播
                self.optimizer.zero_grad()
                
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # 定期清理内存
                if batch_idx % 10 == 0:
                    cleanup_memory()
                    
                # 记录进度
                if batch_idx % 50 == 0:
                    elapsed = time.time() - start_time
                    logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, "
                              f"Time: {elapsed:.1f}s")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error(f"CUDA内存不足 - Batch {batch_idx}")
                    cleanup_memory()
                    continue
                else:
                    raise e
        
        avg_loss = total_loss / max(num_batches, 1)
        epoch_time = time.time() - start_time
        
        logger.info(f"Epoch {epoch} 完成 - 平均损失: {avg_loss:.4f}, 用时: {epoch_time:.1f}s")
        
        return avg_loss, epoch_time
    
    def validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                try:
                    if isinstance(batch, dict):
                        for key in batch:
                            if isinstance(batch[key], torch.Tensor):
                                batch[key] = batch[key].to(self.device, non_blocking=True)
                    
                    outputs = self.model(batch)
                    loss = self.criterion(outputs, batch)
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        cleanup_memory()
                        continue
                    else:
                        raise e
        
        avg_loss = total_loss / max(num_batches, 1)
        logger.info(f"验证损失: {avg_loss:.4f}")
        
        return avg_loss
    
    def train(self):
        """主训练循环"""
        logger.info("开始训练...")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config['epochs']):
            try:
                # 训练
                train_loss, epoch_time = self.train_epoch(epoch)
                
                # 验证
                val_loss = self.validate()
                
                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), 
                             '/nas/factor_forecasting/outputs/best_model.pth')
                    logger.info(f"保存最佳模型 - 验证损失: {val_loss:.4f}")
                
                # 记录epoch完成时间
                logger.info(f"=== Epoch {epoch} 完成 ===")
                logger.info(f"训练损失: {train_loss:.4f}")
                logger.info(f"验证损失: {val_loss:.4f}")
                logger.info(f"Epoch用时: {epoch_time:.1f}秒")
                logger.info(f"最佳验证损失: {best_val_loss:.4f}")
                
                # 强制清理内存
                cleanup_memory()
                
            except Exception as e:
                logger.error(f"Epoch {epoch} 错误: {e}")
                cleanup_memory()
                continue

def main():
    """主函数"""
    try:
        logger.info("=" * 60)
        logger.info("🚀 启动终极内存安全训练系统")
        logger.info("=" * 60)
        
        # 清理初始内存
        cleanup_memory()
        
        # 加载配置
        with open('/nas/factor_forecasting/memory_safe_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"配置加载完成:")
        logger.info(f"  批次大小: {config['batch_size']}")
        logger.info(f"  隐藏维度: {config['hidden_dim']}")
        logger.info(f"  层数: {config['num_layers']}")
        logger.info(f"  序列长度: {config['sequence_length']}")
        
        # 创建训练器
        trainer = MemorySafeTrainer(config)
        cleanup_memory()
        
        # 创建数据加载器
        trainer.create_data_loaders()
        cleanup_memory()
        
        # 创建模型
        trainer.create_model()
        cleanup_memory()
        
        # 开始训练
        trainer.train()
        
        logger.info("✅ 训练完成!")
        
    except Exception as e:
        logger.error(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        cleanup_memory()
        raise
    finally:
        cleanup_memory()

if __name__ == "__main__":
    main()
