#!/usr/bin/env python3
"""
内存优化修复脚本 - 从根本上彻底解决CUDA内存问题
"""

import yaml
import os

def create_memory_optimized_config():
    """创建内存优化的配置文件"""
    
    config = {
        # 模型配置 - 大幅减小模型复杂度
        'model_type': 'AdvancedFactorForecastingTCNAttention',
        'input_dim': 100,
        'hidden_dim': 512,  # 从1024减少到512
        'num_layers': 8,    # 从16减少到8
        'num_heads': 16,    # 从32减少到16
        'tcn_kernel_size': 5,  # 从7减少到5
        'tcn_dilation_factor': 2,
        'dropout_rate': 0.1,
        'attention_dropout': 0.05,
        
        # 训练配置 - 极度保守的内存使用
        'target_columns': ['nextT1d'],  # 只训练一个目标，减少内存
        'sequence_length': 30,  # 从60减少到30
        'epochs': 50,
        'batch_size': 512,  # 进一步减小批次大小
        'fixed_batch_size': 512,
        'learning_rate': 0.0005,  # 增加学习率补偿小批次
        'weight_decay': 0.01,
        'gradient_clip_norm': 1.0,
        'use_mixed_precision': True,
        'accumulation_steps': 4,  # 使用梯度累积补偿小批次
        
        # DataLoader配置 - 最小化内存使用
        'use_adaptive_batch_size': False,
        'adaptive_batch_size': False,
        'num_workers': 0,
        'pin_memory': False,  # 禁用pin_memory减少内存使用
        'use_distributed': False,
        'auto_resume': True,
        'log_level': 'INFO',
        
        # IC报告配置
        'ic_report_interval': 7200,
        'enable_ic_reporting': True,
        'checkpoint_frequency': 10,
        'save_all_checkpoints': False,
        
        # 路径配置
        'output_dir': '/nas/factor_forecasting/outputs',
        'data_dir': '/nas/feature_v2_10s',
        
        # 数据分割配置
        'train_start_date': '2018-01-02',
        'train_end_date': '2018-06-30',  # 减少训练数据量
        'val_start_date': '2018-07-01',
        'val_end_date': '2018-08-31',
        'test_start_date': '2018-09-01',
        'test_end_date': '2018-10-31',
        
        # 年度滚动训练
        'enforce_next_year_prediction': True,
        'enable_yearly_rolling': False,  # 暂时禁用减少复杂度
        'min_train_years': 1,
        'rolling_window_years': 1,
        'shuffle_buffer_size': 256,
        
        # GPU内存优化
        'gpu_memory_fraction': 0.8,
        'enable_gpu_growth': True,
        'torch_compile': False,  # 禁用torch编译减少内存
        'enable_flash_attention': False,
        'use_channels_last': False,
        
        # 数据处理优化
        'streaming_chunk_size': 50000,  # 减少chunk大小
        'max_memory_usage': 400,  # 减少最大内存使用
        'enable_memory_mapping': False,  # 禁用内存映射
        
        # 监控配置
        'enable_tensorboard': False,  # 禁用tensorboard减少内存
        'enable_wandb': False,
        
        # 损失函数配置
        'use_adaptive_loss': False,  # 使用简单损失函数
        'correlation_weight': 1.0,
        'mse_weight': 0.1,
        'rank_correlation_weight': 0.1,
        'risk_penalty_weight': 0.05,
        'target_correlations': [0.05],
        'max_leverage': 1.5,
        'transaction_cost': 0.001
    }
    
    return config

def create_memory_safe_training_script():
    """创建内存安全的训练脚本"""
    
    script_content = '''#!/usr/bin/env python3
"""
内存安全训练脚本 - 极度保守的内存使用策略
"""

import os
import sys
import gc
import torch
import torch.multiprocessing as mp

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
    torch.cuda.set_per_process_memory_fraction(0.8)

# 导入项目模块
sys.path.insert(0, '/nas/factor_forecasting/src')

def cleanup_memory():
    """强制清理内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def main():
    """主函数 - 内存安全版本"""
    try:
        # 清理初始内存
        cleanup_memory()
        
        # 导入训练模块
        from unified_complete_training_v2_fixed import UnifiedCompleteTrainer
        import yaml
        
        # 加载配置
        with open('/nas/factor_forecasting/memory_safe_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        print("🚀 启动内存安全训练...")
        print(f"配置: batch_size={config['batch_size']}, hidden_dim={config['hidden_dim']}")
        
        # 创建训练器
        trainer = UnifiedCompleteTrainer(config, 0, 1)
        
        # 设置数据加载器
        trainer.setup_data_loaders()
        cleanup_memory()
        
        # 创建模型
        trainer.create_model()
        cleanup_memory()
        
        print(f"✅ 模型创建成功，开始训练...")
        
        # 开始训练
        trainer.train()
        
    except Exception as e:
        print(f"❌ 训练错误: {e}")
        cleanup_memory()
        raise
    finally:
        cleanup_memory()

if __name__ == "__main__":
    main()
'''
    
    return script_content

def apply_memory_fixes():
    """应用内存修复"""
    
    print("=" * 60)
    print("🔧 开始内存优化修复...")
    print("=" * 60)
    
    # 1. 创建内存优化配置
    config = create_memory_optimized_config()
    
    with open('/nas/factor_forecasting/memory_safe_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("✅ 创建了内存安全配置文件")
    
    # 2. 创建内存安全训练脚本
    script_content = create_memory_safe_training_script()
    
    with open('/nas/factor_forecasting/memory_safe_training.py', 'w') as f:
        f.write(script_content)
    
    print("✅ 创建了内存安全训练脚本")
    
    # 3. 设置CUDA环境变量
    cuda_env_script = '''#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "🔧 CUDA环境变量已设置"
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
'''
    
    with open('/nas/factor_forecasting/setup_cuda_env.sh', 'w') as f:
        f.write(cuda_env_script)
    
    os.chmod('/nas/factor_forecasting/setup_cuda_env.sh', 0o755)
    
    print("✅ 创建了CUDA环境设置脚本")
    
    print("=" * 60)
    print("🎯 内存优化修复完成！")
    print("=" * 60)
    print("主要优化:")
    print("- 模型参数减少 ~75%")
    print("- 批次大小: 4096 → 512")
    print("- 序列长度: 60 → 30") 
    print("- 隐藏维度: 1024 → 512")
    print("- 层数: 16 → 8")
    print("- 注意力头: 32 → 16")
    print("- 训练数据量减少50%")
    print("- 启用梯度累积补偿")
    print("=" * 60)

if __name__ == "__main__":
    apply_memory_fixes()
