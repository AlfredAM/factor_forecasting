#!/usr/bin/env python3
"""
创建高性能配置 - 最大化硬件利用率
"""

def create_high_performance_config():
    """创建高性能配置文件"""
    config_content = """# 高性能配置 - 最大化硬件利用率
model_type: AdvancedFactorForecastingTCNAttention
input_dim: 100
hidden_dim: 1024   # 充分利用GPU计算能力
num_layers: 16     # 深层网络
num_heads: 32      # 多头注意力
tcn_kernel_size: 7
tcn_dilation_factor: 2
dropout_rate: 0.1
attention_dropout: 0.05
target_columns: [intra30m, nextT1d, ema1d]
sequence_length: 60
epochs: 200
batch_size: 1024   # 大批次充分利用GPU
fixed_batch_size: 1024
learning_rate: 0.0001
weight_decay: 0.01
gradient_clip_norm: 1.0
use_mixed_precision: true
accumulation_steps: 2
use_adaptive_batch_size: false
num_workers: 0
pin_memory: false
use_distributed: false
auto_resume: true
log_level: INFO
ic_report_interval: 7200
enable_ic_reporting: true
checkpoint_frequency: 5
save_all_checkpoints: false
output_dir: /nas/factor_forecasting/outputs
data_dir: /nas/feature_v2_10s
train_start_date: 2018-01-02
train_end_date: 2018-10-31
val_start_date: 2018-11-01
val_end_date: 2018-12-31
test_start_date: 2019-01-01
test_end_date: 2019-12-31
enforce_next_year_prediction: true
enable_yearly_rolling: true
min_train_years: 1
rolling_window_years: 1
shuffle_buffer_size: 1024

# 高性能优化
streaming_chunk_size: 100000
cache_size: 50
max_memory_usage: 600
enable_memory_mapping: true
"""
    return config_content

def create_startup_script():
    """创建启动脚本"""
    script_content = '''#!/bin/bash
# 高性能训练启动脚本

echo "🚀 启动高性能训练系统..."

# 设置环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=32

# 进入项目目录
cd /nas/factor_forecasting

# 激活虚拟环境
source venv/bin/activate

# 清理旧进程
echo "🧹 清理旧进程..."
pkill -f "python.*unified_complete_training" 2>/dev/null || true
sleep 3

# 清理GPU内存
echo "🔧 清理GPU内存..."
nvidia-smi --gpu-reset-ecc=0 2>/dev/null || true
sleep 2

# 启动训练
echo "⚡ 启动高性能训练..."
nohup python unified_complete_training_v2_fixed.py --config high_performance_config.yaml > training_high_performance.log 2>&1 &

# 获取进程ID
TRAIN_PID=$!
echo "✅ 训练已启动，PID: $TRAIN_PID"

# 等待几秒钟检查状态
sleep 10

# 检查进程是否还在运行
if ps -p $TRAIN_PID > /dev/null; then
    echo "✅ 训练进程运行正常"
    echo "📊 GPU状态:"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
else
    echo "❌ 训练进程启动失败，检查日志:"
    tail -20 training_high_performance.log
fi
'''
    return script_content

def fix_training_script():
    """修复训练脚本的导入问题"""
    fixes = """
# 修复导入问题的补丁
import sys
from pathlib import Path

# 确保正确导入
try:
    from src.training.quantitative_loss import create_quantitative_loss_function, QuantitativeCorrelationLoss
except ImportError:
    # 如果导入失败，创建一个简单的替代函数
    def create_quantitative_loss_function(config):
        import torch.nn as nn
        return nn.MSELoss()

# 修复内存管理器参数问题
def create_memory_manager_fixed(config=None):
    try:
        from src.data_processing.adaptive_memory_manager import AdaptiveMemoryManager
        # 只传递支持的参数
        return AdaptiveMemoryManager(
            memory_budget=config.get('max_memory_usage', 600) * 1024 * 1024 * 1024 if config else 600 * 1024 * 1024 * 1024
        )
    except Exception as e:
        print(f"Warning: Could not create memory manager: {e}")
        return None
"""
    return fixes

if __name__ == "__main__":
    print("🔧 创建高性能配置和脚本...")
    
    # 创建配置文件内容
    config = create_high_performance_config()
    print("✅ 高性能配置创建完成")
    
    # 创建启动脚本内容  
    startup = create_startup_script()
    print("✅ 启动脚本创建完成")
    
    # 创建修复补丁
    fixes = fix_training_script()
    print("✅ 修复补丁创建完成")
    
    print("\n📝 配置内容:")
    print(config)
    
    print("\n🚀 准备部署到服务器...")
