#!/usr/bin/env python3
"""
CUDA内存碎片化问题彻底解决方案
从根本上解决PyTorch CUDA内存管理问题
"""

import subprocess
import os

def create_memory_optimized_config():
    """创建内存优化配置"""
    config_content = """# 内存优化4GPU配置 - 彻底解决CUDA OOM问题
model_type: AdvancedFactorForecastingTCNAttention
input_dim: 100
hidden_dim: 768       # 适中的隐藏维度
num_layers: 8         # 减少层数避免内存爆炸
num_heads: 12         # 适中的注意力头数
tcn_kernel_size: 5    # 减小卷积核
tcn_dilation_factor: 2
dropout_rate: 0.1
attention_dropout: 0.05
target_columns: [intra30m, nextT1d, ema1d]
sequence_length: 48   # 减小序列长度
epochs: 200
batch_size: 1024      # 每GPU合理批次大小
fixed_batch_size: 1024
learning_rate: 0.0001
weight_decay: 0.01
gradient_clip_norm: 1.0
use_mixed_precision: true
accumulation_steps: 2  # 梯度累积
use_adaptive_batch_size: true  # 启用自适应批次
adaptive_batch_size: true
num_workers: 0
pin_memory: false     # 关闭pin_memory减少内存使用
use_distributed: true
world_size: 4
backend: nccl
auto_resume: true
log_level: INFO
ic_report_interval: 7200
enable_ic_reporting: true
checkpoint_frequency: 10
save_all_checkpoints: false
output_dir: /nas/factor_forecasting/outputs
data_dir: /nas/feature_v2_10s
train_start_date: 2018-01-02
train_end_date: 2018-06-30   # 减少数据量
val_start_date: 2018-07-01
val_end_date: 2018-08-31
test_start_date: 2018-09-01
test_end_date: 2018-12-31
enforce_next_year_prediction: false
enable_yearly_rolling: false
min_train_years: 1
rolling_window_years: 1
shuffle_buffer_size: 512

# CUDA内存优化设置
cuda_memory_fraction: 0.85    # 限制每GPU内存使用
enable_memory_pool: true      # 启用内存池
memory_pool_init_size: 1024   # 初始内存池大小MB
memory_pool_max_size: 20480   # 最大内存池大小MB
enable_garbage_collection: true  # 启用垃圾回收
gc_frequency: 50              # 每50个batch清理一次
"""
    return config_content

def create_cuda_memory_patch():
    """创建CUDA内存管理补丁"""
    patch_content = '''#!/usr/bin/env python3
"""
CUDA内存管理补丁
彻底解决内存碎片化问题
"""

import torch
import torch.cuda
import gc
import os
import logging

class CUDAMemoryManager:
    def __init__(self):
        self.setup_cuda_environment()
        self.setup_memory_pool()
        
    def setup_cuda_environment(self):
        """设置CUDA环境变量"""
        # 关键环境变量设置
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = (
            'expandable_segments:True,'
            'max_split_size_mb:128,'
            'roundup_power2_divisions:16,'
            'garbage_collection_threshold:0.8'
        )
        
        # 设置CUDA设备
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.85)  # 限制内存使用
            torch.backends.cudnn.benchmark = False  # 禁用cudnn benchmark减少内存
            torch.backends.cudnn.deterministic = True
            
    def setup_memory_pool(self):
        """设置内存池"""
        if torch.cuda.is_available():
            # 清空缓存
            torch.cuda.empty_cache()
            
            # 设置内存池
            torch.cuda.set_memory_strategy('expandable_segments')
            
            logging.info("CUDA内存管理器初始化完成")
            
    def cleanup_memory(self, aggressive=False):
        """清理内存"""
        if torch.cuda.is_available():
            # 强制垃圾回收
            gc.collect()
            
            # 清空CUDA缓存
            torch.cuda.empty_cache()
            
            if aggressive:
                # 激进清理模式
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gc.collect()
                
            # 记录内存状态
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                logging.info(f"GPU {i}: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB")
                
    def get_memory_info(self, device_id=0):
        """获取内存信息"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(device_id) / 1024**3
            reserved = torch.cuda.memory_reserved(device_id) / 1024**3
            max_allocated = torch.cuda.max_memory_allocated(device_id) / 1024**3
            
            return {
                'allocated': allocated,
                'reserved': reserved,
                'max_allocated': max_allocated,
                'free': (torch.cuda.get_device_properties(device_id).total_memory / 1024**3) - reserved
            }
        return None

# 全局内存管理器
memory_manager = CUDAMemoryManager()

def patch_training_script():
    """修补训练脚本"""
    script_path = "/nas/factor_forecasting/unified_complete_training_v2_fixed.py"
    
    # 读取原文件
    with open(script_path, 'r') as f:
        content = f.read()
    
    # 在文件开头添加内存管理
    memory_imports = """
# CUDA内存管理补丁
import torch
import torch.cuda
import gc
import os

# 设置CUDA环境
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = (
    'expandable_segments:True,'
    'max_split_size_mb:128,'
    'roundup_power2_divisions:16,'
    'garbage_collection_threshold:0.8'
)

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.85)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def cleanup_cuda_memory():
    \"\"\"清理CUDA内存\"\"\"
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

"""
    
    # 如果还没有添加内存管理代码
    if "cleanup_cuda_memory" not in content:
        # 找到imports部分并添加
        import_pos = content.find("import torch")
        if import_pos != -1:
            content = content[:import_pos] + memory_imports + content[import_pos:]
        
        # 在训练循环中添加内存清理
        # 查找训练循环
        training_patterns = [
            "for batch_idx, batch in enumerate(",
            "for i, batch in enumerate(",
            "for step, batch in enumerate("
        ]
        
        for pattern in training_patterns:
            if pattern in content:
                # 在每个batch后添加内存清理
                lines = content.split('\n')
                new_lines = []
                
                for i, line in enumerate(lines):
                    new_lines.append(line)
                    
                    # 在batch处理后添加内存清理
                    if "loss.backward()" in line:
                        new_lines.append("                    # 内存清理")
                        new_lines.append("                    if batch_idx % 10 == 0:")
                        new_lines.append("                        cleanup_cuda_memory()")
                
                content = '\n'.join(new_lines)
                break
    
    # 写回文件
    with open(script_path, 'w') as f:
        f.write(content)
    
    print("✅ 训练脚本已添加CUDA内存管理补丁")

def apply_memory_fixes():
    """应用所有内存修复"""
    print("🔧 应用CUDA内存修复...")
    
    # 1. 创建内存优化配置
    config = create_memory_optimized_config()
    with open("/nas/factor_forecasting/memory_optimized_config.yaml", "w") as f:
        f.write(config)
    print("✅ 创建内存优化配置文件")
    
    # 2. 修补训练脚本
    patch_training_script()
    
    # 3. 创建启动脚本
    launch_script = """#!/bin/bash
# CUDA内存优化启动脚本

set -e

echo "🚀 启动内存优化的4GPU训练..."

cd /nas/factor_forecasting
source venv/bin/activate

# 设置CUDA环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128,roundup_power2_divisions:16,garbage_collection_threshold:0.8"
export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=1

# 清理旧进程
pkill -f "torchrun.*unified_complete" 2>/dev/null || true
sleep 5

# 清理GPU内存
nvidia-smi --gpu-reset || true
sleep 2

echo "🔍 检查GPU状态..."
nvidia-smi

echo "🚀 启动优化训练..."
nohup torchrun --standalone --nproc_per_node=4 \\
    unified_complete_training_v2_fixed.py \\
    --config memory_optimized_config.yaml \\
    > training_memory_optimized.log 2>&1 &

TRAIN_PID=$!
echo "训练已启动，PID: $TRAIN_PID"

# 等待启动
sleep 10

echo "检查训练状态..."
ps aux | grep unified_complete | grep -v grep || echo "训练进程未找到"

echo "检查GPU使用..."
nvidia-smi

echo "✅ 内存优化训练启动完成"
"""
    
    with open("/nas/factor_forecasting/launch_memory_optimized.sh", "w") as f:
        f.write(launch_script)
    
    print("✅ 创建内存优化启动脚本")
    print("🎉 CUDA内存修复完成!")

if __name__ == "__main__":
    apply_memory_fixes()
'''
    return patch_content

def main():
    """主函数"""
    print("🔧 创建CUDA内存修复脚本...")
    
    # 创建配置
    config = create_memory_optimized_config()
    print("✅ 内存优化配置创建完成")
    
    # 创建补丁
    patch = create_cuda_memory_patch()
    print("✅ CUDA内存补丁创建完成")
    
    return config, patch

if __name__ == "__main__":
    main()
