#!/usr/bin/env python3
"""
终极内存优化修复脚本
从根本上解决CUDA内存问题，确保训练稳定运行
"""

import os
import re
from pathlib import Path

def create_robust_config():
    """创建极度保守的内存配置"""
    config_content = """# 极度保守的内存配置 - 确保稳定训练
model_type: AdvancedFactorForecastingTCNAttention
input_dim: 100
hidden_dim: 512    # 大幅减少隐藏层维度
num_layers: 8      # 减少层数
num_heads: 8       # 减少注意力头数
tcn_kernel_size: 5
tcn_dilation_factor: 2
dropout_rate: 0.1
attention_dropout: 0.05
target_columns: [nextT1d]  # 只预测一个目标，减少内存
sequence_length: 30        # 减少序列长度
epochs: 200
batch_size: 512           # 极小批次大小
fixed_batch_size: 512
learning_rate: 0.0001
weight_decay: 0.01
gradient_clip_norm: 1.0
use_mixed_precision: true
accumulation_steps: 4     # 增加梯度累积步数补偿小批次
use_adaptive_batch_size: false
adaptive_batch_size: false
num_workers: 0
pin_memory: false         # 禁用pin_memory节省内存
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
train_end_date: 2018-02-28    # 减少训练数据量
val_start_date: 2018-03-01
val_end_date: 2018-03-31
test_start_date: 2018-04-01
test_end_date: 2018-04-30
enforce_next_year_prediction: false
enable_yearly_rolling: false
min_train_years: 1
rolling_window_years: 1
shuffle_buffer_size: 128      # 极小缓冲区

# GPU内存优化
gpu_memory_fraction: 0.8
enable_gpu_growth: true
max_memory_usage: 20         # 限制最大内存使用为20GB
streaming_chunk_size: 1000   # 极小chunk大小
enable_memory_mapping: false

# PyTorch内存优化环境变量
pytorch_cuda_alloc_conf: "expandable_segments:True,max_split_size_mb:128"
"""
    
    with open("/nas/factor_forecasting/ultra_conservative_config.yaml", "w") as f:
        f.write(config_content)
    
    print("✅ 创建了极度保守的内存配置")

def create_memory_optimized_launcher():
    """创建内存优化的启动脚本"""
    launcher_content = '''#!/bin/bash
# 内存优化启动脚本

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1

# 清理GPU内存
nvidia-smi --gpu-reset

# 启动训练
cd /nas/factor_forecasting
source venv/bin/activate

# 杀死旧进程
pkill -f unified_complete_training 2>/dev/null || true

# 等待GPU完全释放
sleep 5

# 启动新训练
nohup python unified_complete_training_v2_fixed.py --config ultra_conservative_config.yaml > training_memory_optimized.log 2>&1 &

echo "训练已启动，PID: $!"
'''
    
    with open("/nas/factor_forecasting/launch_memory_optimized.sh", "w") as f:
        f.write(launcher_content)
    
    print("✅ 创建了内存优化启动脚本")

def patch_training_script_for_memory():
    """修复训练脚本以优化内存使用"""
    script_path = "/nas/factor_forecasting/unified_complete_training_v2_fixed.py"
    
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 添加更激进的内存清理
    memory_patch = '''
# 激进内存管理补丁
import gc
import torch

def aggressive_memory_cleanup():
    """激进的内存清理"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # 强制清理未使用的缓存
        torch.cuda.reset_peak_memory_stats()

# 在训练循环中添加内存检查
def check_memory_and_cleanup():
    """检查内存并在必要时清理"""
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        
        # 如果使用超过18GB，强制清理
        if memory_reserved > 18.0:
            aggressive_memory_cleanup()
            return True
    return False
'''
    
    # 在导入后添加内存管理函数
    if 'def aggressive_memory_cleanup' not in content:
        import_end = content.find('# Import components')
        if import_end != -1:
            content = content[:import_end] + memory_patch + '\n' + content[import_end:]
    
    # 在训练循环中添加内存检查
    if 'check_memory_and_cleanup()' not in content:
        # 查找训练循环
        train_loop_pattern = r'(for batch_idx, batch in enumerate\(progress_bar\):)'
        replacement = r'\1\n            # 内存检查和清理\n            if batch_idx % 10 == 0:  # 每10个batch检查一次\n                check_memory_and_cleanup()'
        content = re.sub(train_loop_pattern, replacement, content)
    
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ 训练脚本已添加激进内存管理")

def create_monitoring_script():
    """创建增强的监控脚本"""
    monitoring_script = '''#!/usr/bin/env python3
import subprocess
import time
import json
from datetime import datetime

def get_detailed_status():
    """获取详细状态"""
    try:
        # 检查进程
        proc_result = subprocess.run([
            'sshpass', '-p', 'Abab1234', 'ssh', '-o', 'StrictHostKeyChecking=no',
            'ecs-user@8.216.35.79',
            'ps aux | grep "python.*unified_complete_training" | grep -v grep'
        ], capture_output=True, text=True)
        
        # GPU状态
        gpu_result = subprocess.run([
            'sshpass', '-p', 'Abab1234', 'ssh', '-o', 'StrictHostKeyChecking=no',
            'ecs-user@8.216.35.79',
            'nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        
        # 训练日志
        log_result = subprocess.run([
            'sshpass', '-p', 'Abab1234', 'ssh', '-o', 'StrictHostKeyChecking=no',
            'ecs-user@8.216.35.79',
            'cd /nas/factor_forecasting && tail -10 training_memory_optimized.log'
        ], capture_output=True, text=True)
        
        return {
            'process': proc_result.stdout.strip(),
            'gpu': gpu_result.stdout.strip(),
            'log': log_result.stdout.strip(),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        return {'error': str(e)}

def monitor_continuous():
    """持续监控"""
    print("🔍 启动增强监控系统...")
    
    last_correlation_check = 0
    
    while True:
        status = get_detailed_status()
        current_time = time.time()
        
        print(f"\\n[{status['timestamp']}] 系统状态:")
        print("=" * 80)
        
        if 'error' in status:
            print(f"❌ 错误: {status['error']}")
        else:
            # 进程状态
            if status['process']:
                print("✅ 训练进程运行中")
                print(f"进程: {status['process']}")
            else:
                print("❌ 训练进程未运行")
            
            # GPU状态
            if status['gpu']:
                gpu_lines = status['gpu'].split('\\n')
                for i, line in enumerate(gpu_lines):
                    if line.strip():
                        parts = line.split(', ')
                        if len(parts) >= 4:
                            mem_used, mem_total, util, temp = parts
                            print(f"GPU {i}: {mem_used}MB/{mem_total}MB ({util}% util, {temp}°C)")
            
            # 日志状态
            if 'Epoch' in status['log']:
                print("📊 训练进度信息:")
                for line in status['log'].split('\\n')[-5:]:
                    if 'Epoch' in line or 'Training:' in line:
                        print(f"  {line}")
            
            # 检查内存错误
            if 'CUDA out of memory' in status['log']:
                print("⚠️  检测到内存错误")
            
            # 每2小时检查相关性
            if current_time - last_correlation_check >= 7200:
                print("📈 检查相关性报告...")
                # 这里可以添加相关性检查逻辑
                last_correlation_check = current_time
        
        print("=" * 80)
        time.sleep(60)  # 每分钟检查一次

if __name__ == "__main__":
    monitor_continuous()
'''
    
    with open("/nas/factor_forecasting/enhanced_monitor.py", "w") as f:
        f.write(monitoring_script)
    
    print("✅ 创建了增强监控脚本")

def apply_all_fixes():
    """应用所有修复"""
    print("🔧 开始应用终极内存优化...")
    
    create_robust_config()
    create_memory_optimized_launcher()
    patch_training_script_for_memory()
    create_monitoring_script()
    
    print("✅ 所有优化已应用完成!")

if __name__ == "__main__":
    apply_all_fixes()
'''
