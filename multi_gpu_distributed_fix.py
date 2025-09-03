#!/usr/bin/env python3
"""
多GPU分布式训练修复脚本
从根本上解决GPU利用率不足问题，充分利用4张A10 GPU
"""

import os
import re
from pathlib import Path

def create_distributed_config():
    """创建真正的多GPU分布式配置"""
    config_content = """# 4GPU分布式训练配置 - 充分利用所有硬件资源
model_type: AdvancedFactorForecastingTCNAttention
input_dim: 100
hidden_dim: 1024   # 增大模型以充分利用GPU
num_layers: 16     # 增加层数
num_heads: 32      # 增加注意力头数
tcn_kernel_size: 7
tcn_dilation_factor: 2
dropout_rate: 0.1
attention_dropout: 0.05
target_columns: [intra30m, nextT1d, ema1d]  # 恢复3个目标
sequence_length: 60        # 恢复完整序列长度
epochs: 200
batch_size: 8192          # 大批次充分利用GPU
fixed_batch_size: 8192
learning_rate: 0.0001
weight_decay: 0.01
gradient_clip_norm: 1.0
use_mixed_precision: true
accumulation_steps: 1
use_adaptive_batch_size: false
num_workers: 0            # 保持0避免多进程问题
pin_memory: true
use_distributed: true     # 启用分布式训练
world_size: 4             # 4个GPU
backend: nccl             # NCCL后端用于GPU通信
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
shuffle_buffer_size: 2048

# 分布式训练参数
master_addr: localhost
master_port: 12355
dist_url: env://

# GPU内存优化
pytorch_cuda_alloc_conf: "expandable_segments:True"
"""
    
    return config_content

def create_distributed_launcher():
    """创建分布式训练启动脚本"""
    launcher_content = '''#!/bin/bash
# 4GPU分布式训练启动脚本

set -e

echo "🚀 启动4GPU分布式训练..."

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=4

cd /nas/factor_forecasting
source venv/bin/activate

# 清理旧进程
echo "清理旧进程..."
pkill -f "python.*unified_complete_training" 2>/dev/null || true
pkill -f "torchrun" 2>/dev/null || true

# 等待进程完全清理
sleep 5

# 检查GPU状态
echo "检查GPU状态..."
nvidia-smi

# 启动分布式训练
echo "启动分布式训练..."
torchrun --standalone --nproc_per_node=4 \\
    unified_complete_training_v2_fixed.py \\
    --config distributed_4gpu_config.yaml \\
    > training_distributed_4gpu.log 2>&1 &

TRAIN_PID=$!
echo "分布式训练已启动，主进程PID: $TRAIN_PID"

# 等待几秒钟让训练启动
sleep 10

# 检查进程状态
echo "检查训练进程..."
ps aux | grep python | grep unified || echo "未找到训练进程"

echo "检查GPU使用情况..."
nvidia-smi

echo "✅ 分布式训练启动完成"
'''
    
    return launcher_content

def patch_training_script_for_distributed():
    """修改训练脚本支持torchrun分布式启动"""
    script_path = "/nas/factor_forecasting/unified_complete_training_v2_fixed.py"
    
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 添加torchrun支持
    torchrun_patch = '''
# Torchrun分布式训练支持
import os

def setup_torchrun_distributed():
    """设置torchrun分布式环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # 运行在torchrun环境下
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
        
        # 设置CUDA设备
        torch.cuda.set_device(local_rank)
        
        # 初始化进程组
        if not dist.is_initialized():
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=world_size,
                rank=rank
            )
        
        return rank, world_size, local_rank
    
    return 0, 1, 0
'''
    
    # 在main函数前添加torchrun支持
    if 'def setup_torchrun_distributed' not in content:
        main_func_pos = content.find('def main():')
        if main_func_pos != -1:
            content = content[:main_func_pos] + torchrun_patch + '\n' + content[main_func_pos:]
    
    # 修改main函数以支持torchrun
    main_pattern = r'def main\(\):(.*?)if __name__ == "__main__":'
    
    def replace_main(match):
        main_body = match.group(1)
        new_main = '''def main():
    """Main entry point with torchrun support"""
    parser = argparse.ArgumentParser(description='Unified Complete Training System')
    parser.add_argument('--config', type=str, default='server_optimized_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup torchrun distributed environment
    rank, world_size, local_rank = setup_torchrun_distributed()
    
    print(f"Rank {rank}/{world_size}, Local rank: {local_rank}")
    
    try:
        trainer = UnifiedCompleteTrainer(config, rank, world_size)
        if world_size > 1:
            trainer.setup_distributed()
        trainer.setup_data_loaders()
        trainer.create_model()
        
        if args.resume:
            trainer.load_checkpoint(args.resume)
        
        trainer.train()
        
    except Exception as e:
        print(f"Training failed on rank {rank}: {e}")
        raise
    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()

'''
        return new_main + 'if __name__ == "__main__":'
    
    content = re.sub(main_pattern, replace_main, content, flags=re.DOTALL)
    
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ 训练脚本已修改为支持torchrun分布式训练")

def create_monitoring_script():
    """创建4GPU监控脚本"""
    monitoring_script = '''#!/usr/bin/env python3
"""
4GPU分布式训练监控脚本
持续监控所有GPU使用情况和训练进度
"""

import subprocess
import time
import json
import re
from datetime import datetime

def get_all_gpu_status():
    """获取所有GPU详细状态"""
    try:
        result = subprocess.run([
            'sshpass', '-p', 'Abab1234', 'ssh', '-o', 'StrictHostKeyChecking=no',
            'ecs-user@8.216.35.79',
            'nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu --format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        
        return result.stdout.strip()
    except Exception as e:
        return f"GPU状态获取失败: {e}"

def get_training_processes():
    """获取训练进程信息"""
    try:
        result = subprocess.run([
            'sshpass', '-p', 'Abab1234', 'ssh', '-o', 'StrictHostKeyChecking=no',
            'ecs-user@8.216.35.79',
            'ps aux | grep -E "(python.*unified|torchrun)" | grep -v grep'
        ], capture_output=True, text=True)
        
        return result.stdout.strip()
    except Exception as e:
        return f"进程信息获取失败: {e}"

def get_training_log():
    """获取训练日志"""
    try:
        result = subprocess.run([
            'sshpass', '-p', 'Abab1234', 'ssh', '-o', 'StrictHostKeyChecking=no',
            'ecs-user@8.216.35.79',
            'cd /nas/factor_forecasting && tail -30 training_distributed_4gpu.log 2>/dev/null || echo "日志文件不存在"'
        ], capture_output=True, text=True)
        
        return result.stdout.strip()
    except Exception as e:
        return f"日志获取失败: {e}"

def extract_epoch_time(log_text):
    """提取epoch完成时间"""
    # 查找epoch完成信息
    epoch_pattern = r'Epoch (\d+) completed.*?time: ([\\d\\.]+)s'
    matches = re.findall(epoch_pattern, log_text)
    
    if matches:
        return matches[-1]  # 返回最新的epoch信息
    
    # 查找训练进度信息
    progress_pattern = r'Epoch (\d+) Training: (\d+)it \\[([^,]+), ([^]]+)\\]'
    matches = re.findall(progress_pattern, log_text)
    
    if matches:
        return matches[-1]
    
    return None

def check_correlation_reports():
    """检查相关性报告"""
    try:
        result = subprocess.run([
            'sshpass', '-p', 'Abab1234', 'ssh', '-o', 'StrictHostKeyChecking=no',
            'ecs-user@8.216.35.79',
            'cd /nas/factor_forecasting && find outputs/ -name "*correlation*" -type f -mtime -1 2>/dev/null | head -5'
        ], capture_output=True, text=True)
        
        return result.stdout.strip()
    except Exception as e:
        return f"相关性报告检查失败: {e}"

def monitor_4gpu_training():
    """监控4GPU分布式训练"""
    print("🔍 4GPU分布式训练监控系统启动")
    print("=" * 100)
    
    last_correlation_check = 0
    
    while True:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_time = time.time()
        
        print(f"\\n[{timestamp}] 4GPU分布式训练状态检查")
        print("=" * 100)
        
        # 检查训练进程
        processes = get_training_processes()
        if processes:
            print("✅ 训练进程运行中:")
            for line in processes.split('\\n'):
                if line.strip():
                    print(f"  {line}")
        else:
            print("❌ 未找到训练进程")
        
        # 检查所有GPU状态
        gpu_status = get_all_gpu_status()
        print("\\n📊 所有GPU使用状态:")
        print("GPU | 显存使用    | GPU利用率 | 内存利用率 | 功耗  | 温度")
        print("-" * 70)
        
        total_memory_used = 0
        total_memory_total = 0
        active_gpus = 0
        
        for line in gpu_status.split('\\n'):
            if line.strip() and ',' in line:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 8:
                    gpu_id, name, mem_used, mem_total, util_gpu, util_mem, power, temp = parts
                    total_memory_used += int(mem_used)
                    total_memory_total += int(mem_total)
                    
                    if int(util_gpu) > 0:
                        active_gpus += 1
                        status = "🟢"
                    else:
                        status = "🔴"
                    
                    print(f"{status} {gpu_id}  | {mem_used:>4}MB/{mem_total:>5}MB | {util_gpu:>6}%    | {util_mem:>7}%     | {power:>4}W | {temp:>2}°C")
        
        # 总结GPU使用情况
        total_util = (total_memory_used / total_memory_total) * 100 if total_memory_total > 0 else 0
        print(f"\\n📈 GPU使用总结: {active_gpus}/4个GPU活跃, 总显存利用率: {total_util:.1f}%")
        
        # 检查训练日志
        log_text = get_training_log()
        epoch_info = extract_epoch_time(log_text)
        
        if epoch_info:
            print(f"\\n⏱️  训练进度: {epoch_info}")
        
        # 检查错误
        if 'error' in log_text.lower() or 'failed' in log_text.lower():
            print("⚠️  检测到训练错误，查看日志获取详情")
        
        # 每2小时检查相关性报告
        if current_time - last_correlation_check >= 7200:
            print("\\n📊 检查相关性报告...")
            correlation_files = check_correlation_reports()
            if correlation_files:
                print(f"发现相关性报告: {correlation_files}")
            else:
                print("暂无新的相关性报告")
            last_correlation_check = current_time
        
        print("=" * 100)
        time.sleep(60)  # 每分钟检查一次

if __name__ == "__main__":
    try:
        monitor_4gpu_training()
    except KeyboardInterrupt:
        print("\\n\\n👋 监控结束")
    except Exception as e:
        print(f"\\n❌ 监控错误: {e}")
'''
    
    return monitoring_script

def apply_distributed_fixes():
    """应用所有分布式修复"""
    print("🚀 开始配置4GPU分布式训练...")
    
    # 创建配置文件
    config_content = create_distributed_config()
    with open("/nas/factor_forecasting/distributed_4gpu_config.yaml", "w") as f:
        f.write(config_content)
    print("✅ 创建了4GPU分布式配置")
    
    # 创建启动脚本
    launcher_content = create_distributed_launcher()
    with open("/nas/factor_forecasting/launch_4gpu_distributed.sh", "w") as f:
        f.write(launcher_content)
    print("✅ 创建了分布式启动脚本")
    
    # 修改训练脚本
    patch_training_script_for_distributed()
    
    # 创建监控脚本
    monitoring_content = create_monitoring_script()
    with open("/nas/factor_forecasting/monitor_4gpu.py", "w") as f:
        f.write(monitoring_content)
    print("✅ 创建了4GPU监控脚本")
    
    print("🎉 4GPU分布式训练配置完成!")
    print("使用以下命令启动:")
    print("bash /nas/factor_forecasting/launch_4gpu_distributed.sh")

if __name__ == "__main__":
    apply_distributed_fixes()
"""
