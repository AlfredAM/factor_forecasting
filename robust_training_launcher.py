#!/usr/bin/env python3
"""
鲁棒的训练启动器 - 解决进程管理和内存问题
"""
import subprocess
import time
import signal
import os
import sys
from datetime import datetime

def cleanup_old_processes():
    """清理旧的训练进程"""
    print("🧹 清理旧的训练进程...")
    
    ssh_cmd = [
        'sshpass', '-p', 'Abab1234',
        'ssh', '-o', 'StrictHostKeyChecking=no',
        '-o', 'UserKnownHostsFile=/dev/null',
        '-o', 'PreferredAuthentications=password',
        '-o', 'PubkeyAuthentication=no',
        'ecs-user@47.120.46.105'
    ]
    
    # 强制杀死所有相关进程
    cleanup_commands = [
        'pkill -f "unified_complete_training"',
        'pkill -f "torchrun.*unified"',
        'pkill -9 -f "python.*unified"',  # 强制杀死
        'nvidia-smi | grep python | awk \'{print $5}\' | xargs -r kill -9',  # 清理GPU进程
        'sleep 3',  # 等待进程完全退出
        'echo "进程清理完成"'
    ]
    
    for cmd in cleanup_commands:
        full_cmd = ssh_cmd + [cmd]
        try:
            result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=30)
            print(f"  执行: {cmd}")
            if result.stdout.strip():
                print(f"    输出: {result.stdout.strip()}")
        except Exception as e:
            print(f"  警告: {cmd} 执行失败: {e}")

def sync_latest_code():
    """同步最新代码"""
    print("📤 同步最新修复的代码...")
    
    files_to_sync = [
        'src/monitoring/ic_reporter.py',
        'src/unified_complete_training_v2.py',
        'src/training/quantitative_loss.py'
    ]
    
    for file_path in files_to_sync:
        if os.path.exists(file_path):
            scp_cmd = [
                'sshpass', '-p', 'Abab1234',
                'scp', '-o', 'StrictHostKeyChecking=no',
                '-o', 'UserKnownHostsFile=/dev/null',
                '-o', 'PreferredAuthentications=password',
                '-o', 'PubkeyAuthentication=no',
                file_path,
                f'ecs-user@47.120.46.105:/nas/factor_forecasting/{file_path}'
            ]
            
            try:
                result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    print(f"  ✅ 同步: {file_path}")
                else:
                    print(f"  ❌ 同步失败: {file_path}")
            except Exception as e:
                print(f"  ❌ 同步错误: {file_path} - {e}")

def create_robust_config():
    """创建鲁棒的训练配置"""
    print("⚙️ 创建鲁棒训练配置...")
    
    config_content = """# 鲁棒训练配置 - 解决内存和进程问题
batch_size: 1024
fixed_batch_size: 1024
use_adaptive_batch_size: false

# 分布式配置
use_distributed: true
gpu_devices: [0, 1]
world_size: 2

# 内存管理优化
num_workers: 4
prefetch_factor: 2
pin_memory: true
persistent_workers: false  # 避免worker进程累积

# 训练参数
epochs: 3
learning_rate: 0.001
validation_interval: 100
save_interval: 200
early_stopping_patience: 5

# IC报告优化
enable_ic_reporting: true
ic_report_interval: 1800  # 30分钟间隔，更频繁

# 内存监控
max_memory_usage: 0.8
memory_check_interval: 50

# 错误恢复
auto_resume: true
checkpoint_frequency: 5  # 更频繁的检查点

# 目标配置
target_columns: ["intra30m", "nextT1d", "ema1d"]
model_type: "advanced_tcn_attention"
num_stocks: 100000
sequence_length: 60

# 数据配置
data_dir: "/nas/feature_v2_10s"
output_dir: "/nas/factor_forecasting/outputs"
checkpoint_path: "/nas/factor_forecasting/checkpoints"
log_path: "/nas/factor_forecasting/logs"

# 损失函数
loss_config:
  type: "quantitative_correlation"
  alpha: 0.7
  beta: 0.3
"""
    
    # 写入配置文件
    with open('robust_config.yaml', 'w') as f:
        f.write(config_content)
    
    # 上传配置文件
    scp_cmd = [
        'sshpass', '-p', 'Abab1234',
        'scp', '-o', 'StrictHostKeyChecking=no',
        '-o', 'UserKnownHostsFile=/dev/null',
        '-o', 'PreferredAuthentications=password',
        '-o', 'PubkeyAuthentication=no',
        'robust_config.yaml',
        'ecs-user@47.120.46.105:/nas/factor_forecasting/robust_config.yaml'
    ]
    
    try:
        result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("  ✅ 配置文件已上传")
        else:
            print("  ❌ 配置文件上传失败")
    except Exception as e:
        print(f"  ❌ 配置文件上传错误: {e}")

def launch_robust_training():
    """启动鲁棒训练"""
    print("🚀 启动鲁棒训练...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"robust_training_{timestamp}.log"
    
    # 构建启动命令
    launch_cmd = f"""
cd /nas/factor_forecasting && 
source venv/bin/activate && 
export PYTHONPATH=/nas/factor_forecasting && 
export CUDA_LAUNCH_BLOCKING=1 && 
export TORCH_USE_CUDA_DSA=1 && 
nohup torchrun --nproc_per_node=2 --master_port=12356 src/unified_complete_training_v2.py --config robust_config.yaml > logs/{log_file} 2>&1 & 
echo $!
"""
    
    ssh_cmd = [
        'sshpass', '-p', 'Abab1234',
        'ssh', '-o', 'StrictHostKeyChecking=no',
        '-o', 'UserKnownHostsFile=/dev/null',
        '-o', 'PreferredAuthentications=password',
        '-o', 'PubkeyAuthentication=no',
        'ecs-user@47.120.46.105',
        launch_cmd.strip()
    ]
    
    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=60)
        if result.stdout.strip():
            pid = result.stdout.strip().split('\n')[-1]
            print(f"  ✅ 训练已启动，PID: {pid}")
            print(f"  📄 日志文件: logs/{log_file}")
            return pid, log_file
        else:
            print("  ❌ 启动失败")
            return None, None
    except Exception as e:
        print(f"  ❌ 启动错误: {e}")
        return None, None

def monitor_training(log_file, duration=300):
    """监控训练状态"""
    print(f"👁️ 监控训练状态 ({duration}秒)...")
    
    start_time = time.time()
    while time.time() - start_time < duration:
        # 检查GPU状态
        gpu_cmd = [
            'sshpass', '-p', 'Abab1234',
            'ssh', '-o', 'StrictHostKeyChecking=no',
            '-o', 'UserKnownHostsFile=/dev/null',
            '-o', 'PreferredAuthentications=password',
            '-o', 'PubkeyAuthentication=no',
            'ecs-user@47.120.46.105',
            'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'
        ]
        
        try:
            gpu_result = subprocess.run(gpu_cmd, capture_output=True, text=True, timeout=10)
            gpu_status = gpu_result.stdout.strip()
            
            # 检查日志
            log_cmd = [
                'sshpass', '-p', 'Abab1234',
                'ssh', '-o', 'StrictHostKeyChecking=no',
                '-o', 'UserKnownHostsFile=/dev/null',
                '-o', 'PreferredAuthentications=password',
                '-o', 'PubkeyAuthentication=no',
                'ecs-user@47.120.46.105',
                f'cd /nas/factor_forecasting && tail -n 3 logs/{log_file}'
            ]
            
            log_result = subprocess.run(log_cmd, capture_output=True, text=True, timeout=10)
            latest_log = log_result.stdout.strip()
            
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"[{current_time}] GPU: {gpu_status} | 最新: {latest_log.split()[-1] if latest_log else 'N/A'}")
            
        except Exception as e:
            print(f"  监控错误: {e}")
        
        time.sleep(30)

def main():
    """主函数"""
    print("🔧 鲁棒训练启动器")
    print("=" * 50)
    
    # 1. 清理旧进程
    cleanup_old_processes()
    
    # 2. 同步最新代码
    sync_latest_code()
    
    # 3. 创建鲁棒配置
    create_robust_config()
    
    # 4. 启动训练
    pid, log_file = launch_robust_training()
    
    if pid and log_file:
        # 5. 监控训练
        monitor_training(log_file)
        
        print("\n✅ 鲁棒训练已启动并监控完成")
        print(f"📄 查看完整日志: logs/{log_file}")
        print("🔍 使用 nvidia-smi 监控GPU状态")
    else:
        print("\n❌ 训练启动失败")

if __name__ == "__main__":
    main()
