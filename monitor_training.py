#!/usr/bin/env python3
"""
训练监控脚本 - 持续监控训练进度和相关性报告
"""

import time
import subprocess
import re
from datetime import datetime

def get_training_status():
    """获取训练状态"""
    try:
        # 检查训练进程
        result = subprocess.run([
            'sshpass', '-p', 'Abab1234', 'ssh', '-o', 'StrictHostKeyChecking=no',
            'ecs-user@8.216.35.79',
            'ps aux | grep "python.*unified_complete_training" | grep -v grep'
        ], capture_output=True, text=True)
        
        if result.stdout.strip():
            return True, result.stdout.strip()
        else:
            return False, "训练进程未找到"
    except Exception as e:
        return False, f"检查失败: {e}"

def get_gpu_status():
    """获取GPU使用状态"""
    try:
        result = subprocess.run([
            'sshpass', '-p', 'Abab1234', 'ssh', '-o', 'StrictHostKeyChecking=no',
            'ecs-user@8.216.35.79',
            'nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        
        return result.stdout.strip()
    except Exception as e:
        return f"GPU状态获取失败: {e}"

def get_training_log():
    """获取最新训练日志"""
    try:
        result = subprocess.run([
            'sshpass', '-p', 'Abab1234', 'ssh', '-o', 'StrictHostKeyChecking=no',
            'ecs-user@8.216.35.79',
            'cd /nas/factor_forecasting && tail -20 training_completely_fixed.log'
        ], capture_output=True, text=True)
        
        return result.stdout
    except Exception as e:
        return f"日志获取失败: {e}"

def extract_epoch_info(log_text):
    """从日志中提取epoch信息"""
    epoch_pattern = r'Epoch (\d+) Training: (\d+)it \[([^,]+), ([^]]+)\]'
    matches = re.findall(epoch_pattern, log_text)
    
    if matches:
        epoch, iterations, time_elapsed, time_per_it = matches[-1]
        return {
            'epoch': int(epoch),
            'iterations': int(iterations),
            'time_elapsed': time_elapsed,
            'time_per_iteration': time_per_it
        }
    return None

def get_correlation_report():
    """获取相关性报告"""
    try:
        result = subprocess.run([
            'sshpass', '-p', 'Abab1234', 'ssh', '-o', 'StrictHostKeyChecking=no',
            'ecs-user@8.216.35.79',
            'cd /nas/factor_forecasting && find outputs/ -name "*.json" -type f 2>/dev/null | head -5'
        ], capture_output=True, text=True)
        
        return result.stdout.strip()
    except Exception as e:
        return f"相关性报告获取失败: {e}"

def monitor_training():
    """持续监控训练"""
    print("=" * 80)
    print("🚀 因子预测模型训练监控系统")
    print("=" * 80)
    
    last_report_time = 0
    
    while True:
        current_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"\n[{timestamp}] 检查训练状态...")
        
        # 检查训练进程
        is_running, process_info = get_training_status()
        
        if is_running:
            print(f"✅ 训练正在运行")
            print(f"进程信息: {process_info}")
            
            # 获取GPU状态
            gpu_status = get_gpu_status()
            print(f"\n📊 GPU使用状态:")
            for line in gpu_status.split('\n'):
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 5:
                        gpu_id, name, mem_used, mem_total, util = parts
                        print(f"  GPU {gpu_id}: {mem_used}MB/{mem_total}MB ({util}%)")
            
            # 获取训练日志
            log_text = get_training_log()
            epoch_info = extract_epoch_info(log_text)
            
            if epoch_info:
                print(f"\n📈 训练进度:")
                print(f"  当前Epoch: {epoch_info['epoch']}")
                print(f"  完成迭代: {epoch_info['iterations']}")
                print(f"  已用时间: {epoch_info['time_elapsed']}")
                print(f"  每次迭代: {epoch_info['time_per_iteration']}")
            
            # 检查是否有CUDA内存错误
            if "CUDA out of memory" in log_text:
                print("⚠️  检测到CUDA内存不足错误")
                
            # 每2小时检查一次相关性报告
            if current_time - last_report_time >= 7200:  # 2小时
                print(f"\n📊 检查相关性报告...")
                correlation_files = get_correlation_report()
                if correlation_files:
                    print(f"发现相关性报告文件: {correlation_files}")
                last_report_time = current_time
                
        else:
            print(f"❌ 训练进程未运行: {process_info}")
            
        print("-" * 80)
        
        # 等待30秒再次检查
        time.sleep(30)

if __name__ == "__main__":
    try:
        monitor_training()
    except KeyboardInterrupt:
        print("\n\n👋 监控结束")
    except Exception as e:
        print(f"\n❌ 监控错误: {e}")
