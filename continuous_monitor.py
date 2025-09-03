#!/usr/bin/env python3
"""
持续监控脚本 - 实时监控训练状态和相关性报告
"""

import subprocess
import time
import re
import json
from datetime import datetime
from pathlib import Path

def get_training_status():
    """获取训练状态"""
    try:
        result = subprocess.run([
            'sshpass', '-p', 'Abab1234', 'ssh', '-o', 'StrictHostKeyChecking=no',
            'ecs-user@8.216.35.79',
            'ps aux | grep "python.*unified_complete_training" | grep -v grep'
        ], capture_output=True, text=True)
        
        return result.stdout.strip() if result.stdout.strip() else None
    except Exception as e:
        return f"检查失败: {e}"

def get_gpu_status():
    """获取GPU状态"""
    try:
        result = subprocess.run([
            'sshpass', '-p', 'Abab1234', 'ssh', '-o', 'StrictHostKeyChecking=no',
            'ecs-user@8.216.35.79',
            'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        
        return result.stdout.strip()
    except Exception as e:
        return f"GPU状态获取失败: {e}"

def get_training_log():
    """获取训练日志"""
    try:
        result = subprocess.run([
            'sshpass', '-p', 'Abab1234', 'ssh', '-o', 'StrictHostKeyChecking=no',
            'ecs-user@8.216.35.79',
            'cd /nas/factor_forecasting && tail -30 training_smart_memory.log'
        ], capture_output=True, text=True)
        
        return result.stdout
    except Exception as e:
        return f"日志获取失败: {e}"

def extract_epoch_info(log_text):
    """提取epoch信息"""
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

def check_correlations():
    """检查相关性报告"""
    try:
        result = subprocess.run([
            'sshpass', '-p', 'Abab1234', 'ssh', '-o', 'StrictHostKeyChecking=no',
            'ecs-user@8.216.35.79',
            'find /nas/factor_forecasting/outputs -name "*.json" -type f 2>/dev/null | head -3'
        ], capture_output=True, text=True)
        
        files = result.stdout.strip().split('\n')
        correlations = {}
        
        for file_path in files:
            if file_path and file_path.endswith('.json'):
                # 读取相关性文件
                cat_result = subprocess.run([
                    'sshpass', '-p', 'Abab1234', 'ssh', '-o', 'StrictHostKeyChecking=no',
                    'ecs-user@8.216.35.79',
                    f'cat {file_path}'
                ], capture_output=True, text=True)
                
                if cat_result.returncode == 0:
                    try:
                        data = json.loads(cat_result.stdout)
                        if 'correlations' in data:
                            for target, corr_data in data['correlations'].items():
                                if 'in_sample_ic' in corr_data:
                                    correlations[f'{target}_in_sample'] = corr_data['in_sample_ic']
                                if 'out_sample_ic' in corr_data:
                                    correlations[f'{target}_out_sample'] = corr_data['out_sample_ic']
                        break  # 只读取最新的一个文件
                    except json.JSONDecodeError:
                        continue
        
        return correlations
    except Exception as e:
        print(f"相关性检查失败: {e}")
        return {}

def parse_time_to_seconds(time_str):
    """解析时间字符串为秒数"""
    try:
        if 's/it' in time_str:
            return float(time_str.replace('s/it', '').strip())
        elif ':' in time_str:
            parts = time_str.split(':')
            if len(parts) == 2:  # MM:SS
                return int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:  # HH:MM:SS
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except:
        pass
    return None

def seconds_to_time_str(seconds):
    """转换秒数为时间字符串"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def main():
    """主监控循环"""
    print("🚀 智能训练监控系统启动")
    print("=" * 80)
    
    last_correlation_check = 0
    epoch_start_times = {}
    
    while True:
        current_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"\n[{timestamp}] 系统状态检查")
        print("-" * 60)
        
        # 检查训练进程
        training_status = get_training_status()
        if training_status and 'python' in training_status:
            print("✅ 训练进程运行中")
            # 提取CPU和内存使用率
            try:
                parts = training_status.split()
                if len(parts) >= 11:
                    cpu_usage = parts[2]
                    mem_usage = parts[3]
                    runtime = parts[9]
                    print(f"   CPU: {cpu_usage}%, 内存: {mem_usage}%, 运行时间: {runtime}")
            except:
                pass
        else:
            print("❌ 训练进程未运行")
        
        # GPU状态
        gpu_status = get_gpu_status()
        print("\n📊 GPU状态:")
        for line in gpu_status.split('\n'):
            if line.strip():
                parts = line.split(', ')
                if len(parts) >= 5:
                    gpu_id, mem_used, mem_total, util, temp = parts
                    mem_percent = (int(mem_used) / int(mem_total)) * 100
                    print(f"   GPU {gpu_id}: {mem_used}MB/{mem_total}MB ({mem_percent:.1f}%), 利用率: {util}%, 温度: {temp}°C")
        
        # 训练进度
        log_text = get_training_log()
        epoch_info = extract_epoch_info(log_text)
        
        if epoch_info:
            print("\n📈 训练进度:")
            print(f"   当前Epoch: {epoch_info['epoch']}")
            print(f"   完成迭代: {epoch_info['iterations']}")
            print(f"   已用时间: {epoch_info['time_elapsed']}")
            print(f"   每次迭代: {epoch_info['time_per_iteration']}")
            
            # 记录epoch开始时间
            epoch_key = epoch_info['epoch']
            if epoch_key not in epoch_start_times:
                epoch_start_times[epoch_key] = current_time
            
            # 估算epoch完成时间
            if epoch_info['epoch'] == 0 and epoch_info['iterations'] > 20:
                time_per_it_seconds = parse_time_to_seconds(epoch_info['time_per_iteration'])
                if time_per_it_seconds:
                    # 假设每个epoch大约需要相同数量的迭代
                    estimated_total_iterations = epoch_info['iterations'] * 1.5  # 保守估计
                    remaining_iterations = max(0, estimated_total_iterations - epoch_info['iterations'])
                    estimated_remaining_time = remaining_iterations * time_per_it_seconds
                    print(f"   预计剩余时间: {seconds_to_time_str(estimated_remaining_time)}")
        
        # 检查错误
        error_count = log_text.count('CUDA out of memory')
        if error_count > 0:
            print(f"\n⚠️  检测到 {error_count} 个CUDA内存错误")
        
        general_errors = log_text.count('ERROR:')
        if general_errors > error_count:
            print(f"⚠️  检测到 {general_errors - error_count} 个其他错误")
        
        # 每2小时检查相关性
        if current_time - last_correlation_check >= 7200:
            print("\n📊 检查相关性报告...")
            correlations = check_correlations()
            if correlations:
                print("   📈 最新相关性数据:")
                for target, corr in correlations.items():
                    print(f"     {target}: {corr:.4f}")
            else:
                print("   暂无相关性数据")
            
            last_correlation_check = current_time
        
        print("=" * 80)
        
        # 每60秒检查一次
        time.sleep(60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 监控结束")