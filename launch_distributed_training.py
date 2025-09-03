#!/usr/bin/env python3
"""
4GPU分布式训练启动脚本
彻底解决GPU利用率问题，最大化硬件性能
"""

import os
import subprocess
import time
import signal
import sys
from pathlib import Path

def setup_distributed_environment():
    """设置分布式训练环境变量"""
    env_vars = {
        'MASTER_ADDR': 'localhost',
        'MASTER_PORT': '12355',
        'WORLD_SIZE': '4',
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True,max_split_size_mb:512',
        'OMP_NUM_THREADS': '32',  # 最大化CPU利用率
        'CUDA_VISIBLE_DEVICES': '0,1,2,3',  # 确保所有GPU可见
        'NCCL_DEBUG': 'INFO',  # NCCL调试信息
        'PYTHONUNBUFFERED': '1'
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    print("✅ 分布式环境变量已设置")
    for key, value in env_vars.items():
        print(f"   {key}={value}")

def kill_old_processes():
    """清理旧的训练进程"""
    try:
        subprocess.run(['pkill', '-f', 'unified_complete_training'], check=False)
        time.sleep(3)
        print("✅ 清理了旧的训练进程")
    except Exception as e:
        print(f"清理进程时出错: {e}")

def launch_distributed_training():
    """启动分布式训练"""
    
    print("🚀 启动4GPU分布式训练...")
    
    # 设置环境
    setup_distributed_environment()
    
    # 清理旧进程
    kill_old_processes()
    
    # 构建torchrun命令
    cmd = [
        'torchrun',
        '--nproc_per_node=4',
        '--master_port=12355',
        'unified_complete_training_v2_fixed.py',
        '--config', 'distributed_4gpu_config.yaml'
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    
    # 启动训练
    try:
        # 使用nohup在后台运行
        with open('training_4gpu_distributed.log', 'w') as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd='/nas/factor_forecasting',
                env=os.environ.copy()
            )
        
        print(f"✅ 分布式训练已启动，PID: {process.pid}")
        print("📊 监控命令:")
        print("   tail -f /nas/factor_forecasting/training_4gpu_distributed.log")
        print("   nvidia-smi")
        
        return process.pid
        
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return None

def create_monitoring_script():
    """创建4GPU监控脚本"""
    
    monitoring_script = '''#!/usr/bin/env python3
"""
4GPU分布式训练监控脚本
"""

import subprocess
import time
import json
import re
from datetime import datetime

def get_gpu_status():
    """获取所有GPU状态"""
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        
        gpu_info = []
        for line in result.stdout.strip().split('\\n'):
            if line.strip():
                parts = line.split(', ')
                if len(parts) >= 6:
                    idx, mem_used, mem_total, util, temp, power = parts
                    gpu_info.append({
                        'id': int(idx),
                        'memory_used': int(mem_used),
                        'memory_total': int(mem_total),
                        'utilization': int(util),
                        'temperature': int(temp),
                        'power': float(power)
                    })
        return gpu_info
    except Exception as e:
        print(f"获取GPU状态失败: {e}")
        return []

def get_training_processes():
    """获取训练进程信息"""
    try:
        result = subprocess.run([
            'ps', 'aux'
        ], capture_output=True, text=True)
        
        processes = []
        for line in result.stdout.split('\\n'):
            if 'unified_complete_training' in line and 'grep' not in line:
                processes.append(line.strip())
        
        return processes
    except Exception as e:
        print(f"获取进程信息失败: {e}")
        return []

def get_training_log():
    """获取训练日志"""
    try:
        with open('/nas/factor_forecasting/training_4gpu_distributed.log', 'r') as f:
            lines = f.readlines()
            return ''.join(lines[-30:])
    except Exception as e:
        return f"日志读取失败: {e}"

def extract_training_metrics(log_text):
    """提取训练指标"""
    metrics = {}
    
    # 提取epoch信息
    epoch_pattern = r'Epoch (\\d+) Training: (\\d+)it \\[([^,]+), ([^]]+)\\]'
    matches = re.findall(epoch_pattern, log_text)
    if matches:
        epoch, iterations, time_elapsed, time_per_it = matches[-1]
        metrics.update({
            'current_epoch': int(epoch),
            'iterations': int(iterations),
            'time_elapsed': time_elapsed,
            'time_per_iteration': time_per_it
        })
    
    # 检查分布式训练状态
    if 'DDP' in log_text or 'distributed' in log_text.lower():
        metrics['distributed_active'] = True
    
    # 检查错误
    if 'CUDA out of memory' in log_text:
        metrics['memory_error'] = True
    if 'ERROR' in log_text:
        metrics['has_errors'] = True
    
    return metrics

def monitor_4gpu_training():
    """监控4GPU训练"""
    print("🔍 4GPU分布式训练监控系统")
    print("=" * 80)
    
    last_correlation_check = 0
    
    while True:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\\n[{timestamp}] 4GPU训练状态检查")
        print("-" * 60)
        
        # GPU状态
        gpu_info = get_gpu_status()
        if gpu_info:
            print("📊 GPU状态:")
            total_memory_used = 0
            total_memory_total = 0
            active_gpus = 0
            
            for gpu in gpu_info:
                mem_percent = (gpu['memory_used'] / gpu['memory_total']) * 100
                total_memory_used += gpu['memory_used']
                total_memory_total += gpu['memory_total']
                
                if gpu['utilization'] > 0 or gpu['memory_used'] > 1000:
                    active_gpus += 1
                
                status = "🟢" if gpu['utilization'] > 0 else "🔴"
                print(f"   {status} GPU {gpu['id']}: {gpu['memory_used']}MB/{gpu['memory_total']}MB "
                      f"({mem_percent:.1f}%), {gpu['utilization']}% util, {gpu['temperature']}°C, "
                      f"{gpu['power']:.1f}W")
            
            total_mem_percent = (total_memory_used / total_memory_total) * 100
            print(f"\\n📈 总体状态: {active_gpus}/4 GPU活跃, "
                  f"总内存: {total_memory_used}MB/{total_memory_total}MB ({total_mem_percent:.1f}%)")
        
        # 训练进程
        processes = get_training_processes()
        if processes:
            print(f"\\n✅ 发现 {len(processes)} 个训练进程:")
            for i, proc in enumerate(processes[:4]):  # 最多显示4个
                parts = proc.split()
                if len(parts) >= 11:
                    cpu_usage = parts[2]
                    mem_usage = parts[3]
                    print(f"   进程 {i+1}: CPU {cpu_usage}%, 内存 {mem_usage}%")
        else:
            print("\\n❌ 未发现训练进程")
        
        # 训练指标
        log_text = get_training_log()
        metrics = extract_training_metrics(log_text)
        
        if metrics:
            print("\\n📊 训练进度:")
            if 'current_epoch' in metrics:
                print(f"   当前Epoch: {metrics['current_epoch']}")
                print(f"   完成迭代: {metrics['iterations']}")
                print(f"   已用时间: {metrics['time_elapsed']}")
                print(f"   每次迭代: {metrics['time_per_iteration']}")
            
            if metrics.get('distributed_active'):
                print("   ✅ 分布式训练活跃")
            
            if metrics.get('memory_error'):
                print("   ⚠️  检测到内存错误")
            
            if metrics.get('has_errors'):
                print("   ⚠️  检测到训练错误")
        
        print("=" * 80)
        
        # 每分钟检查一次
        time.sleep(60)

if __name__ == "__main__":
    try:
        monitor_4gpu_training()
    except KeyboardInterrupt:
        print("\\n\\n👋 监控结束")
'''
    
    with open('/nas/factor_forecasting/monitor_4gpu.py', 'w') as f:
        f.write(monitoring_script)
    
    print("✅ 4GPU监控脚本已创建")

if __name__ == "__main__":
    print("🚀 4GPU分布式训练部署系统")
    print("=" * 50)
    
    # 创建监控脚本
    create_monitoring_script()
    
    # 启动分布式训练
    pid = launch_distributed_training()
    
    if pid:
        print("\\n🎉 4GPU分布式训练启动成功!")
        print("\\n📋 后续操作:")
        print("1. 监控训练: python monitor_4gpu.py")
        print("2. 查看日志: tail -f training_4gpu_distributed.log")
        print("3. 检查GPU: watch -n 1 nvidia-smi")
    else:
        print("\\n❌ 启动失败，请检查错误信息")
