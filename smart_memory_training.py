#!/usr/bin/env python3
"""
智能内存管理训练脚本
动态调整批次大小，最大化硬件利用率同时避免内存溢出
"""

import os
import sys
import time
import psutil
import subprocess
from pathlib import Path

def get_gpu_memory_info():
    """获取GPU内存信息"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_info = []
            for line in lines:
                used, total = map(int, line.split(', '))
                gpu_info.append({'used': used, 'total': total, 'free': total - used})
            return gpu_info
    except Exception as e:
        print(f"获取GPU信息失败: {e}")
    return []

def get_optimal_batch_size():
    """根据当前内存状况计算最优批次大小"""
    gpu_info = get_gpu_memory_info()
    if not gpu_info:
        return 512  # 保守默认值
    
    # 使用第一个GPU的信息
    gpu = gpu_info[0]
    free_memory_gb = gpu['free'] / 1024  # 转换为GB
    
    # 根据可用内存动态计算批次大小
    if free_memory_gb > 15:
        return 2048
    elif free_memory_gb > 10:
        return 1024
    elif free_memory_gb > 5:
        return 512
    else:
        return 256

def create_launcher_script():
    """创建智能启动脚本"""
    launcher_content = f'''#!/bin/bash
# 智能内存管理启动脚本

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export CUDA_LAUNCH_BLOCKING=0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=32  # 最大化CPU利用率

cd /nas/factor_forecasting
source venv/bin/activate

# 清理旧进程
pkill -f unified_complete_training 2>/dev/null || true
sleep 3

# 清理GPU缓存
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

echo "启动智能内存管理训练..."
nohup python unified_complete_training_v2_fixed.py --config balanced_high_performance_config.yaml > training_smart_memory.log 2>&1 &

echo "训练已启动，PID: $!"
echo "监控命令: tail -f training_smart_memory.log"
'''
    
    return launcher_content

def create_continuous_monitor():
    """创建持续监控脚本"""
    monitor_content = '''#!/usr/bin/env python3
"""
持续训练监控和相关性报告脚本
"""

import subprocess
import time
import re
import json
from datetime import datetime
from pathlib import Path

class TrainingMonitor:
    def __init__(self):
        self.last_correlation_check = 0
        self.epoch_times = []
        
    def get_training_status(self):
        """获取训练状态"""
        try:
            # 检查进程
            proc_result = subprocess.run([
                'ps', 'aux'
            ], capture_output=True, text=True)
            
            training_process = None
            for line in proc_result.stdout.split('\\n'):
                if 'unified_complete_training' in line and 'grep' not in line:
                    training_process = line.strip()
                    break
            
            return training_process
        except Exception as e:
            return f"检查失败: {e}"
    
    def get_gpu_status(self):
        """获取GPU状态"""
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)
            
            return result.stdout.strip()
        except Exception as e:
            return f"GPU状态获取失败: {e}"
    
    def get_training_log(self):
        """获取训练日志"""
        try:
            with open('/nas/factor_forecasting/training_smart_memory.log', 'r') as f:
                lines = f.readlines()
                return ''.join(lines[-50:])  # 最后50行
        except Exception as e:
            return f"日志读取失败: {e}"
    
    def extract_epoch_info(self, log_text):
        """提取epoch信息"""
        # 查找最新的epoch信息
        epoch_pattern = r'Epoch (\\d+) Training: (\\d+)it \\[([^,]+), ([^]]+)\\]'
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
    
    def check_correlations(self):
        """检查相关性报告"""
        try:
            output_dir = Path('/nas/factor_forecasting/outputs')
            if output_dir.exists():
                json_files = list(output_dir.glob('**/*.json'))
                if json_files:
                    # 读取最新的相关性报告
                    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
                    with open(latest_file, 'r') as f:
                        data = json.load(f)
                    
                    correlations = {}
                    if 'correlations' in data:
                        for target, corr_data in data['correlations'].items():
                            if 'in_sample_ic' in corr_data:
                                correlations[f'{target}_in_sample'] = corr_data['in_sample_ic']
                            if 'out_sample_ic' in corr_data:
                                correlations[f'{target}_out_sample'] = corr_data['out_sample_ic']
                    
                    return correlations
        except Exception as e:
            print(f"相关性检查失败: {e}")
        
        return {}
    
    def monitor(self):
        """主监控循环"""
        print("🚀 智能训练监控系统启动")
        print("=" * 80)
        
        while True:
            current_time = time.time()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            print(f"\\n[{timestamp}] 系统状态检查")
            print("-" * 60)
            
            # 检查训练进程
            training_status = self.get_training_status()
            if training_status and 'python' in training_status:
                print("✅ 训练进程运行中")
                # 提取CPU和内存使用率
                parts = training_status.split()
                if len(parts) >= 11:
                    cpu_usage = parts[2]
                    mem_usage = parts[3]
                    print(f"   CPU: {cpu_usage}%, 内存: {mem_usage}%")
            else:
                print("❌ 训练进程未运行")
            
            # GPU状态
            gpu_status = self.get_gpu_status()
            print("\\n📊 GPU状态:")
            for line in gpu_status.split('\\n'):
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 5:
                        gpu_id, mem_used, mem_total, util, temp = parts
                        mem_percent = (int(mem_used) / int(mem_total)) * 100
                        print(f"   GPU {gpu_id}: {mem_used}MB/{mem_total}MB ({mem_percent:.1f}%), 利用率: {util}%, 温度: {temp}°C")
            
            # 训练进度
            log_text = self.get_training_log()
            epoch_info = self.extract_epoch_info(log_text)
            
            if epoch_info:
                print("\\n📈 训练进度:")
                print(f"   当前Epoch: {epoch_info['epoch']}")
                print(f"   完成迭代: {epoch_info['iterations']}")
                print(f"   已用时间: {epoch_info['time_elapsed']}")
                print(f"   每次迭代: {epoch_info['time_per_iteration']}")
                
                # 估算epoch完成时间
                if epoch_info['epoch'] == 0:  # 第一个epoch
                    time_per_it_seconds = self.parse_time_to_seconds(epoch_info['time_per_iteration'])
                    if time_per_it_seconds and epoch_info['iterations'] > 10:
                        # 估算总迭代数（假设数据量固定）
                        estimated_total_iterations = epoch_info['iterations'] * 2  # 粗略估计
                        remaining_iterations = estimated_total_iterations - epoch_info['iterations']
                        estimated_remaining_time = remaining_iterations * time_per_it_seconds
                        print(f"   预计剩余时间: {self.seconds_to_time_str(estimated_remaining_time)}")
            
            # 检查内存错误
            if 'CUDA out of memory' in log_text:
                print("\\n⚠️  检测到CUDA内存不足")
            elif 'ERROR' in log_text:
                print("\\n⚠️  检测到训练错误")
            
            # 每2小时检查相关性
            if current_time - self.last_correlation_check >= 7200:
                print("\\n📊 检查相关性报告...")
                correlations = self.check_correlations()
                if correlations:
                    print("   最新相关性数据:")
                    for target, corr in correlations.items():
                        print(f"     {target}: {corr:.4f}")
                else:
                    print("   暂无相关性数据")
                
                self.last_correlation_check = current_time
            
            print("=" * 80)
            
            # 每30秒检查一次
            time.sleep(30)
    
    def parse_time_to_seconds(self, time_str):
        """解析时间字符串为秒数"""
        try:
            if 's/it' in time_str:
                return float(time_str.replace('s/it', ''))
        except:
            pass
        return None
    
    def seconds_to_time_str(self, seconds):
        """转换秒数为时间字符串"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

if __name__ == "__main__":
    monitor = TrainingMonitor()
    try:
        monitor.monitor()
    except KeyboardInterrupt:
        print("\\n\\n👋 监控结束")
'''
    
    return monitor_content

def deploy_smart_training():
    """部署智能训练系统"""
    print("🚀 部署智能内存管理训练系统...")
    
    # 创建启动脚本
    launcher_script = create_launcher_script()
    with open('/tmp/smart_launcher.sh', 'w') as f:
        f.write(launcher_script)
    
    # 创建监控脚本
    monitor_script = create_continuous_monitor()
    with open('/tmp/smart_monitor.py', 'w') as f:
        f.write(monitor_script)
    
    print("✅ 脚本创建完成")
    return True

if __name__ == "__main__":
    deploy_smart_training()
