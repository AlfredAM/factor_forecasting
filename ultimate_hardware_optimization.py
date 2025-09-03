#!/usr/bin/env python3
"""
终极硬件优化脚本
彻底解决GPU利用率、内存碎片化和分布式训练问题
"""

import os
import sys
import subprocess
import time
from pathlib import Path

class HardwareOptimizer:
    def __init__(self):
        self.gpu_count = 4
        self.total_memory_per_gpu = 23028  # MB
        self.safe_memory_per_gpu = 20000   # MB，留出安全边界
        
    def analyze_current_problems(self):
        """分析当前问题"""
        print("🔍 深度分析当前硬件利用率问题...")
        
        problems = []
        
        # 检查GPU内存碎片化
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for i, line in enumerate(lines):
                    used, total = map(int, line.split(', '))
                    usage_percent = (used / total) * 100
                    if usage_percent > 90:
                        problems.append(f"GPU {i}: 内存使用率过高 ({usage_percent:.1f}%)")
                    elif usage_percent > 30 and usage_percent < 50:
                        problems.append(f"GPU {i}: 可能存在内存碎片化")
        except Exception as e:
            problems.append(f"GPU检查失败: {e}")
        
        # 检查进程分布
        try:
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            python_processes = [line for line in result.stdout.split('\n') 
                              if 'unified_complete_training' in line and 'grep' not in line]
            if len(python_processes) > 1:
                problems.append(f"检测到{len(python_processes)}个训练进程，可能存在资源竞争")
        except Exception as e:
            problems.append(f"进程检查失败: {e}")
        
        return problems
    
    def create_optimized_distributed_config(self):
        """创建优化的分布式配置"""
        config_content = f"""# 优化的4GPU分布式配置
model_type: AdvancedFactorForecastingTCNAttention
input_dim: 100
hidden_dim: 512    # 减小模型大小以适应4GPU分布式
num_layers: 8      # 适中层数
num_heads: 8       # 适中注意力头数
tcn_kernel_size: 3
tcn_dilation_factor: 2
dropout_rate: 0.1
attention_dropout: 0.05
target_columns: [intra30m, nextT1d, ema1d]
sequence_length: 30        # 减小序列长度
epochs: 200
batch_size: 512           # 每GPU批次大小
fixed_batch_size: 512
learning_rate: 0.0001
weight_decay: 0.01
gradient_clip_norm: 1.0
use_mixed_precision: true
accumulation_steps: 1     # 分布式训练不需要梯度累积
use_adaptive_batch_size: false
num_workers: 0
pin_memory: false
use_distributed: true     # 启用分布式训练
auto_resume: true
log_level: INFO
ic_report_interval: 7200
enable_ic_reporting: true
checkpoint_frequency: 10
save_all_checkpoints: false
output_dir: /nas/factor_forecasting/outputs
data_dir: /nas/feature_v2_10s
train_start_date: 2018-01-02
train_end_date: 2018-10-31   # 使用前10个月数据
val_start_date: 2018-11-01
val_end_date: 2018-11-30
test_start_date: 2018-12-01
test_end_date: 2018-12-31
enforce_next_year_prediction: false
enable_yearly_rolling: false
min_train_years: 1
rolling_window_years: 1
shuffle_buffer_size: 256

# 分布式训练参数
world_size: 4
backend: nccl

# 内存优化参数
max_memory_usage_per_gpu: 18    # 每GPU最大内存使用
streaming_chunk_size: 10000     # 小chunk避免内存峰值
enable_memory_mapping: false    # 禁用内存映射减少碎片
enable_gradient_checkpointing: true

# PyTorch优化
torch_compile: false
enable_flash_attention: false
use_channels_last: false  # 分布式训练中可能有问题
"""
        
        with open('/tmp/optimized_4gpu_config.yaml', 'w') as f:
            f.write(config_content)
        
        return '/tmp/optimized_4gpu_config.yaml'
    
    def create_memory_optimized_training_script(self):
        """创建内存优化的训练脚本补丁"""
        patch_content = '''
# 内存优化补丁
import torch
import gc
import os

# 设置PyTorch内存管理
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256,expandable_segments:True'

def optimize_memory_settings():
    """优化内存设置"""
    if torch.cuda.is_available():
        # 启用内存池
        torch.cuda.set_per_process_memory_fraction(0.85)  # 每个进程最多使用85%显存
        
        # 清空缓存
        torch.cuda.empty_cache()
        
        # 设置内存增长模式
        torch.cuda.memory.set_per_process_memory_fraction(0.85)

def aggressive_cleanup():
    """激进的内存清理"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # 重置峰值内存统计
        torch.cuda.reset_peak_memory_stats()

# 在训练开始前调用
optimize_memory_settings()
'''
        
        return patch_content
    
    def create_distributed_launcher(self):
        """创建分布式训练启动器"""
        launcher_content = f'''#!/bin/bash
# 4GPU分布式训练启动器

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256,expandable_segments:True"
export NCCL_DEBUG=INFO
export NCCL_TREE_THRESHOLD=0
export OMP_NUM_THREADS=8  # 限制CPU线程避免过度竞争

cd /nas/factor_forecasting
source venv/bin/activate

# 清理旧进程
pkill -f unified_complete_training 2>/dev/null || true
pkill -f torchrun 2>/dev/null || true
sleep 5

# 清理GPU内存
python -c "import torch; [torch.cuda.empty_cache() for _ in range(4) if torch.cuda.is_available()]" 2>/dev/null || true

echo "启动4GPU分布式训练..."
echo "配置文件: optimized_4gpu_config.yaml"
echo "数据范围: 2018年前10个月"

# 使用torchrun启动分布式训练
torchrun \\
    --nproc_per_node=4 \\
    --master_port=12355 \\
    unified_complete_training_v2_fixed.py \\
    --config optimized_4gpu_config.yaml \\
    > training_4gpu_optimized.log 2>&1 &

TRAIN_PID=$!
echo "训练已启动，主进程PID: $TRAIN_PID"
echo "日志文件: training_4gpu_optimized.log"
echo "监控命令: tail -f training_4gpu_optimized.log"

# 等待进程启动
sleep 10
echo "检查进程状态..."
ps aux | grep unified_complete_training | grep -v grep || echo "警告: 训练进程可能未正常启动"
'''
        
        with open('/tmp/launch_4gpu_optimized.sh', 'w') as f:
            f.write(launcher_content)
        
        os.chmod('/tmp/launch_4gpu_optimized.sh', 0o755)
        return '/tmp/launch_4gpu_optimized.sh'
    
    def create_comprehensive_monitor(self):
        """创建综合监控脚本"""
        monitor_content = '''#!/usr/bin/env python3
"""
综合硬件利用率监控
实时监控4GPU分布式训练的硬件利用率和相关性报告
"""

import subprocess
import time
import json
import re
from datetime import datetime
from pathlib import Path

class ComprehensiveMonitor:
    def __init__(self):
        self.last_correlation_check = 0
        self.gpu_count = 4
        
    def get_detailed_gpu_status(self):
        """获取详细GPU状态"""
        try:
            # GPU利用率和内存
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)
            
            gpu_stats = []
            if result.returncode == 0:
                for line in result.stdout.strip().split('\\n'):
                    if line.strip():
                        parts = line.split(', ')
                        if len(parts) >= 7:
                            gpu_stats.append({
                                'id': parts[0],
                                'name': parts[1],
                                'mem_used': int(parts[2]),
                                'mem_total': int(parts[3]),
                                'gpu_util': int(parts[4]),
                                'mem_util': int(parts[5]) if parts[5] != '[N/A]' else 0,
                                'temp': int(parts[6]),
                                'power': float(parts[7]) if parts[7] != '[N/A]' else 0
                            })
            
            return gpu_stats
        except Exception as e:
            return [{'error': str(e)}]
    
    def get_training_processes(self):
        """获取训练进程信息"""
        try:
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            processes = []
            
            for line in result.stdout.split('\\n'):
                if 'unified_complete_training' in line and 'grep' not in line:
                    parts = line.split()
                    if len(parts) >= 11:
                        processes.append({
                            'pid': parts[1],
                            'cpu_percent': parts[2],
                            'mem_percent': parts[3],
                            'command': ' '.join(parts[10:])
                        })
            
            return processes
        except Exception as e:
            return [{'error': str(e)}]
    
    def get_system_resources(self):
        """获取系统资源使用情况"""
        try:
            # CPU信息
            cpu_result = subprocess.run(['top', '-bn1'], capture_output=True, text=True)
            cpu_usage = 0
            for line in cpu_result.stdout.split('\\n'):
                if 'Cpu(s):' in line:
                    # 提取CPU使用率
                    match = re.search(r'(\\d+\\.\\d+)%us', line)
                    if match:
                        cpu_usage = float(match.group(1))
                    break
            
            # 内存信息
            mem_result = subprocess.run(['free', '-m'], capture_output=True, text=True)
            mem_info = {}
            for line in mem_result.stdout.split('\\n'):
                if line.startswith('Mem:'):
                    parts = line.split()
                    mem_info = {
                        'total': int(parts[1]),
                        'used': int(parts[2]),
                        'free': int(parts[3]),
                        'usage_percent': (int(parts[2]) / int(parts[1])) * 100
                    }
                    break
            
            return {'cpu_usage': cpu_usage, 'memory': mem_info}
        except Exception as e:
            return {'error': str(e)}
    
    def get_training_progress(self):
        """获取训练进度"""
        try:
            log_file = Path('/nas/factor_forecasting/training_4gpu_optimized.log')
            if log_file.exists():
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    recent_lines = ''.join(lines[-100:])  # 最后100行
                
                # 提取epoch信息
                epoch_pattern = r'Epoch (\\d+) Training: (\\d+)it \\[([^,]+), ([^]]+)\\]'
                matches = re.findall(epoch_pattern, recent_lines)
                
                if matches:
                    epoch, iterations, time_elapsed, time_per_it = matches[-1]
                    return {
                        'epoch': int(epoch),
                        'iterations': int(iterations),
                        'time_elapsed': time_elapsed,
                        'time_per_iteration': time_per_it,
                        'has_errors': 'ERROR' in recent_lines,
                        'memory_errors': 'CUDA out of memory' in recent_lines
                    }
            
            return {'status': 'no_progress_found'}
        except Exception as e:
            return {'error': str(e)}
    
    def check_correlations(self):
        """检查相关性报告"""
        try:
            output_dir = Path('/nas/factor_forecasting/outputs')
            if output_dir.exists():
                json_files = list(output_dir.glob('**/*.json'))
                if json_files:
                    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
                    with open(latest_file, 'r') as f:
                        data = json.load(f)
                    
                    correlations = {}
                    timestamp = data.get('timestamp', 'unknown')
                    
                    if 'correlations' in data:
                        for target, corr_data in data['correlations'].items():
                            if isinstance(corr_data, dict):
                                for metric, value in corr_data.items():
                                    if 'ic' in metric.lower():
                                        correlations[f'{target}_{metric}'] = value
                    
                    return {'correlations': correlations, 'timestamp': timestamp}
        except Exception as e:
            return {'error': str(e)}
        
        return {'status': 'no_correlations_found'}
    
    def monitor_continuously(self):
        """持续监控"""
        print("🚀 启动4GPU分布式训练综合监控系统")
        print("=" * 100)
        
        while True:
            current_time = time.time()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            print(f"\\n[{timestamp}] 硬件利用率监控报告")
            print("=" * 100)
            
            # GPU状态
            gpu_stats = self.get_detailed_gpu_status()
            print("\\n📊 GPU利用率详情:")
            total_gpu_util = 0
            total_mem_util = 0
            
            for gpu in gpu_stats:
                if 'error' not in gpu:
                    mem_percent = (gpu['mem_used'] / gpu['mem_total']) * 100
                    print(f"   GPU {gpu['id']}: 计算利用率 {gpu['gpu_util']}%, "
                          f"显存 {gpu['mem_used']}MB/{gpu['mem_total']}MB ({mem_percent:.1f}%), "
                          f"温度 {gpu['temp']}°C, 功耗 {gpu['power']:.1f}W")
                    total_gpu_util += gpu['gpu_util']
                    total_mem_util += mem_percent
            
            if len(gpu_stats) > 0 and 'error' not in gpu_stats[0]:
                avg_gpu_util = total_gpu_util / len(gpu_stats)
                avg_mem_util = total_mem_util / len(gpu_stats)
                print(f"   平均GPU利用率: {avg_gpu_util:.1f}%, 平均显存利用率: {avg_mem_util:.1f}%")
            
            # 系统资源
            system_resources = self.get_system_resources()
            if 'error' not in system_resources:
                print(f"\\n💻 系统资源:")
                print(f"   CPU利用率: {system_resources['cpu_usage']:.1f}%")
                if 'memory' in system_resources:
                    mem = system_resources['memory']
                    print(f"   内存使用: {mem['used']}MB/{mem['total']}MB ({mem['usage_percent']:.1f}%)")
            
            # 训练进程
            processes = self.get_training_processes()
            print(f"\\n🏃 训练进程状态:")
            if processes:
                for proc in processes:
                    if 'error' not in proc:
                        print(f"   PID {proc['pid']}: CPU {proc['cpu_percent']}%, 内存 {proc['mem_percent']}%")
            else:
                print("   ❌ 未检测到训练进程")
            
            # 训练进度
            progress = self.get_training_progress()
            if 'error' not in progress and 'epoch' in progress:
                print(f"\\n📈 训练进度:")
                print(f"   当前Epoch: {progress['epoch']}")
                print(f"   完成迭代: {progress['iterations']}")
                print(f"   已用时间: {progress['time_elapsed']}")
                print(f"   每次迭代: {progress['time_per_iteration']}")
                
                if progress.get('memory_errors'):
                    print("   ⚠️  检测到内存错误")
                elif progress.get('has_errors'):
                    print("   ⚠️  检测到训练错误")
                else:
                    print("   ✅ 训练正常进行")
            
            # 每2小时检查相关性
            if current_time - self.last_correlation_check >= 7200:
                print(f"\\n📊 相关性报告检查...")
                correlation_data = self.check_correlations()
                
                if 'correlations' in correlation_data:
                    print(f"   报告时间: {correlation_data['timestamp']}")
                    print("   相关性数据:")
                    for metric, value in correlation_data['correlations'].items():
                        print(f"     {metric}: {value:.4f}")
                else:
                    print("   暂无相关性数据")
                
                self.last_correlation_check = current_time
            
            print("=" * 100)
            
            # 每1分钟检查一次
            time.sleep(60)

if __name__ == "__main__":
    monitor = ComprehensiveMonitor()
    try:
        monitor.monitor_continuously()
    except KeyboardInterrupt:
        print("\\n\\n👋 监控结束")
'''
        
        with open('/tmp/comprehensive_monitor.py', 'w') as f:
            f.write(monitor_content)
        
        return '/tmp/comprehensive_monitor.py'
    
    def deploy_optimization(self):
        """部署优化方案"""
        print("🚀 开始部署终极硬件优化方案...")
        
        # 分析当前问题
        problems = self.analyze_current_problems()
        if problems:
            print("发现的问题:")
            for problem in problems:
                print(f"  ❌ {problem}")
        
        # 创建配置文件
        config_path = self.create_optimized_distributed_config()
        print(f"✅ 创建优化配置: {config_path}")
        
        # 创建启动脚本
        launcher_path = self.create_distributed_launcher()
        print(f"✅ 创建分布式启动器: {launcher_path}")
        
        # 创建监控脚本
        monitor_path = self.create_comprehensive_monitor()
        print(f"✅ 创建综合监控: {monitor_path}")
        
        print("\\n🎯 优化方案部署完成!")
        print("\\n下一步操作:")
        print("1. 上传配置文件到服务器")
        print("2. 停止当前训练进程")
        print("3. 启动优化的4GPU分布式训练")
        print("4. 启动综合监控")
        
        return {
            'config': config_path,
            'launcher': launcher_path,
            'monitor': monitor_path
        }

if __name__ == "__main__":
    optimizer = HardwareOptimizer()
    result = optimizer.deploy_optimization()
    print(f"\\n📁 生成的文件:")
    for key, path in result.items():
        print(f"  {key}: {path}")
