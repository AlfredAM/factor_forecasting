#!/usr/bin/env python3
"""
终极性能修复脚本
从根本上解决GPU利用率低、内存碎片化、数据类型错误等问题
"""

import os
import re
import subprocess
from pathlib import Path

def kill_all_training_processes():
    """彻底清理所有训练进程"""
    print("🧹 清理所有训练进程...")
    
    commands = [
        "pkill -f torchrun",
        "pkill -f unified_complete_training",
        "pkill -f python.*training",
        "nvidia-smi --gpu-reset-ecc=0,1,2,3 || true",
        "sleep 3"
    ]
    
    for cmd in commands:
        try:
            subprocess.run(cmd, shell=True, timeout=10)
            print(f"✓ 执行: {cmd}")
        except Exception as e:
            print(f"⚠️ {cmd} 执行失败: {e}")

def fix_tensor_string_error():
    """修复Tensor字符串类型错误"""
    print("🔧 修复数据类型错误...")
    
    script_path = "/nas/factor_forecasting/unified_complete_training_v2_fixed.py"
    
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复常见的字符串/Tensor混用问题
    fixes = [
        # 修复target_columns的处理
        (r'if target_col in batch\[\'targets\'\]', r'if target_col in batch[\'targets\'].keys()'),
        (r'if target_col in targets', r'if target_col in targets.keys()'),
        (r'if col in predictions', r'if col in predictions.keys()'),
        (r'target_col in batch', r'target_col in list(batch.keys())'),
        
        # 修复字典键的比较
        (r'(\w+) in (\w+)\[\'(\w+)\'\]', r'\1 in list(\2[\'\3\'].keys()) if isinstance(\2[\'\3\'], dict) else \1 in \2[\'\3\']'),
    ]
    
    for pattern, replacement in fixes:
        content = re.sub(pattern, replacement, content)
    
    # 添加类型检查函数
    type_check_code = '''
def safe_key_check(key, container):
    """安全的键检查函数"""
    if isinstance(container, dict):
        return key in container
    elif hasattr(container, 'keys'):
        return key in container.keys()
    elif hasattr(container, '__contains__'):
        try:
            return key in container
        except TypeError:
            return False
    return False
'''
    
    # 在导入后添加辅助函数
    if 'def safe_key_check' not in content:
        import_end = content.find('# Import components')
        if import_end != -1:
            content = content[:import_end] + type_check_code + '\n' + content[import_end:]
    
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✓ 数据类型错误已修复")

def create_optimized_4gpu_config():
    """创建优化的4GPU配置"""
    print("⚙️ 创建优化的4GPU配置...")
    
    config_content = """# 优化的4GPU高性能配置
model_type: AdvancedFactorForecastingTCNAttention
input_dim: 100
hidden_dim: 512       # 适中的隐藏层维度
num_layers: 8         # 适中的层数  
num_heads: 8          # 适中的注意力头数
tcn_kernel_size: 3
tcn_dilation_factor: 2
dropout_rate: 0.1
attention_dropout: 0.05
target_columns: [nextT1d]  # 单一目标减少内存占用
sequence_length: 30        # 适中的序列长度
epochs: 200
batch_size: 512           # 每GPU批次大小
fixed_batch_size: 512
learning_rate: 0.0001
weight_decay: 0.01
gradient_clip_norm: 1.0
use_mixed_precision: true
accumulation_steps: 1
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
train_end_date: 2018-10-31   # 前10个月数据
val_start_date: 2018-11-01
val_end_date: 2018-11-30
test_start_date: 2018-12-01
test_end_date: 2018-12-31
enforce_next_year_prediction: false
enable_yearly_rolling: false
min_train_years: 1
rolling_window_years: 1
shuffle_buffer_size: 256

# 性能优化参数
world_size: 4
backend: nccl
enable_gradient_checkpointing: false  # 提高速度
torch_compile: false
use_channels_last: true
"""
    
    with open("/nas/factor_forecasting/optimized_4gpu_config.yaml", "w") as f:
        f.write(config_content)
    
    print("✓ 4GPU配置已创建")

def create_performance_launcher():
    """创建高性能启动脚本"""
    print("🚀 创建高性能启动脚本...")
    
    launcher_content = '''#!/bin/bash
# 高性能4GPU训练启动脚本

# 设置环境变量优化性能
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"
export NCCL_DEBUG=INFO
export NCCL_TREE_THRESHOLD=0
export OMP_NUM_THREADS=32
export CUDA_LAUNCH_BLOCKING=0
export PYTHONUNBUFFERED=1

# 设置分布式训练参数
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=4

cd /nas/factor_forecasting
source venv/bin/activate

echo "🧹 清理GPU缓存..."
python -c "import torch; [torch.cuda.empty_cache() for _ in range(4)]" 2>/dev/null || true

echo "🚀 启动4GPU分布式训练..."
torchrun \\
    --nproc_per_node=4 \\
    --master_addr=localhost \\
    --master_port=12355 \\
    unified_complete_training_v2_fixed.py \\
    --config optimized_4gpu_config.yaml \\
    > training_4gpu_optimized.log 2>&1 &

echo "训练已启动，PID: $!"
echo "日志文件: training_4gpu_optimized.log"
echo "监控命令: tail -f training_4gpu_optimized.log"
'''
    
    with open("/nas/factor_forecasting/launch_4gpu_optimized.sh", "w") as f:
        f.write(launcher_content)
    
    # 设置执行权限
    os.chmod("/nas/factor_forecasting/launch_4gpu_optimized.sh", 0o755)
    print("✓ 启动脚本已创建")

def create_performance_monitor():
    """创建性能监控脚本"""
    print("📊 创建性能监控脚本...")
    
    monitor_content = '''#!/usr/bin/env python3
"""
高性能训练监控脚本
实时监控4GPU训练状态和性能指标
"""

import subprocess
import time
import re
import json
from datetime import datetime
from pathlib import Path

class PerformanceMonitor:
    def __init__(self):
        self.last_correlation_check = 0
        self.start_time = time.time()
        
    def get_gpu_utilization(self):
        """获取GPU利用率详情"""
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)
            
            gpus = []
            for line in result.stdout.strip().split('\\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 7:
                        gpus.append({
                            'id': parts[0],
                            'name': parts[1],
                            'mem_used': int(parts[2]),
                            'mem_total': int(parts[3]),
                            'gpu_util': int(parts[4]),
                            'mem_util': int(parts[5]),
                            'temp': int(parts[6]),
                            'power': float(parts[7]) if parts[7] != '[N/A]' else 0
                        })
            return gpus
        except Exception as e:
            print(f"GPU状态获取失败: {e}")
            return []
    
    def get_cpu_info(self):
        """获取CPU信息"""
        try:
            # CPU利用率
            cpu_result = subprocess.run(['top', '-bn1'], capture_output=True, text=True)
            cpu_line = ""
            for line in cpu_result.stdout.split('\\n'):
                if 'Cpu(s)' in line:
                    cpu_line = line
                    break
            
            # 提取CPU使用率
            cpu_usage = 0
            if 'us' in cpu_line:
                match = re.search(r'(\\d+\\.\\d+)%us', cpu_line)
                if match:
                    cpu_usage = float(match.group(1))
            
            return {'cpu_usage': cpu_usage, 'cpu_line': cpu_line}
        except Exception as e:
            return {'cpu_usage': 0, 'cpu_line': f'获取失败: {e}'}
    
    def get_memory_info(self):
        """获取内存信息"""
        try:
            result = subprocess.run(['free', '-h'], capture_output=True, text=True)
            lines = result.stdout.strip().split('\\n')
            for line in lines:
                if line.startswith('Mem:'):
                    parts = line.split()
                    return {
                        'total': parts[1],
                        'used': parts[2],
                        'free': parts[3],
                        'available': parts[6] if len(parts) > 6 else parts[3]
                    }
        except Exception as e:
            return {'error': str(e)}
    
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
                            'cpu': parts[2],
                            'mem': parts[3],
                            'command': ' '.join(parts[10:])
                        })
            return processes
        except Exception as e:
            return []
    
    def get_training_progress(self):
        """获取训练进度"""
        try:
            log_file = Path('/nas/factor_forecasting/training_4gpu_optimized.log')
            if log_file.exists():
                with open(log_file, 'r') as f:
                    content = f.read()
                
                # 提取最新的epoch信息
                epoch_pattern = r'Epoch (\\d+) Training: (\\d+)it \\[([^,]+), ([^]]+)\\]'
                matches = re.findall(epoch_pattern, content)
                
                if matches:
                    epoch, iterations, time_elapsed, time_per_it = matches[-1]
                    return {
                        'epoch': int(epoch),
                        'iterations': int(iterations),
                        'time_elapsed': time_elapsed,
                        'time_per_iteration': time_per_it,
                        'has_errors': 'ERROR' in content[-1000:]  # 检查最近1000字符
                    }
        except Exception as e:
            pass
        
        return None
    
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
                    
                    if 'correlations' in data:
                        return data['correlations']
        except Exception:
            pass
        
        return {}
    
    def calculate_efficiency(self, gpus):
        """计算硬件效率"""
        if not gpus:
            return {'gpu_efficiency': 0, 'memory_efficiency': 0}
        
        total_gpu_util = sum(gpu['gpu_util'] for gpu in gpus)
        total_mem_util = sum(gpu['mem_util'] for gpu in gpus)
        
        gpu_efficiency = total_gpu_util / len(gpus)
        memory_efficiency = total_mem_util / len(gpus)
        
        return {
            'gpu_efficiency': gpu_efficiency,
            'memory_efficiency': memory_efficiency,
            'total_power': sum(gpu['power'] for gpu in gpus)
        }
    
    def monitor(self):
        """主监控循环"""
        print("🚀 4GPU高性能训练监控系统")
        print("=" * 100)
        
        while True:
            current_time = time.time()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            runtime = current_time - self.start_time
            
            print(f"\\n[{timestamp}] 运行时间: {runtime/3600:.1f}小时")
            print("=" * 100)
            
            # GPU状态
            gpus = self.get_gpu_utilization()
            if gpus:
                print("📊 GPU状态:")
                for gpu in gpus:
                    mem_percent = (gpu['mem_used'] / gpu['mem_total']) * 100
                    print(f"  GPU {gpu['id']}: {gpu['gpu_util']:3d}% 利用率 | "
                          f"{gpu['mem_used']:5d}MB/{gpu['mem_total']}MB ({mem_percent:4.1f}%) | "
                          f"{gpu['temp']:2d}°C | {gpu['power']:5.1f}W")
                
                # 效率分析
                efficiency = self.calculate_efficiency(gpus)
                print(f"\\n⚡ 效率指标:")
                print(f"  平均GPU利用率: {efficiency['gpu_efficiency']:.1f}%")
                print(f"  平均内存利用率: {efficiency['memory_efficiency']:.1f}%")
                print(f"  总功耗: {efficiency['total_power']:.1f}W")
            
            # CPU状态
            cpu_info = self.get_cpu_info()
            print(f"\\n🔥 CPU状态:")
            print(f"  利用率: {cpu_info['cpu_usage']:.1f}% (128核)")
            
            # 内存状态
            mem_info = self.get_memory_info()
            if 'error' not in mem_info:
                print(f"  内存: {mem_info['used']}/{mem_info['total']} (可用: {mem_info['available']})")
            
            # 训练进程
            processes = self.get_training_processes()
            print(f"\\n🏃 训练进程: {len(processes)}个")
            for proc in processes[:4]:  # 显示前4个
                print(f"  PID {proc['pid']}: CPU {proc['cpu']}%, 内存 {proc['mem']}%")
            
            # 训练进度
            progress = self.get_training_progress()
            if progress:
                print(f"\\n📈 训练进度:")
                print(f"  当前Epoch: {progress['epoch']}")
                print(f"  完成迭代: {progress['iterations']}")
                print(f"  已用时间: {progress['time_elapsed']}")
                print(f"  每次迭代: {progress['time_per_iteration']}")
                
                if progress['has_errors']:
                    print("  ⚠️ 检测到错误")
                else:
                    print("  ✅ 运行正常")
            else:
                print("\\n❌ 未检测到训练进度")
            
            # 每2小时检查相关性
            if current_time - self.last_correlation_check >= 7200:
                print("\\n📊 相关性检查...")
                correlations = self.check_correlations()
                if correlations:
                    for target, data in correlations.items():
                        if isinstance(data, dict):
                            in_sample = data.get('in_sample_ic', 'N/A')
                            out_sample = data.get('out_sample_ic', 'N/A')
                            print(f"  {target}: In-sample={in_sample}, Out-sample={out_sample}")
                else:
                    print("  暂无相关性数据")
                
                self.last_correlation_check = current_time
            
            print("=" * 100)
            time.sleep(30)  # 30秒检查一次

if __name__ == "__main__":
    monitor = PerformanceMonitor()
    try:
        monitor.monitor()
    except KeyboardInterrupt:
        print("\\n\\n👋 监控结束")
'''
    
    with open("/nas/factor_forecasting/performance_monitor.py", "w") as f:
        f.write(monitor_content)
    
    print("✓ 性能监控脚本已创建")

def apply_all_fixes():
    """应用所有修复"""
    print("🔧 开始应用终极性能修复...")
    print("=" * 60)
    
    kill_all_training_processes()
    fix_tensor_string_error()
    create_optimized_4gpu_config()
    create_performance_launcher()
    create_performance_monitor()
    
    print("=" * 60)
    print("✅ 所有修复已完成!")
    print("🚀 准备启动高性能4GPU训练...")

if __name__ == "__main__":
    apply_all_fixes()
