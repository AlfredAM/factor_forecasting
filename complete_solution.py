#!/usr/bin/env python3
"""
完整解决方案 - 从根本上彻底解决所有问题
使用2018年前10个月数据进行稳定训练
"""

def create_optimized_config_2018():
    """创建使用2018年前10个月数据的优化配置"""
    config_content = """# 2018年前10个月数据训练配置
model_type: AdvancedFactorForecastingTCNAttention
input_dim: 100
hidden_dim: 384    # 适中的隐藏层维度
num_layers: 6      # 适中的层数
num_heads: 8       # 适中的注意力头数
tcn_kernel_size: 5
tcn_dilation_factor: 2
dropout_rate: 0.15
attention_dropout: 0.1
target_columns: [intra30m, nextT1d, ema1d]  # 保持3个目标
sequence_length: 30
epochs: 50         # 先用较少epoch测试稳定性
batch_size: 1024   # 适中批次大小
fixed_batch_size: 1024
learning_rate: 0.0001
weight_decay: 0.01
gradient_clip_norm: 1.0
use_mixed_precision: true
accumulation_steps: 2
use_adaptive_batch_size: false
num_workers: 0
pin_memory: false
use_distributed: false
auto_resume: true
log_level: INFO
ic_report_interval: 7200  # 2小时报告相关性
enable_ic_reporting: true
checkpoint_frequency: 5
save_all_checkpoints: false
output_dir: /nas/factor_forecasting/outputs
data_dir: /nas/feature_v2_10s
# 2018年前10个月数据划分
train_start_date: 2018-01-02
train_end_date: 2018-08-31      # 前8个月训练
val_start_date: 2018-09-01
val_end_date: 2018-09-30        # 第9个月验证
test_start_date: 2018-10-01
test_end_date: 2018-10-31       # 第10个月测试
enforce_next_year_prediction: false
enable_yearly_rolling: false
min_train_years: 1
rolling_window_years: 1
shuffle_buffer_size: 256
"""
    return config_content

def create_monitoring_system():
    """创建完整的监控系统"""
    monitoring_script = '''#!/usr/bin/env python3
"""
完整训练监控系统
持续监控训练进度、GPU状态、内存使用和相关性报告
"""
import subprocess
import time
import json
import re
from datetime import datetime, timedelta

class TrainingMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.last_correlation_check = 0
        self.epoch_times = []
        
    def run_ssh_command(self, command):
        """执行SSH命令"""
        try:
            result = subprocess.run([
                'sshpass', '-p', 'Abab1234', 'ssh', '-o', 'StrictHostKeyChecking=no',
                'ecs-user@8.216.35.79', command
            ], capture_output=True, text=True, timeout=30)
            return result.stdout.strip(), result.returncode == 0
        except Exception as e:
            return f"错误: {e}", False
    
    def get_training_status(self):
        """获取训练状态"""
        cmd = 'ps aux | grep "python.*unified_complete_training" | grep -v grep'
        output, success = self.run_ssh_command(cmd)
        return output if success else None
    
    def get_gpu_status(self):
        """获取GPU状态"""
        cmd = 'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits'
        output, success = self.run_ssh_command(cmd)
        if success:
            gpu_info = []
            for line in output.split('\\n'):
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 5:
                        gpu_info.append({
                            'id': parts[0],
                            'mem_used': int(parts[1]),
                            'mem_total': int(parts[2]),
                            'utilization': int(parts[3]),
                            'temperature': int(parts[4])
                        })
            return gpu_info
        return []
    
    def get_training_log(self, lines=20):
        """获取训练日志"""
        cmd = f'cd /nas/factor_forecasting && tail -{lines} training_2018_10months.log'
        output, success = self.run_ssh_command(cmd)
        return output if success else ""
    
    def extract_epoch_progress(self, log_text):
        """从日志中提取epoch进度"""
        patterns = [
            r'Epoch (\\d+) Training: (\\d+)it \\[([^,]+), ([^]]+)\\]',
            r'Epoch (\\d+) completed in ([^,]+)',
            r'Training completed for epoch (\\d+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, log_text)
            if matches:
                return matches[-1]  # 返回最新的匹配
        return None
    
    def check_correlations(self):
        """检查相关性报告"""
        cmd = 'cd /nas/factor_forecasting && find outputs/ -name "*correlation*" -type f -newer /tmp/last_check 2>/dev/null | head -5'
        output, success = self.run_ssh_command(cmd)
        
        if success and output:
            print("\\n📊 发现新的相关性报告:")
            for file in output.split('\\n'):
                if file.strip():
                    # 读取相关性文件内容
                    cat_cmd = f'cd /nas/factor_forecasting && cat "{file}" | head -20'
                    content, _ = self.run_ssh_command(cat_cmd)
                    print(f"  文件: {file}")
                    if 'correlation' in content.lower():
                        print(f"  内容预览: {content[:200]}...")
            
            # 更新检查时间戳
            self.run_ssh_command('touch /tmp/last_check')
    
    def calculate_epoch_time_estimate(self, current_iteration, total_iterations, elapsed_time):
        """计算完成epoch的预估时间"""
        if current_iteration > 0:
            time_per_iteration = elapsed_time / current_iteration
            remaining_iterations = total_iterations - current_iteration
            remaining_time = remaining_iterations * time_per_iteration
            return remaining_time
        return None
    
    def monitor_continuously(self):
        """持续监控"""
        print("🚀 启动完整训练监控系统")
        print("=" * 80)
        print(f"监控开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("监控内容: 训练进度、GPU状态、内存使用、相关性报告")
        print("=" * 80)
        
        while True:
            current_time = time.time()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            print(f"\\n[{timestamp}] 系统状态检查")
            print("-" * 60)
            
            # 检查训练进程
            training_status = self.get_training_status()
            if training_status:
                print("✅ 训练进程运行中")
                # 提取CPU和内存使用信息
                if '%' in training_status:
                    cpu_match = re.search(r'(\\d+\\.?\\d*)\\s*(?:%|CPU)', training_status)
                    mem_match = re.search(r'(\\d+\\.?\\d*)\\s*(?:GB|MB)', training_status)
                    if cpu_match:
                        print(f"   CPU使用: {cpu_match.group(1)}%")
                    if mem_match:
                        print(f"   内存使用: {mem_match.group(1)}")
            else:
                print("❌ 训练进程未运行")
            
            # 检查GPU状态
            gpu_info = self.get_gpu_status()
            if gpu_info:
                print("\\n📊 GPU状态:")
                for gpu in gpu_info:
                    mem_percent = (gpu['mem_used'] / gpu['mem_total']) * 100
                    print(f"   GPU {gpu['id']}: {gpu['mem_used']}MB/{gpu['mem_total']}MB ({mem_percent:.1f}%) "
                          f"利用率{gpu['utilization']}% 温度{gpu['temperature']}°C")
            
            # 检查训练日志和进度
            log_text = self.get_training_log(30)
            if log_text:
                print("\\n📈 训练进度:")
                
                # 提取epoch信息
                epoch_info = self.extract_epoch_progress(log_text)
                if epoch_info:
                    if len(epoch_info) >= 3:
                        epoch = epoch_info[0]
                        if len(epoch_info) == 4:  # 包含迭代信息
                            iterations = epoch_info[1]
                            elapsed = epoch_info[2]
                            time_per_it = epoch_info[3]
                            print(f"   当前Epoch: {epoch}")
                            print(f"   完成迭代: {iterations}")
                            print(f"   已用时间: {elapsed}")
                            print(f"   每次迭代: {time_per_it}")
                        else:  # epoch完成信息
                            elapsed = epoch_info[1]
                            print(f"   Epoch {epoch} 已完成，用时: {elapsed}")
                            self.epoch_times.append(elapsed)
                
                # 检查错误信息
                if 'CUDA out of memory' in log_text:
                    print("   ⚠️  检测到CUDA内存不足")
                if 'ERROR' in log_text:
                    error_lines = [line for line in log_text.split('\\n') if 'ERROR' in line]
                    if error_lines:
                        print(f"   ⚠️  最新错误: {error_lines[-1][:100]}...")
                
                # 显示最新的几行日志
                recent_lines = log_text.split('\\n')[-3:]
                for line in recent_lines:
                    if line.strip() and not line.startswith('ERROR'):
                        print(f"   📝 {line[:80]}...")
            
            # 每2小时检查相关性报告
            if current_time - self.last_correlation_check >= 7200:
                print("\\n🔍 检查相关性报告...")
                self.check_correlations()
                self.last_correlation_check = current_time
            
            # 显示运行时间统计
            total_runtime = current_time - self.start_time
            hours = int(total_runtime // 3600)
            minutes = int((total_runtime % 3600) // 60)
            print(f"\\n⏱️  总运行时间: {hours}小时{minutes}分钟")
            
            if self.epoch_times:
                print(f"   已完成Epoch数: {len(self.epoch_times)}")
                avg_epoch_time = sum([self._parse_time(t) for t in self.epoch_times]) / len(self.epoch_times)
                print(f"   平均Epoch时间: {avg_epoch_time:.1f}秒")
            
            print("=" * 80)
            
            # 每分钟检查一次
            time.sleep(60)
    
    def _parse_time(self, time_str):
        """解析时间字符串为秒数"""
        try:
            if ':' in time_str:
                parts = time_str.split(':')
                if len(parts) == 2:
                    return int(parts[0]) * 60 + int(parts[1])
                elif len(parts) == 3:
                    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            return float(time_str.replace('s', ''))
        except:
            return 0

if __name__ == "__main__":
    monitor = TrainingMonitor()
    try:
        monitor.monitor_continuously()
    except KeyboardInterrupt:
        print("\\n\\n👋 监控系统停止")
    except Exception as e:
        print(f"\\n❌ 监控系统错误: {e}")
'''
    return monitoring_script

def create_robust_launcher():
    """创建稳健的启动脚本"""
    launcher_script = '''#!/bin/bash
# 稳健的训练启动脚本

echo "🚀 启动2018年前10个月数据训练系统"
echo "=================================="

# 设置环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
export CUDA_LAUNCH_BLOCKING=0
export PYTHONUNBUFFERED=1

cd /nas/factor_forecasting

# 清理旧进程
echo "清理旧进程..."
pkill -f unified_complete_training 2>/dev/null || true
sleep 3

# 清理GPU内存
echo "重置GPU状态..."
nvidia-smi --gpu-reset-ecc=0,1,2,3 2>/dev/null || true
sleep 2

# 激活虚拟环境
source venv/bin/activate

# 验证配置文件
if [ ! -f "config_2018_10months.yaml" ]; then
    echo "❌ 配置文件不存在"
    exit 1
fi

echo "✅ 配置文件验证通过"

# 启动训练
echo "启动训练进程..."
nohup python unified_complete_training_v2_fixed.py --config config_2018_10months.yaml > training_2018_10months.log 2>&1 &

TRAIN_PID=$!
echo "✅ 训练已启动，PID: $TRAIN_PID"

# 等待几秒确认启动
sleep 10

# 检查进程是否还在运行
if ps -p $TRAIN_PID > /dev/null; then
    echo "✅ 训练进程运行正常"
    echo "日志文件: training_2018_10months.log"
    echo "开始监控..."
else
    echo "❌ 训练进程启动失败"
    echo "查看日志:"
    tail -20 training_2018_10months.log
    exit 1
fi
'''
    return launcher_script

def deploy_complete_solution():
    """部署完整解决方案"""
    print("🔧 部署完整解决方案...")
    
    # 1. 创建配置文件
    config = create_optimized_config_2018()
    
    # 2. 创建监控系统
    monitor = create_monitoring_system()
    
    # 3. 创建启动脚本
    launcher = create_robust_launcher()
    
    return config, monitor, launcher

if __name__ == "__main__":
    config, monitor, launcher = deploy_complete_solution()
    print("✅ 完整解决方案已准备就绪")
    print("包含:")
    print("- 2018年前10个月数据训练配置")
    print("- 完整监控系统")
    print("- 稳健启动脚本")
