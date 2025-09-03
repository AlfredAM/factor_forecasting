#!/usr/bin/env python3
"""
智能资源优化器 - 最大化硬件利用率同时避免OOM
动态调整训练参数以充分利用4张A10 GPU + 739GB RAM + 128核CPU
"""

import subprocess
import time
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

class IntelligentResourceOptimizer:
    def __init__(self):
        self.server_ip = "8.216.35.79"
        self.password = "Abab1234"
        self.project_path = "/nas/factor_forecasting"
        
        # 硬件规格
        self.total_gpus = 4
        self.gpu_memory_mb = 23028
        self.total_ram_gb = 739
        self.cpu_cores = 128
        
        # 安全边界
        self.gpu_memory_safety_margin = 0.05  # 5%安全边界
        self.ram_safety_margin = 0.1  # 10%安全边界
        
        # 监控历史
        self.performance_history = []
        self.correlation_reports = []
        
    def ssh_execute(self, command):
        """执行SSH命令"""
        try:
            result = subprocess.run([
                'sshpass', '-p', self.password, 'ssh', '-o', 'StrictHostKeyChecking=no',
                f'ecs-user@{self.server_ip}', command
            ], capture_output=True, text=True, timeout=30)
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Command timeout"
        except Exception as e:
            return False, "", str(e)
    
    def get_system_status(self):
        """获取系统状态"""
        status = {
            'timestamp': datetime.now(),
            'gpu_status': {},
            'memory_status': {},
            'process_status': {},
            'training_metrics': {}
        }
        
        # GPU状态
        success, gpu_output, _ = self.ssh_execute(
            "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits"
        )
        
        if success and gpu_output.strip():
            for line in gpu_output.strip().split('\n'):
                parts = line.split(', ')
                if len(parts) >= 6:
                    gpu_id, name, mem_used, mem_total, util, temp = parts
                    status['gpu_status'][int(gpu_id)] = {
                        'memory_used_mb': int(mem_used),
                        'memory_total_mb': int(mem_total),
                        'utilization_pct': int(util),
                        'temperature_c': int(temp),
                        'memory_usage_pct': int(mem_used) / int(mem_total) * 100
                    }
        
        # 内存状态
        success, mem_output, _ = self.ssh_execute("free -g | grep Mem:")
        if success and mem_output.strip():
            parts = mem_output.strip().split()
            if len(parts) >= 7:
                total, used, free = int(parts[1]), int(parts[2]), int(parts[3])
                status['memory_status'] = {
                    'total_gb': total,
                    'used_gb': used,
                    'free_gb': free,
                    'usage_pct': used / total * 100
                }
        
        # 训练进程状态
        success, proc_output, _ = self.ssh_execute(
            'ps aux | grep "python.*unified_complete_training" | grep -v grep'
        )
        
        if success and proc_output.strip():
            status['process_status']['training_active'] = True
            # 提取CPU和内存使用
            for line in proc_output.strip().split('\n'):
                parts = line.split()
                if len(parts) >= 11:
                    status['process_status']['cpu_pct'] = float(parts[2])
                    status['process_status']['memory_pct'] = float(parts[3])
                    status['process_status']['pid'] = int(parts[1])
        else:
            status['process_status']['training_active'] = False
        
        # 训练指标
        success, log_output, _ = self.ssh_execute(
            f"cd {self.project_path} && tail -10 training_memory_optimized.log 2>/dev/null || tail -10 *.log | tail -10"
        )
        
        if success and log_output:
            status['training_metrics'] = self.parse_training_metrics(log_output)
        
        return status
    
    def parse_training_metrics(self, log_output):
        """解析训练指标"""
        metrics = {}
        
        # 查找epoch信息
        epoch_pattern = r'Epoch (\d+) Training: (\d+)it \[([^,]+), ([^]]+)\]'
        matches = re.findall(epoch_pattern, log_output)
        
        if matches:
            epoch, iterations, time_elapsed, time_per_it = matches[-1]
            metrics['current_epoch'] = int(epoch)
            metrics['iterations'] = int(iterations)
            metrics['time_elapsed'] = time_elapsed
            metrics['time_per_iteration'] = time_per_it
            
            # 估算epoch完成时间
            if 'it/s' in time_per_it:
                its_per_sec = float(time_per_it.split('it/s')[0])
                metrics['iterations_per_second'] = its_per_sec
            elif 's/it' in time_per_it:
                sec_per_it = float(time_per_it.split('s/it')[0])
                metrics['seconds_per_iteration'] = sec_per_it
        
        # 检查错误
        if 'CUDA out of memory' in log_output:
            metrics['memory_error'] = True
        
        return metrics
    
    def calculate_optimal_config(self, current_status):
        """根据当前状态计算最优配置"""
        config = {}
        
        # 分析GPU使用情况
        gpu_0_usage = current_status['gpu_status'].get(0, {})
        gpu_memory_used_pct = gpu_0_usage.get('memory_usage_pct', 0)
        
        # 动态调整批次大小
        if gpu_memory_used_pct > 95:  # 接近满负荷
            config['batch_size'] = 512  # 减小批次
            config['accumulation_steps'] = 8
        elif gpu_memory_used_pct < 80:  # 有余量
            config['batch_size'] = 1024  # 增大批次
            config['accumulation_steps'] = 4
        else:
            config['batch_size'] = 768  # 中等批次
            config['accumulation_steps'] = 6
        
        # 利用多GPU（如果内存允许）
        if gpu_memory_used_pct < 85:
            config['use_distributed'] = True
            config['world_size'] = 2  # 先用2个GPU测试
        else:
            config['use_distributed'] = False
        
        # CPU优化
        cpu_cores_to_use = min(32, self.cpu_cores // 4)  # 保守使用CPU
        config['num_workers'] = 0  # 保持为0避免CUDA问题
        config['dataloader_threads'] = cpu_cores_to_use
        
        # 内存优化
        ram_usage_pct = current_status['memory_status'].get('usage_pct', 0)
        if ram_usage_pct < 50:  # 内存充足
            config['cache_size'] = 50  # 增大缓存
            config['streaming_chunk_size'] = 50000
        else:
            config['cache_size'] = 20
            config['streaming_chunk_size'] = 10000
        
        return config
    
    def create_optimized_config(self, config_params):
        """创建优化配置文件"""
        config_content = f"""# 智能优化配置 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
model_type: AdvancedFactorForecastingTCNAttention
input_dim: 100
hidden_dim: 768    # 平衡性能和内存
num_layers: 12     # 充分利用GPU计算能力
num_heads: 16      # 充分的注意力头
tcn_kernel_size: 7
tcn_dilation_factor: 2
dropout_rate: 0.1
attention_dropout: 0.05
target_columns: [intra30m, nextT1d, ema1d]  # 所有目标
sequence_length: 60
epochs: 200
batch_size: {config_params['batch_size']}
fixed_batch_size: {config_params['batch_size']}
learning_rate: 0.0001
weight_decay: 0.01
gradient_clip_norm: 1.0
use_mixed_precision: true
accumulation_steps: {config_params['accumulation_steps']}
use_adaptive_batch_size: false
num_workers: 0
pin_memory: false
use_distributed: {str(config_params['use_distributed']).lower()}
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
shuffle_buffer_size: {config_params.get('cache_size', 20) * 10}

# 性能优化
streaming_chunk_size: {config_params.get('streaming_chunk_size', 10000)}
cache_size: {config_params.get('cache_size', 20)}
max_memory_usage: {int(self.total_ram_gb * 0.8)}
enable_memory_mapping: true
torch_compile: false  # 避免编译开销
"""
        
        return config_content
    
    def restart_training_if_needed(self, current_status):
        """如果需要则重启训练"""
        if not current_status['process_status'].get('training_active', False):
            print("🔄 训练进程未运行，准备重新启动...")
            
            # 计算最优配置
            optimal_config = self.calculate_optimal_config(current_status)
            config_content = self.create_optimized_config(optimal_config)
            
            # 创建配置文件
            config_file = "intelligent_optimized_config.yaml"
            success, _, _ = self.ssh_execute(f"""cd {self.project_path} && cat > {config_file} << 'EOF'
{config_content}
EOF""")
            
            if success:
                print(f"✅ 创建优化配置: {optimal_config}")
                
                # 启动训练
                start_cmd = f"""cd {self.project_path} && 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256 &&
source venv/bin/activate &&
nohup python unified_complete_training_v2_fixed.py --config {config_file} > training_intelligent.log 2>&1 &"""
                
                success, output, error = self.ssh_execute(start_cmd)
                if success:
                    print("✅ 训练重新启动成功")
                    return True
                else:
                    print(f"❌ 启动失败: {error}")
            
            return False
        return True
    
    def check_correlation_reports(self):
        """检查相关性报告"""
        success, output, _ = self.ssh_execute(
            f"cd {self.project_path} && find outputs/ -name '*.json' -mtime -1 2>/dev/null | head -5"
        )
        
        if success and output.strip():
            print("📊 发现相关性报告:")
            for file_path in output.strip().split('\n'):
                if file_path.strip():
                    # 读取报告内容
                    success, content, _ = self.ssh_execute(f"cat {file_path}")
                    if success:
                        try:
                            report = json.loads(content)
                            timestamp = report.get('timestamp', 'Unknown')
                            print(f"  📈 报告时间: {timestamp}")
                            
                            # 显示相关性指标
                            for target in ['intra30m', 'nextT1d', 'ema1d']:
                                in_sample = report.get(f'{target}_in_sample_ic', 'N/A')
                                out_sample = report.get(f'{target}_out_sample_ic', 'N/A')
                                print(f"    {target}: In-sample={in_sample:.4f}, Out-sample={out_sample:.4f}")
                                
                        except json.JSONDecodeError:
                            print(f"    ⚠️  无法解析报告: {file_path}")
    
    def monitor_continuously(self):
        """持续监控和优化"""
        print("🚀 启动智能资源优化监控系统")
        print("=" * 80)
        
        last_correlation_check = 0
        last_optimization = 0
        
        while True:
            current_time = time.time()
            
            try:
                # 获取系统状态
                status = self.get_system_status()
                timestamp = status['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
                
                print(f"\n[{timestamp}] 系统状态监控")
                print("-" * 60)
                
                # GPU状态
                for gpu_id, gpu_info in status['gpu_status'].items():
                    mem_used = gpu_info['memory_used_mb']
                    mem_total = gpu_info['memory_total_mb']
                    util = gpu_info['utilization_pct']
                    temp = gpu_info['temperature_c']
                    mem_pct = gpu_info['memory_usage_pct']
                    
                    print(f"🔥 GPU {gpu_id}: {mem_used}MB/{mem_total}MB ({mem_pct:.1f}%) | {util}% util | {temp}°C")
                
                # 内存状态
                if status['memory_status']:
                    mem_info = status['memory_status']
                    print(f"💾 RAM: {mem_info['used_gb']}GB/{mem_info['total_gb']}GB ({mem_info['usage_pct']:.1f}%)")
                
                # 训练状态
                if status['process_status'].get('training_active'):
                    proc_info = status['process_status']
                    print(f"⚡ 训练进程: PID {proc_info['pid']} | CPU {proc_info['cpu_pct']:.1f}% | 内存 {proc_info['memory_pct']:.1f}%")
                    
                    # 训练指标
                    if status['training_metrics']:
                        metrics = status['training_metrics']
                        if 'current_epoch' in metrics:
                            epoch = metrics['current_epoch']
                            iterations = metrics['iterations']
                            time_elapsed = metrics['time_elapsed']
                            time_per_it = metrics.get('time_per_iteration', 'N/A')
                            
                            print(f"📊 训练进度: Epoch {epoch} | {iterations} iterations | {time_elapsed} elapsed | {time_per_it}")
                            
                            # 估算epoch完成时间
                            if 'seconds_per_iteration' in metrics:
                                sec_per_it = metrics['seconds_per_iteration']
                                # 假设每个epoch大约1000个iteration
                                remaining_its = max(0, 1000 - iterations)
                                remaining_time = remaining_its * sec_per_it
                                eta_minutes = remaining_time / 60
                                print(f"⏱️  预计Epoch完成时间: {eta_minutes:.1f} 分钟")
                        
                        if metrics.get('memory_error'):
                            print("⚠️  检测到CUDA内存错误")
                else:
                    print("❌ 训练进程未运行")
                
                # 重启训练（如果需要）
                if current_time - last_optimization > 600:  # 每10分钟检查一次
                    self.restart_training_if_needed(status)
                    last_optimization = current_time
                
                # 检查相关性报告（每2小时）
                if current_time - last_correlation_check >= 7200:
                    print("\n📈 检查相关性报告...")
                    self.check_correlation_reports()
                    last_correlation_check = current_time
                
                # 记录性能历史
                self.performance_history.append(status)
                if len(self.performance_history) > 100:  # 保留最近100条记录
                    self.performance_history.pop(0)
                
                print("-" * 60)
                
            except Exception as e:
                print(f"❌ 监控错误: {e}")
            
            # 等待30秒
            time.sleep(30)

def main():
    """主函数"""
    optimizer = IntelligentResourceOptimizer()
    
    try:
        optimizer.monitor_continuously()
    except KeyboardInterrupt:
        print("\n\n👋 监控结束")
    except Exception as e:
        print(f"\n❌ 系统错误: {e}")

if __name__ == "__main__":
    main()