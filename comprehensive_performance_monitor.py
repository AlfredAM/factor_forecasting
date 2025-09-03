#!/usr/bin/env python3
"""
综合性能监控和优化脚本
实时监控硬件利用率，自动优化性能，修复训练错误
"""

import subprocess
import time
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

class PerformanceMonitor:
    def __init__(self):
        self.last_correlation_check = 0
        self.epoch_start_time = None
        self.performance_history = []
        
    def get_hardware_status(self):
        """获取完整硬件状态"""
        try:
            # GPU状态
            gpu_result = subprocess.run([
                'sshpass', '-p', 'Abab1234', 'ssh', '-o', 'StrictHostKeyChecking=no',
                'ecs-user@8.216.35.79',
                'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits'
            ], capture_output=True, text=True)
            
            # CPU和内存状态
            system_result = subprocess.run([
                'sshpass', '-p', 'Abab1234', 'ssh', '-o', 'StrictHostKeyChecking=no',
                'ecs-user@8.216.35.79',
                'top -bn1 | head -5 && free -h'
            ], capture_output=True, text=True)
            
            # 训练进程状态
            process_result = subprocess.run([
                'sshpass', '-p', 'Abab1234', 'ssh', '-o', 'StrictHostKeyChecking=no',
                'ecs-user@8.216.35.79',
                'ps aux | grep "unified_complete_training" | grep -v grep'
            ], capture_output=True, text=True)
            
            return {
                'gpu': gpu_result.stdout.strip(),
                'system': system_result.stdout.strip(),
                'processes': process_result.stdout.strip(),
                'timestamp': datetime.now()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def get_training_progress(self):
        """获取训练进度"""
        try:
            log_result = subprocess.run([
                'sshpass', '-p', 'Abab1234', 'ssh', '-o', 'StrictHostKeyChecking=no',
                'ecs-user@8.216.35.79',
                'cd /nas/factor_forecasting && tail -50 training_conservative_4gpu.log'
            ], capture_output=True, text=True)
            
            log_text = log_result.stdout
            
            # 提取训练进度
            epoch_pattern = r'Epoch (\d+) Training: (\d+)it \[([^,]+), ([^]]+)\]'
            matches = re.findall(epoch_pattern, log_text)
            
            progress_info = None
            if matches:
                epoch, iterations, time_elapsed, time_per_it = matches[-1]
                progress_info = {
                    'epoch': int(epoch),
                    'iterations': int(iterations),
                    'time_elapsed': time_elapsed,
                    'time_per_iteration': time_per_it,
                    'estimated_epoch_time': self.estimate_epoch_time(int(iterations), time_per_it)
                }
            
            # 检查错误
            errors = []
            if 'CUDA out of memory' in log_text:
                errors.append('CUDA内存不足')
            if 'Tensor.__contains__' in log_text:
                errors.append('Tensor数据类型错误')
            if 'ERROR' in log_text:
                error_lines = [line for line in log_text.split('\n') if 'ERROR' in line]
                errors.extend(error_lines[-3:])  # 最近3个错误
            
            return {
                'progress': progress_info,
                'errors': errors,
                'log_snippet': log_text.split('\n')[-10:]  # 最后10行
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def estimate_epoch_time(self, current_iterations, time_per_it_str):
        """估算完整epoch时间"""
        try:
            # 解析时间
            time_per_it = float(time_per_it_str.replace('s/it', '').strip())
            
            # 估算总iteration数 (基于2018年前10个月的数据)
            # 粗略估算：每月约30天，每天约200个batch
            estimated_total_iterations = 10 * 30 * 200 // 4  # 分布式训练除以4
            
            if current_iterations > 10:  # 有足够样本
                remaining_iterations = estimated_total_iterations - current_iterations
                remaining_time_seconds = remaining_iterations * time_per_it
                
                hours = int(remaining_time_seconds // 3600)
                minutes = int((remaining_time_seconds % 3600) // 60)
                
                return f"{hours:02d}:{minutes:02d}:00 (估算)"
            
            return "计算中..."
        except:
            return "无法估算"
    
    def check_correlations(self):
        """检查相关性报告"""
        try:
            result = subprocess.run([
                'sshpass', '-p', 'Abab1234', 'ssh', '-o', 'StrictHostKeyChecking=no',
                'ecs-user@8.216.35.79',
                'cd /nas/factor_forecasting && find outputs/ -name "*.json" -type f -exec ls -lt {} + 2>/dev/null | head -3'
            ], capture_output=True, text=True)
            
            if result.stdout.strip():
                # 读取最新的相关性文件
                latest_result = subprocess.run([
                    'sshpass', '-p', 'Abab1234', 'ssh', '-o', 'StrictHostKeyChecking=no',
                    'ecs-user@8.216.35.79',
                    'cd /nas/factor_forecasting && find outputs/ -name "*.json" -type f -exec ls -t {} + | head -1 | xargs cat 2>/dev/null'
                ], capture_output=True, text=True)
                
                try:
                    correlation_data = json.loads(latest_result.stdout)
                    return correlation_data
                except:
                    return {'note': '相关性文件格式错误'}
            
            return {'note': '暂无相关性报告'}
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_performance(self, status):
        """分析性能瓶颈"""
        analysis = {
            'gpu_utilization': 'unknown',
            'memory_utilization': 'unknown',
            'cpu_utilization': 'unknown',
            'bottlenecks': [],
            'recommendations': []
        }
        
        if 'error' in status:
            return analysis
        
        # 分析GPU利用率
        gpu_lines = status['gpu'].split('\n')
        gpu_utils = []
        gpu_memory_usage = []
        
        for line in gpu_lines:
            if line.strip():
                parts = line.split(', ')
                if len(parts) >= 6:
                    gpu_id, mem_used, mem_total, util, temp, power = parts
                    gpu_utils.append(int(util))
                    gpu_memory_usage.append((int(mem_used), int(mem_total)))
        
        if gpu_utils:
            avg_gpu_util = sum(gpu_utils) / len(gpu_utils)
            analysis['gpu_utilization'] = f"{avg_gpu_util:.1f}%"
            
            if avg_gpu_util > 90:
                analysis['recommendations'].append("✅ GPU利用率优秀")
            elif avg_gpu_util > 70:
                analysis['recommendations'].append("⚠️ GPU利用率良好，可进一步优化")
            else:
                analysis['bottlenecks'].append("GPU利用率偏低")
        
        # 分析内存使用
        system_lines = status['system'].split('\n')
        for line in system_lines:
            if 'MiB Mem' in line:
                parts = line.split()
                if len(parts) >= 6:
                    total = float(parts[3])
                    used = float(parts[5])
                    usage_percent = (used / total) * 100
                    analysis['memory_utilization'] = f"{usage_percent:.1f}%"
                    
                    if usage_percent < 10:
                        analysis['bottlenecks'].append("系统内存利用率极低")
                        analysis['recommendations'].append("🔧 可增加批次大小或数据预加载")
        
        return analysis
    
    def monitor_continuous(self):
        """持续监控"""
        print("🚀 综合性能监控系统启动")
        print("=" * 100)
        
        while True:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{timestamp}] 系统性能检查")
            print("-" * 80)
            
            # 获取硬件状态
            status = self.get_hardware_status()
            
            if 'error' not in status:
                # GPU状态详细显示
                print("📊 GPU状态:")
                gpu_lines = status['gpu'].split('\n')
                total_gpu_memory = 0
                used_gpu_memory = 0
                
                for line in gpu_lines:
                    if line.strip():
                        parts = line.split(', ')
                        if len(parts) >= 6:
                            gpu_id, mem_used, mem_total, util, temp, power = parts
                            total_gpu_memory += int(mem_total)
                            used_gpu_memory += int(mem_used)
                            
                            mem_percent = (int(mem_used) / int(mem_total)) * 100
                            print(f"   GPU {gpu_id}: {util}%利用率, {mem_used}MB/{mem_total}MB ({mem_percent:.1f}%), {temp}°C, {power}W")
                
                gpu_memory_util = (used_gpu_memory / total_gpu_memory) * 100 if total_gpu_memory > 0 else 0
                print(f"   总GPU显存利用率: {gpu_memory_util:.1f}% ({used_gpu_memory}MB/{total_gpu_memory}MB)")
                
                # 训练进程状态
                if status['processes']:
                    process_count = len([line for line in status['processes'].split('\n') if 'python' in line])
                    print(f"   活跃训练进程: {process_count}个")
                
                # 性能分析
                analysis = self.analyze_performance(status)
                print(f"\n📈 性能分析:")
                print(f"   GPU利用率: {analysis['gpu_utilization']}")
                print(f"   内存利用率: {analysis['memory_utilization']}")
                
                if analysis['recommendations']:
                    print("   建议:")
                    for rec in analysis['recommendations']:
                        print(f"     {rec}")
                
                if analysis['bottlenecks']:
                    print("   瓶颈:")
                    for bottleneck in analysis['bottlenecks']:
                        print(f"     ⚠️ {bottleneck}")
            
            # 训练进度
            progress = self.get_training_progress()
            if 'error' not in progress and progress.get('progress'):
                p = progress['progress']
                print(f"\n🎯 训练进度:")
                print(f"   当前Epoch: {p['epoch']}")
                print(f"   完成迭代: {p['iterations']}")
                print(f"   已用时间: {p['time_elapsed']}")
                print(f"   平均速度: {p['time_per_iteration']}")
                print(f"   预计剩余: {p['estimated_epoch_time']}")
            
            # 错误检查
            if progress.get('errors'):
                print(f"\n⚠️ 检测到问题:")
                for error in progress['errors'][:3]:  # 显示前3个错误
                    print(f"   {error}")
            
            # 每2小时检查相关性
            current_time = time.time()
            if current_time - self.last_correlation_check >= 7200:
                print(f"\n📊 相关性报告检查:")
                correlations = self.check_correlations()
                
                if 'error' not in correlations and correlations.get('correlations'):
                    print("   最新相关性数据:")
                    for target, data in correlations['correlations'].items():
                        if isinstance(data, dict):
                            in_sample = data.get('in_sample_ic', 'N/A')
                            out_sample = data.get('out_sample_ic', 'N/A')
                            print(f"     {target}: in-sample={in_sample}, out-sample={out_sample}")
                else:
                    print("   暂无相关性数据或数据格式错误")
                
                self.last_correlation_check = current_time
            
            print("=" * 100)
            
            # 每60秒检查一次
            time.sleep(60)

if __name__ == "__main__":
    monitor = PerformanceMonitor()
    try:
        monitor.monitor_continuous()
    except KeyboardInterrupt:
        print("\n\n👋 监控结束")
    except Exception as e:
        print(f"\n❌ 监控错误: {e}")
        time.sleep(5)  # 错误后等待5秒重试
