#!/usr/bin/env python3
"""
持续训练监控系统
- 实时监控4GPU使用情况
- 跟踪每个epoch完成时间
- 每2小时报告in-sample和out-of-sample相关性
- 监控服务器资源使用
"""

import subprocess
import time
import json
import re
from datetime import datetime, timedelta
import os

class ContinuousTrainingMonitor:
    def __init__(self):
        self.last_correlation_check = 0
        self.correlation_interval = 7200  # 2小时
        self.monitor_interval = 60  # 1分钟检查一次
        self.epoch_times = []
        
    def get_server_status(self):
        """获取服务器状态"""
        try:
            result = subprocess.run([
                'sshpass', '-p', 'Abab1234', 'ssh', '-o', 'StrictHostKeyChecking=no',
                'ecs-user@8.216.35.79',
                'nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu --format=csv,noheader,nounits'
            ], capture_output=True, text=True)
            
            return result.stdout.strip()
        except Exception as e:
            return f"服务器状态获取失败: {e}"

    def get_system_resources(self):
        """获取系统资源使用情况"""
        try:
            result = subprocess.run([
                'sshpass', '-p', 'Abab1234', 'ssh', '-o', 'StrictHostKeyChecking=no',
                'ecs-user@8.216.35.79',
                'echo "=== CPU使用率 ===" && top -bn1 | grep "Cpu(s)" && echo "=== 内存使用 ===" && free -h && echo "=== 磁盘使用 ===" && df -h /nas'
            ], capture_output=True, text=True)
            
            return result.stdout.strip()
        except Exception as e:
            return f"系统资源获取失败: {e}"

    def get_training_processes(self):
        """获取训练进程信息"""
        try:
            result = subprocess.run([
                'sshpass', '-p', 'Abab1234', 'ssh', '-o', 'StrictHostKeyChecking=no',
                'ecs-user@8.216.35.79',
                'ps aux | grep -E "(torchrun|unified_complete)" | grep -v grep'
            ], capture_output=True, text=True)
            
            return result.stdout.strip()
        except Exception as e:
            return f"训练进程获取失败: {e}"

    def get_latest_training_log(self):
        """获取最新训练日志"""
        try:
            result = subprocess.run([
                'sshpass', '-p', 'Abab1234', 'ssh', '-o', 'StrictHostKeyChecking=no',
                'ecs-user@8.216.35.79',
                'cd /nas/factor_forecasting && ls -t training_*.log | head -1 | xargs tail -100'
            ], capture_output=True, text=True)
            
            return result.stdout.strip()
        except Exception as e:
            return f"训练日志获取失败: {e}"

    def extract_epoch_info(self, log_text):
        """提取epoch完成信息和时间"""
        # 查找epoch进度信息
        epoch_patterns = [
            r'Epoch (\d+) Training: (\d+)it \[([^,]+), ([^]]+)\]',
            r'Epoch (\d+) completed.*?time: ([\\d\\.]+)s',
            r'Epoch (\d+).*?(\d+)it.*?\[([^]]+)\]'
        ]
        
        latest_info = None
        for pattern in epoch_patterns:
            matches = re.findall(pattern, log_text)
            if matches:
                latest_info = matches[-1]
                break
        
        return latest_info

    def calculate_epoch_time_estimate(self, current_info):
        """计算epoch完成时间估计"""
        if not current_info or len(current_info) < 3:
            return None
            
        try:
            # 解析当前进度
            epoch_num = int(current_info[0])
            iterations = int(current_info[1])
            time_info = current_info[2]
            
            # 解析时间信息 (格式: MM:SS 或 s/it)
            if ':' in time_info:
                # 格式: MM:SS
                parts = time_info.split(':')
                total_seconds = int(parts[0]) * 60 + int(parts[1])
                avg_time_per_it = total_seconds / iterations if iterations > 0 else 0
            elif 's/it' in time_info:
                # 格式: X.XXs/it
                avg_time_per_it = float(time_info.replace('s/it', ''))
            else:
                return None
            
            # 估算总iteration数 (基于之前的观察，大约需要几百个iteration)
            estimated_total_iterations = 500  # 可以根据实际情况调整
            remaining_iterations = max(0, estimated_total_iterations - iterations)
            estimated_remaining_time = remaining_iterations * avg_time_per_it
            
            return {
                'epoch': epoch_num,
                'current_iterations': iterations,
                'avg_time_per_iteration': avg_time_per_it,
                'estimated_remaining_seconds': estimated_remaining_time,
                'estimated_completion': datetime.now() + timedelta(seconds=estimated_remaining_time)
            }
            
        except Exception as e:
            print(f"时间估算错误: {e}")
            return None

    def check_correlation_reports(self):
        """检查相关性报告"""
        try:
            result = subprocess.run([
                'sshpass', '-p', 'Abab1234', 'ssh', '-o', 'StrictHostKeyChecking=no',
                'ecs-user@8.216.35.79',
                'cd /nas/factor_forecasting && find outputs/ -name "*correlation*" -type f -mtime -1 2>/dev/null | head -10'
            ], capture_output=True, text=True)
            
            correlation_files = result.stdout.strip()
            
            if correlation_files:
                # 读取最新的相关性报告
                latest_file_result = subprocess.run([
                    'sshpass', '-p', 'Abab1234', 'ssh', '-o', 'StrictHostKeyChecking=no',
                    'ecs-user@8.216.35.79',
                    f'cd /nas/factor_forecasting && ls -t outputs/*correlation* 2>/dev/null | head -1 | xargs cat 2>/dev/null || echo "无法读取相关性文件"'
                ], capture_output=True, text=True)
                
                return {
                    'files': correlation_files,
                    'latest_content': latest_file_result.stdout.strip()
                }
            
            return None
            
        except Exception as e:
            return f"相关性报告检查失败: {e}"

    def format_gpu_status(self, gpu_status):
        """格式化GPU状态显示"""
        if not gpu_status or 'GPU状态获取失败' in gpu_status:
            return gpu_status
            
        lines = gpu_status.split('\n')
        formatted = "📊 4GPU使用状态:\n"
        formatted += "GPU | 显存使用      | GPU利用率 | 内存利用率 | 功耗   | 温度\n"
        formatted += "-" * 75 + "\n"
        
        total_memory_used = 0
        total_memory_total = 0
        active_gpus = 0
        
        for line in lines:
            if line.strip() and ',' in line:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 8:
                    gpu_id, name, mem_used, mem_total, util_gpu, util_mem, power, temp = parts
                    total_memory_used += int(mem_used)
                    total_memory_total += int(mem_total)
                    
                    if int(util_gpu) > 0:
                        active_gpus += 1
                        status = "🟢"
                    else:
                        status = "🔴"
                    
                    formatted += f"{status} {gpu_id}  | {mem_used:>4}MB/{mem_total:>5}MB | {util_gpu:>6}%     | {util_mem:>7}%      | {power:>4}W  | {temp:>2}°C\n"
        
        # GPU使用总结
        total_util = (total_memory_used / total_memory_total) * 100 if total_memory_total > 0 else 0
        formatted += f"\n📈 GPU总结: {active_gpus}/4个GPU活跃, 总显存利用率: {total_util:.1f}%\n"
        formatted += f"💾 总显存使用: {total_memory_used}MB / {total_memory_total}MB\n"
        
        return formatted

    def run_continuous_monitor(self):
        """运行持续监控"""
        print("🚀 4GPU分布式训练持续监控系统启动")
        print("=" * 100)
        print(f"⏰ 监控间隔: {self.monitor_interval}秒")
        print(f"📊 相关性报告间隔: {self.correlation_interval}秒 (2小时)")
        print("=" * 100)
        
        while True:
            try:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                current_time = time.time()
                
                print(f"\n[{timestamp}] 🔍 4GPU分布式训练状态检查")
                print("=" * 100)
                
                # 1. 检查训练进程
                processes = self.get_training_processes()
                if processes and "unified_complete" in processes:
                    print("✅ 训练进程运行中:")
                    for line in processes.split('\n'):
                        if 'unified_complete' in line or 'torchrun' in line:
                            # 提取关键信息
                            parts = line.split()
                            if len(parts) > 2:
                                cpu_usage = parts[2] if '%' not in parts[2] else parts[2]
                                memory_usage = parts[3] if parts[3] != cpu_usage else parts[4]
                                print(f"  📊 PID {parts[1]}: CPU {cpu_usage}%, 内存 {memory_usage}%")
                else:
                    print("❌ 未找到训练进程")
                
                # 2. GPU状态
                gpu_status = self.get_server_status()
                formatted_gpu = self.format_gpu_status(gpu_status)
                print(f"\n{formatted_gpu}")
                
                # 3. 训练进度和epoch时间
                log_text = self.get_latest_training_log()
                epoch_info = self.extract_epoch_info(log_text)
                
                if epoch_info:
                    print(f"\n⏱️  训练进度信息:")
                    print(f"   当前状态: Epoch {epoch_info[0]}, {epoch_info[1]} iterations")
                    
                    # 计算时间估算
                    time_estimate = self.calculate_epoch_time_estimate(epoch_info)
                    if time_estimate:
                        print(f"   平均每iteration: {time_estimate['avg_time_per_iteration']:.2f}秒")
                        print(f"   预计epoch完成时间: {time_estimate['estimated_completion'].strftime('%H:%M:%S')}")
                        remaining_minutes = time_estimate['estimated_remaining_seconds'] / 60
                        print(f"   预计剩余时间: {remaining_minutes:.1f}分钟")
                        
                        # 记录epoch时间
                        if time_estimate['epoch'] not in [et.get('epoch', -1) for et in self.epoch_times]:
                            self.epoch_times.append(time_estimate)
                
                # 4. 检查错误
                if 'error' in log_text.lower() or 'failed' in log_text.lower():
                    error_lines = [line for line in log_text.split('\n') 
                                 if 'error' in line.lower() or 'failed' in line.lower()]
                    if error_lines:
                        print(f"\n⚠️  检测到错误 (最近5条):")
                        for error in error_lines[-5:]:
                            print(f"   {error.strip()}")
                
                # 5. 系统资源
                system_resources = self.get_system_resources()
                if system_resources:
                    print(f"\n💻 系统资源状态:")
                    for line in system_resources.split('\n'):
                        if 'Cpu(s)' in line or 'Mem:' in line or '/nas' in line:
                            print(f"   {line.strip()}")
                
                # 6. 每2小时检查相关性报告
                if current_time - self.last_correlation_check >= self.correlation_interval:
                    print(f"\n📊 检查相关性报告 (每2小时)...")
                    correlation_data = self.check_correlation_reports()
                    
                    if correlation_data and isinstance(correlation_data, dict):
                        if correlation_data.get('files'):
                            print(f"✅ 发现相关性报告文件:")
                            for file_line in correlation_data['files'].split('\n'):
                                if file_line.strip():
                                    print(f"   📄 {file_line.strip()}")
                        
                        if correlation_data.get('latest_content') and '无法读取' not in correlation_data['latest_content']:
                            print(f"\n📈 最新相关性数据:")
                            content_lines = correlation_data['latest_content'].split('\n')
                            for line in content_lines[:10]:  # 显示前10行
                                if line.strip():
                                    print(f"   {line.strip()}")
                    else:
                        print("📊 暂无新的相关性报告")
                    
                    self.last_correlation_check = current_time
                
                # 7. Epoch完成历史
                if self.epoch_times:
                    print(f"\n📈 Epoch完成历史 (最近5个):")
                    for et in self.epoch_times[-5:]:
                        print(f"   Epoch {et['epoch']}: {et['avg_time_per_iteration']:.2f}s/it, {et['current_iterations']} iterations")
                
                print("=" * 100)
                time.sleep(self.monitor_interval)
                
            except KeyboardInterrupt:
                print("\n\n👋 监控结束")
                break
            except Exception as e:
                print(f"\n❌ 监控错误: {e}")
                time.sleep(30)  # 错误时等待30秒再继续

if __name__ == "__main__":
    monitor = ContinuousTrainingMonitor()
    monitor.run_continuous_monitor()
