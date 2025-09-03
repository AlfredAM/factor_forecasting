#!/usr/bin/env python3
"""
综合监控系统 - 持续监控训练状态、硬件利用率和相关性报告
"""

import subprocess
import time
import json
import re
from datetime import datetime, timedelta

class ComprehensiveMonitor:
    def __init__(self):
        self.server_ip = "8.216.35.79"
        self.password = "Abab1234"
        self.project_path = "/nas/factor_forecasting"
        self.last_correlation_check = 0
        self.epoch_times = []
        
    def ssh_execute(self, command):
        """执行SSH命令"""
        try:
            result = subprocess.run([
                'sshpass', '-p', self.password, 'ssh', '-o', 'StrictHostKeyChecking=no',
                f'ecs-user@{self.server_ip}', command
            ], capture_output=True, text=True, timeout=30)
            return result.returncode == 0, result.stdout, result.stderr
        except Exception as e:
            return False, "", str(e)
    
    def get_training_status(self):
        """获取训练状态"""
        # 检查进程
        success, proc_output, _ = self.ssh_execute(
            'ps aux | grep "python.*unified_complete_training" | grep -v grep'
        )
        
        training_active = success and proc_output.strip()
        
        # GPU状态
        success, gpu_output, _ = self.ssh_execute(
            "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits"
        )
        
        gpu_status = {}
        if success and gpu_output.strip():
            for line in gpu_output.strip().split('\n'):
                parts = line.split(', ')
                if len(parts) >= 5:
                    gpu_id, mem_used, mem_total, util, temp = parts
                    gpu_status[int(gpu_id)] = {
                        'memory_used_mb': int(mem_used),
                        'memory_total_mb': int(mem_total),
                        'utilization_pct': int(util),
                        'temperature_c': int(temp),
                        'memory_usage_pct': int(mem_used) / int(mem_total) * 100
                    }
        
        # 系统内存
        success, mem_output, _ = self.ssh_execute("free -g | grep Mem:")
        memory_status = {}
        if success and mem_output.strip():
            parts = mem_output.strip().split()
            if len(parts) >= 7:
                total, used, free = int(parts[1]), int(parts[2]), int(parts[3])
                memory_status = {
                    'total_gb': total,
                    'used_gb': used,
                    'free_gb': free,
                    'usage_pct': used / total * 100 if total > 0 else 0
                }
        
        # 训练日志
        success, log_output, _ = self.ssh_execute(
            f"cd {self.project_path} && tail -20 training_high_performance.log 2>/dev/null || tail -20 *.log | tail -20"
        )
        
        training_metrics = self.parse_training_log(log_output if success else "")
        
        return {
            'timestamp': datetime.now(),
            'training_active': training_active,
            'process_info': proc_output.strip() if training_active else "",
            'gpu_status': gpu_status,
            'memory_status': memory_status,
            'training_metrics': training_metrics
        }
    
    def parse_training_log(self, log_output):
        """解析训练日志"""
        metrics = {}
        
        if not log_output:
            return metrics
        
        # 查找epoch信息
        epoch_pattern = r'Epoch (\d+) Training: (\d+)it \[([^,]+), ([^]]+)\]'
        matches = re.findall(epoch_pattern, log_output)
        
        if matches:
            epoch, iterations, time_elapsed, time_per_it = matches[-1]
            metrics['current_epoch'] = int(epoch)
            metrics['iterations'] = int(iterations)
            metrics['time_elapsed'] = time_elapsed
            metrics['time_per_iteration'] = time_per_it
            
            # 计算epoch完成时间
            if 's/it' in time_per_it:
                try:
                    sec_per_it = float(time_per_it.split('s/it')[0])
                    metrics['seconds_per_iteration'] = sec_per_it
                    
                    # 估算epoch总时间（假设1000个iteration per epoch）
                    estimated_total_time = 1000 * sec_per_it
                    elapsed_seconds = self.parse_time_to_seconds(time_elapsed)
                    remaining_seconds = max(0, estimated_total_time - elapsed_seconds)
                    
                    metrics['estimated_epoch_time_minutes'] = estimated_total_time / 60
                    metrics['remaining_time_minutes'] = remaining_seconds / 60
                    
                except (ValueError, IndexError):
                    pass
        
        # 检查错误
        if 'CUDA out of memory' in log_output:
            metrics['memory_error'] = True
        
        if 'ERROR' in log_output:
            metrics['has_errors'] = True
            # 提取最新错误
            error_lines = [line for line in log_output.split('\n') if 'ERROR' in line]
            if error_lines:
                metrics['latest_error'] = error_lines[-1]
        
        # 检查成功完成的epoch
        epoch_complete_pattern = r'Epoch (\d+) completed'
        completed_epochs = re.findall(epoch_complete_pattern, log_output)
        if completed_epochs:
            metrics['completed_epochs'] = [int(e) for e in completed_epochs]
        
        return metrics
    
    def parse_time_to_seconds(self, time_str):
        """将时间字符串转换为秒"""
        try:
            if ':' in time_str:
                parts = time_str.split(':')
                if len(parts) == 2:  # MM:SS
                    return int(parts[0]) * 60 + int(parts[1])
                elif len(parts) == 3:  # HH:MM:SS
                    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        except ValueError:
            pass
        return 0
    
    def check_correlation_reports(self):
        """检查相关性报告"""
        success, output, _ = self.ssh_execute(
            f"cd {self.project_path} && find outputs/ -name '*.json' -mtime -1 2>/dev/null | head -5"
        )
        
        reports = []
        if success and output.strip():
            for file_path in output.strip().split('\n'):
                if file_path.strip():
                    success, content, _ = self.ssh_execute(f"cat {file_path}")
                    if success:
                        try:
                            report = json.loads(content)
                            reports.append({
                                'file': file_path,
                                'timestamp': report.get('timestamp', 'Unknown'),
                                'correlations': {
                                    target: {
                                        'in_sample': report.get(f'{target}_in_sample_ic', 'N/A'),
                                        'out_sample': report.get(f'{target}_out_sample_ic', 'N/A')
                                    }
                                    for target in ['intra30m', 'nextT1d', 'ema1d']
                                }
                            })
                        except json.JSONDecodeError:
                            pass
        
        return reports
    
    def display_status(self, status):
        """显示系统状态"""
        timestamp = status['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"\n{'='*80}")
        print(f"🚀 因子预测模型训练监控 - {timestamp}")
        print(f"{'='*80}")
        
        # 训练状态
        if status['training_active']:
            print("✅ 训练状态: 运行中")
            if status['process_info']:
                # 提取CPU和内存使用
                parts = status['process_info'].split()
                if len(parts) >= 11:
                    cpu_pct = parts[2]
                    mem_pct = parts[3]
                    pid = parts[1]
                    print(f"   进程ID: {pid} | CPU: {cpu_pct}% | 内存: {mem_pct}%")
        else:
            print("❌ 训练状态: 未运行")
        
        # GPU状态
        print(f"\n🔥 GPU状态:")
        for gpu_id, gpu_info in status['gpu_status'].items():
            mem_used = gpu_info['memory_used_mb']
            mem_total = gpu_info['memory_total_mb']
            util = gpu_info['utilization_pct']
            temp = gpu_info['temperature_c']
            mem_pct = gpu_info['memory_usage_pct']
            
            status_icon = "🔥" if mem_pct > 90 else "⚡" if mem_pct > 50 else "💤"
            print(f"   {status_icon} GPU {gpu_id}: {mem_used:,}MB/{mem_total:,}MB ({mem_pct:.1f}%) | {util}% util | {temp}°C")
        
        # 系统内存
        if status['memory_status']:
            mem_info = status['memory_status']
            mem_icon = "🔥" if mem_info['usage_pct'] > 80 else "📊"
            print(f"\n{mem_icon} 系统内存: {mem_info['used_gb']}GB/{mem_info['total_gb']}GB ({mem_info['usage_pct']:.1f}%)")
        
        # 训练指标
        metrics = status['training_metrics']
        if metrics:
            print(f"\n📈 训练进度:")
            
            if 'current_epoch' in metrics:
                epoch = metrics['current_epoch']
                iterations = metrics['iterations']
                time_elapsed = metrics['time_elapsed']
                print(f"   当前Epoch: {epoch} | 完成迭代: {iterations} | 已用时间: {time_elapsed}")
                
                if 'seconds_per_iteration' in metrics:
                    sec_per_it = metrics['seconds_per_iteration']
                    print(f"   每次迭代: {sec_per_it:.2f}s")
                    
                    if 'estimated_epoch_time_minutes' in metrics:
                        total_time = metrics['estimated_epoch_time_minutes']
                        remaining_time = metrics.get('remaining_time_minutes', 0)
                        print(f"   预计Epoch总时间: {total_time:.1f}分钟")
                        print(f"   预计剩余时间: {remaining_time:.1f}分钟")
            
            if metrics.get('completed_epochs'):
                completed = metrics['completed_epochs']
                print(f"   ✅ 已完成Epoch: {completed}")
            
            if metrics.get('memory_error'):
                print("   ⚠️  检测到CUDA内存错误")
            
            if metrics.get('has_errors'):
                print("   ❌ 检测到训练错误:")
                if 'latest_error' in metrics:
                    print(f"      {metrics['latest_error']}")
        
        print(f"{'='*80}")
    
    def display_correlation_reports(self, reports):
        """显示相关性报告"""
        if not reports:
            print("📊 暂无相关性报告")
            return
        
        print(f"\n📊 相关性报告 ({len(reports)}个):")
        print("-" * 60)
        
        for report in reports[-3:]:  # 显示最新的3个报告
            print(f"📈 报告时间: {report['timestamp']}")
            for target, corr in report['correlations'].items():
                in_sample = corr['in_sample']
                out_sample = corr['out_sample']
                
                if isinstance(in_sample, (int, float)) and isinstance(out_sample, (int, float)):
                    print(f"   {target:10s}: In-sample={in_sample:7.4f} | Out-sample={out_sample:7.4f}")
                else:
                    print(f"   {target:10s}: In-sample={str(in_sample):>7s} | Out-sample={str(out_sample):>7s}")
            print("-" * 60)
    
    def monitor_continuously(self):
        """持续监控"""
        print("🚀 启动综合监控系统...")
        print("   - 实时监控训练状态和硬件利用率")
        print("   - 每2小时检查相关性报告")
        print("   - 自动记录epoch完成时间")
        print("   - Ctrl+C 停止监控")
        
        try:
            while True:
                current_time = time.time()
                
                # 获取系统状态
                status = self.get_training_status()
                
                # 显示状态
                self.display_status(status)
                
                # 记录epoch时间
                if status['training_metrics'].get('completed_epochs'):
                    completed_epochs = status['training_metrics']['completed_epochs']
                    for epoch in completed_epochs:
                        if epoch not in [e['epoch'] for e in self.epoch_times]:
                            self.epoch_times.append({
                                'epoch': epoch,
                                'completion_time': datetime.now(),
                                'total_time_minutes': status['training_metrics'].get('estimated_epoch_time_minutes', 0)
                            })
                            print(f"✅ Epoch {epoch} 完成记录已保存")
                
                # 每2小时检查相关性报告
                if current_time - self.last_correlation_check >= 7200:
                    print("\n🔍 检查相关性报告...")
                    reports = self.check_correlation_reports()
                    self.display_correlation_reports(reports)
                    self.last_correlation_check = current_time
                
                # 等待60秒
                print(f"\n⏱️  下次检查: {(datetime.now() + timedelta(seconds=60)).strftime('%H:%M:%S')}")
                time.sleep(60)
                
        except KeyboardInterrupt:
            print("\n\n👋 监控已停止")
            
            # 显示epoch完成统计
            if self.epoch_times:
                print(f"\n📊 Epoch完成统计:")
                for epoch_info in self.epoch_times[-5:]:  # 显示最近5个
                    epoch = epoch_info['epoch']
                    completion_time = epoch_info['completion_time'].strftime('%H:%M:%S')
                    total_time = epoch_info['total_time_minutes']
                    print(f"   Epoch {epoch}: 完成时间 {completion_time} | 总用时 {total_time:.1f}分钟")
        
        except Exception as e:
            print(f"\n❌ 监控错误: {e}")

def main():
    """主函数"""
    monitor = ComprehensiveMonitor()
    monitor.monitor_continuously()

if __name__ == "__main__":
    main()
