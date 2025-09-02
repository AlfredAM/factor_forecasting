#!/usr/bin/env python3
"""
监控Epoch 0完成并检查各target的correlation
"""
import subprocess
import time
import re
import json
from datetime import datetime

def run_ssh_command(cmd):
    """执行SSH命令"""
    ssh_cmd = [
        'sshpass', '-p', 'Abab1234',
        'ssh', '-o', 'StrictHostKeyChecking=no',
        '-o', 'UserKnownHostsFile=/dev/null',
        '-o', 'PreferredAuthentications=password',
        '-o', 'PubkeyAuthentication=no',
        'ecs-user@47.120.46.105',
        cmd
    ]
    
    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=30)
        return result.stdout.strip() if result.returncode == 0 else f"Error: {result.stderr}"
    except Exception as e:
        return f"SSH Error: {e}"

def get_training_progress():
    """获取训练进度"""
    cmd = "cd /nas/factor_forecasting && L=$(ls -t logs/*run*.log | head -1) && tail -n 3 \"$L\""
    return run_ssh_command(cmd)

def check_epoch_completion():
    """检查epoch是否完成"""
    cmd = 'cd /nas/factor_forecasting && L=$(ls -t logs/*run*.log | head -1) && grep -E "(Epoch.*completed|Epoch.*finished|epoch.*time|验证.*完成|Validation.*completed)" "$L" | tail -5'
    result = run_ssh_command(cmd)
    return result if result and "Error" not in result and result.strip() else None

def check_validation_results():
    """检查验证结果和correlation"""
    cmd = '''cd /nas/factor_forecasting && L=$(ls -t logs/*run*.log | head -1) && grep -E "(correlation|IC|validation.*loss|val_loss)" "$L" | tail -10'''
    return run_ssh_command(cmd)

def extract_correlations_from_logs():
    """从日志中提取correlation信息"""
    cmd = '''cd /nas/factor_forecasting && L=$(ls -t logs/*run*.log | head -1) && grep -E "(intra30m.*correlation|nextT1d.*correlation|ema1d.*correlation|Pearson|Spearman|IC)" "$L" | tail -15'''
    return run_ssh_command(cmd)

def estimate_completion_time():
    """估算epoch完成时间"""
    progress = get_training_progress()
    if "it [" in progress:
        # 提取iteration信息: "Epoch 0 Training: 4567it [2:29:49, 1.53s/it, ..."
        match = re.search(r'(\d+)it \[([^,]+), ([0-9.]+)s/it', progress)
        if match:
            current_iter = int(match.group(1))
            elapsed_time = match.group(2)
            iter_time = float(match.group(3))
            
            # 估算总iteration数（基于数据集大小）
            estimated_total = 6000  # 根据之前的观察调整
            remaining_iters = max(0, estimated_total - current_iter)
            remaining_seconds = remaining_iters * iter_time
            
            return {
                'current_iter': current_iter,
                'elapsed_time': elapsed_time,
                'iter_time': iter_time,
                'estimated_remaining_minutes': remaining_seconds / 60,
                'progress_percentage': (current_iter / estimated_total) * 100
            }
    return None

def monitor_epoch_completion():
    """主监控函数"""
    print("🚀 开始监控Epoch 0完成状态...")
    print("=" * 60)
    
    epoch_completed = False
    last_check_time = 0
    check_interval = 60  # 每分钟检查一次
    
    while not epoch_completed:
        current_time = time.time()
        
        if current_time - last_check_time >= check_interval:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n📊 [{timestamp}] 检查训练状态")
            print("-" * 40)
            
            # 检查训练进度
            progress = get_training_progress()
            if progress and "Error" not in progress:
                print("📈 当前训练进度:")
                for line in progress.split('\n')[-3:]:
                    if line.strip():
                        print(f"  {line.strip()}")
                
                # 估算完成时间
                completion_info = estimate_completion_time()
                if completion_info:
                    print(f"\n⏱️ 进度估算:")
                    print(f"  当前iteration: {completion_info['current_iter']}")
                    print(f"  已运行时间: {completion_info['elapsed_time']}")
                    print(f"  每iteration: {completion_info['iter_time']:.2f}秒")
                    print(f"  预计剩余: {completion_info['estimated_remaining_minutes']:.1f}分钟")
                    print(f"  完成度: {completion_info['progress_percentage']:.1f}%")
            
            # 检查epoch完成
            completion_status = check_epoch_completion()
            if completion_status:
                print(f"\n🎉 Epoch完成检测:")
                print(f"  {completion_status}")
                epoch_completed = True
                break
            
            # 检查validation结果
            validation_results = check_validation_results()
            if validation_results and validation_results.strip():
                print(f"\n📋 验证结果:")
                for line in validation_results.split('\n'):
                    if line.strip():
                        print(f"  {line.strip()}")
            
            last_check_time = current_time
        
        if not epoch_completed:
            print(f"⏳ Epoch 0 尚未完成，{check_interval}秒后再次检查...")
            time.sleep(check_interval)
    
    # Epoch完成后，检查correlation
    print("\n" + "=" * 60)
    print("🎯 Epoch 0 已完成！开始检查各target的correlation...")
    print("=" * 60)
    
    # 等待几秒让validation完成
    time.sleep(10)
    
    # 检查correlation结果
    correlation_results = extract_correlations_from_logs()
    if correlation_results and correlation_results.strip():
        print("\n📊 各target的correlation结果:")
        print("-" * 40)
        for line in correlation_results.split('\n'):
            if line.strip():
                print(f"  {line.strip()}")
    else:
        print("❌ 未找到correlation结果，检查日志文件...")
        
        # 尝试获取更多信息
        cmd = 'cd /nas/factor_forecasting && L=$(ls -t logs/*run*.log | head -1) && tail -n 50 "$L" | grep -E "(target|correlation|IC|validation)"'
        additional_info = run_ssh_command(cmd)
        if additional_info:
            print("📋 额外信息:")
            for line in additional_info.split('\n'):
                if line.strip():
                    print(f"  {line.strip()}")
    
    # 检查输出文件
    print("\n📁 检查输出文件:")
    cmd = 'cd /nas/factor_forecasting && find outputs/ -type f -mmin -30 | head -10'
    output_files = run_ssh_command(cmd)
    if output_files and output_files.strip():
        print("最近生成的输出文件:")
        for line in output_files.split('\n'):
            if line.strip():
                print(f"  {line.strip()}")
    
    print(f"\n✅ 监控完成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    monitor_epoch_completion()