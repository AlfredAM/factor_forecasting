#!/usr/bin/env python3
"""
简单检查Epoch状态的脚本
"""
import subprocess
import time
import re

def check_training_status():
    """检查训练状态"""
    cmd = [
        'sshpass', '-p', 'Abab1234',
        'ssh', '-o', 'StrictHostKeyChecking=no',
        '-o', 'UserKnownHostsFile=/dev/null',
        '-o', 'PreferredAuthentications=password',
        '-o', 'PubkeyAuthentication=no',
        'ecs-user@47.120.46.105',
        'cd /nas/factor_forecasting && L=$(ls -t logs/*run*.log | head -1) && tail -n 3 "$L"'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"Error: {result.stderr}"
    except Exception as e:
        return f"SSH Error: {e}"

def extract_iteration_info(output):
    """提取iteration信息"""
    # 匹配格式: "Epoch 0 Training: 4616it [2:31:18, 1.48s/it, Loss=0.037378, Avg=0.236601]"
    match = re.search(r'Epoch (\d+).*?(\d+)it \[([^,]+), ([0-9.]+)s/it.*?Loss=([0-9.]+)', output)
    if match:
        return {
            'epoch': int(match.group(1)),
            'iteration': int(match.group(2)),
            'elapsed_time': match.group(3),
            'iter_time': float(match.group(4)),
            'current_loss': float(match.group(5))
        }
    return None

# 检查当前状态
print("🔍 检查当前训练状态...")
status = check_training_status()
print(f"📊 训练状态: {status}")

if status and "Error" not in status:
    info = extract_iteration_info(status)
    if info:
        print(f"\n📈 训练详情:")
        print(f"  Epoch: {info['epoch']}")
        print(f"  当前Iteration: {info['iteration']}")
        print(f"  已运行时间: {info['elapsed_time']}")
        print(f"  每iteration时间: {info['iter_time']:.2f}秒")
        print(f"  当前Loss: {info['current_loss']:.6f}")
        
        # 估算剩余时间
        if info['iteration'] > 0:
            estimated_total = 6000
            remaining = max(0, estimated_total - info['iteration'])
            remaining_minutes = (remaining * info['iter_time']) / 60
            print(f"  预计剩余时间: {remaining_minutes:.1f}分钟")
            print(f"  完成进度: {(info['iteration']/estimated_total)*100:.1f}%")
            
            if info['iteration'] >= estimated_total * 0.95:
                print(f"\n🎯 Epoch 0 即将完成！请等待validation结果...")
            elif info['iteration'] >= estimated_total * 0.8:
                print(f"\n⏰ Epoch 0 接近完成，建议继续监控...")
