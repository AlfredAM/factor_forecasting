#!/usr/bin/env python3
"""
Epoch完成监控脚本 - 等待epoch完成并提取correlation信息
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
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"

def get_latest_training_progress():
    """获取最新训练进度"""
    cmd = 'cd /nas/factor_forecasting && L=$(ls -t logs/*run*.log | head -1) && tail -n 3 "$L"'
    return run_ssh_command(cmd)

def check_validation_outputs():
    """检查验证阶段输出"""
    cmd = 'cd /nas/factor_forecasting && find outputs/ -name "*.json" -mmin -30 | head -5'
    return run_ssh_command(cmd)

def search_correlation_in_logs():
    """在日志中搜索correlation信息"""
    cmd = 'cd /nas/factor_forecasting && L=$(ls -t logs/*run*.log | head -1) && grep -i "correlation\\|ic.*[0-9]" "$L" | tail -10'
    return run_ssh_command(cmd)

def extract_iteration_number(progress_text):
    """从进度文本中提取iteration数"""
    match = re.search(r'Epoch 0 Training: (\d+)it', progress_text)
    return int(match.group(1)) if match else 0

def monitor_epoch_completion():
    """监控epoch完成"""
    print("🔍 开始监控Epoch 0完成状态...")
    print("=" * 60)
    
    last_iteration = 0
    stall_count = 0
    
    while True:
        try:
            current_time = datetime.now().strftime("%H:%M:%S")
            
            # 获取最新进度
            progress = get_latest_training_progress()
            current_iteration = extract_iteration_number(progress)
            
            # 检查进度
            if current_iteration > last_iteration:
                print(f"[{current_time}] 📈 Progress: {current_iteration} iterations")
                last_iteration = current_iteration
                stall_count = 0
                
                # 检查是否接近validation点 (每500次)
                if current_iteration % 500 < 10:
                    print(f"📊 接近validation点 ({current_iteration}), 检查输出...")
                    correlation_info = search_correlation_in_logs()
                    if correlation_info and "Error" not in correlation_info:
                        print("🎯 发现相关性信息:")
                        print(correlation_info)
                        
            elif current_iteration == last_iteration:
                stall_count += 1
                if stall_count > 5:
                    print(f"[{current_time}] ⚠️ 训练可能停滞在 {current_iteration} iterations")
                    
            # 检查是否有新的validation输出
            validation_files = check_validation_outputs()
            if validation_files and "Error" not in validation_files and validation_files.strip():
                print(f"📁 发现新的validation文件:")
                print(validation_files)
            
            # 检查epoch完成迹象
            if "Epoch 1" in progress or "epoch.*complete" in progress.lower():
                print("🎉 Epoch 0 已完成！")
                break
                
            # 如果iteration数很高，可能数据集很大
            if current_iteration > 15000:
                print(f"📈 当前 {current_iteration} iterations，数据集较大，继续等待...")
                
            time.sleep(30)  # 每30秒检查一次
            
        except KeyboardInterrupt:
            print("\n⏹️ 监控已停止")
            break
        except Exception as e:
            print(f"❌ 错误: {e}")
            time.sleep(10)
    
    # 最终检查correlation信息
    print("\n" + "=" * 60)
    print("🔍 最终检查correlation信息...")
    
    # 检查所有可能的输出位置
    final_correlation = search_correlation_in_logs()
    if final_correlation and "Error" not in final_correlation:
        print("📊 在日志中发现的correlation信息:")
        print(final_correlation)
    
    # 检查输出文件
    output_check = run_ssh_command('cd /nas/factor_forecasting && find outputs/ -name "*.json" | head -10')
    if output_check and "Error" not in output_check:
        print("📁 输出文件:")
        print(output_check)

if __name__ == "__main__":
    monitor_epoch_completion()