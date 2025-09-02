#!/usr/bin/env python3
"""
Epoch完成监控器 - 监控Epoch 0完成并获取correlation结果
"""
import subprocess
import time
import re
from datetime import datetime

def run_ssh_command(cmd: str) -> str:
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

def get_latest_progress():
    """获取最新训练进度"""
    cmd = 'cd /nas/factor_forecasting && L=$(ls -t logs/*run*.log | head -1) && tail -n 5 "$L"'
    return run_ssh_command(cmd)

def check_epoch_completion():
    """检查epoch是否完成"""
    cmd = '''cd /nas/factor_forecasting && L=$(ls -t logs/*run*.log | head -1) && 
    grep -E "(Epoch.*completed|Epoch.*finished|Epoch 1|validation.*epoch.*0|epoch.*0.*time)" "$L" | tail -3'''
    result = run_ssh_command(cmd)
    return result and len(result.strip()) > 0

def extract_correlation_metrics():
    """提取correlation指标"""
    cmd = '''cd /nas/factor_forecasting && L=$(ls -t logs/*run*.log | head -1) && 
    grep -i -E "(correlation|IC|pearson|spearman|相关)" "$L" | grep -v "type.*correlation" | tail -10'''
    return run_ssh_command(cmd)

def get_validation_outputs():
    """获取验证输出"""
    cmd = '''cd /nas/factor_forecasting && L=$(ls -t logs/*run*.log | head -1) && 
    grep -A 10 -B 10 -i -E "(validation|valid|验证)" "$L" | tail -20'''
    return run_ssh_command(cmd)

def parse_iteration_progress(log_text):
    """解析iteration进度"""
    pattern = r'Epoch 0 Training: (\d+)it.*?Loss=([0-9.]+).*?Avg=([0-9.]+)'
    matches = re.findall(pattern, log_text)
    if matches:
        latest = matches[-1]
        return int(latest[0]), float(latest[1]), float(latest[2])
    return None, None, None

def monitor_epoch_completion():
    """主监控函数"""
    print("🔍 开始监控Epoch 0完成状态...")
    print("=" * 60)
    
    start_time = time.time()
    last_iteration = 0
    check_interval = 30  # 每30秒检查一次
    
    while True:
        try:
            current_time = datetime.now().strftime("%H:%M:%S")
            
            # 获取最新进度
            progress = get_latest_progress()
            
            if progress and "Error" not in progress:
                # 解析进度
                iteration, loss, avg_loss = parse_iteration_progress(progress)
                
                if iteration:
                    print(f"[{current_time}] Iteration {iteration}, Loss: {loss:.6f}, Avg: {avg_loss:.6f}")
                    
                    # 检查进度变化
                    if iteration > last_iteration:
                        last_iteration = iteration
                        
                        # 估算剩余时间
                        if iteration > 3500:  # 接近完成
                            estimated_total = max(4000, iteration + 100)
                            remaining = estimated_total - iteration
                            print(f"    📊 预计剩余 {remaining} iterations")
                    
                    else:
                        print("    ⚠️ iteration无变化，可能已完成或停滞")
                
                # 检查是否有epoch完成标记
                completion_check = check_epoch_completion()
                if completion_check and len(completion_check.strip()) > 0:
                    print(f"\n🎉 检测到Epoch完成标记!")
                    print(f"完成信息: {completion_check}")
                    break
                
                # 检查是否进入验证阶段
                validation_check = get_validation_outputs()
                if validation_check and "validation" in validation_check.lower():
                    print(f"\n📈 检测到验证阶段!")
                    print(f"验证信息: {validation_check}")
                
            else:
                print(f"[{current_time}] ❌ 无法获取训练进度: {progress}")
            
            # 运行时间统计
            total_time = time.time() - start_time
            if total_time > 3600:  # 超过1小时
                hours = int(total_time // 3600)
                minutes = int((total_time % 3600) // 60)
                print(f"    ⏱️ 已监控 {hours}h {minutes}m")
            
            print("-" * 40)
            time.sleep(check_interval)
            
        except KeyboardInterrupt:
            print("\n🛑 监控中断")
            break
        except Exception as e:
            print(f"❌ 监控错误: {e}")
            time.sleep(10)
    
    # Epoch完成后获取详细信息
    print("\n" + "=" * 60)
    print("📊 Epoch 0 完成 - 获取详细correlation信息")
    print("=" * 60)
    
    # 获取correlation指标
    correlation_metrics = extract_correlation_metrics()
    if correlation_metrics:
        print("\n📈 Correlation指标:")
        print(correlation_metrics)
    else:
        print("❌ 未找到correlation指标")
    
    # 获取最终验证结果
    final_validation = get_validation_outputs()
    if final_validation:
        print("\n✅ 验证结果:")
        print(final_validation)
    else:
        print("❌ 未找到验证结果")
    
    # 检查输出文件
    output_files = run_ssh_command('find /nas/factor_forecasting/outputs/ -name "*" -type f -mmin -30')
    if output_files:
        print(f"\n📁 最新输出文件:")
        print(output_files)
    
    print("\n🏁 监控完成!")

if __name__ == "__main__":
    monitor_epoch_completion()
