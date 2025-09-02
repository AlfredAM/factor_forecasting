#!/usr/bin/env python3
"""
Correlation提取器 - 从训练结果中提取各target的correlation
"""
import subprocess
import json
import re
import time
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

def check_epoch_completion():
    """检查epoch是否完成"""
    # 检查是否有验证阶段开始
    cmd = '''cd /nas/factor_forecasting && L=$(ls -t logs/*run*.log | head -1) && 
    grep -E "(Validation|validation|Epoch 1|验证)" "$L" | tail -5'''
    result = run_ssh_command(cmd)
    
    # 检查iteration数量是否停止增长
    cmd2 = '''cd /nas/factor_forecasting && L=$(ls -t logs/*run*.log | head -1) && 
    tail -n 1 "$L" | grep -o "Epoch 0.*: [0-9]*it"'''
    current_iter = run_ssh_command(cmd2)
    
    return result, current_iter

def extract_loss_trend():
    """提取损失趋势，判断是否收敛"""
    cmd = '''cd /nas/factor_forecasting && L=$(ls -t logs/*run*.log | head -1) && 
    grep -o "Loss=[0-9.]*" "$L" | tail -20 | cut -d= -f2'''
    result = run_ssh_command(cmd)
    
    if result and "Error" not in result:
        losses = [float(x) for x in result.split('\n') if x.strip()]
        return losses
    return []

def compute_correlation_from_logs():
    """尝试从日志中计算correlation（如果有预测输出）"""
    # 搜索任何correlation相关的数值输出
    cmd = '''cd /nas/factor_forecasting && L=$(ls -t logs/*run*.log | head -1) && 
    grep -i -E "(corr[^u]|IC[^C]|pearson|spearman)" "$L" | grep -v "type.*correlation" | grep -E "[0-9]"'''
    result = run_ssh_command(cmd)
    return result

def get_training_statistics():
    """获取训练统计信息"""
    cmd = '''cd /nas/factor_forecasting && L=$(ls -t logs/*run*.log | head -1) && 
    echo "=== 训练时长 ===" &&
    echo "开始时间: $(head -20 "$L" | grep -E "[0-9]{2}:[0-9]{2}:[0-9]{2}" | head -1)" &&
    echo "当前时间: $(date)" &&
    echo "=== iteration统计 ===" &&
    LATEST_ITER=$(tail -n 1 "$L" | grep -o "[0-9]*it" | head -1 | sed "s/it//") &&
    echo "当前iteration: $LATEST_ITER" &&
    echo "=== 损失统计 ===" &&
    LATEST_LOSS=$(tail -n 1 "$L" | grep -o "Loss=[0-9.]*" | cut -d= -f2) &&
    LATEST_AVG=$(tail -n 1 "$L" | grep -o "Avg=[0-9.]*" | cut -d= -f2) &&
    echo "当前Loss: $LATEST_LOSS" &&
    echo "平均Loss: $LATEST_AVG"'''
    
    return run_ssh_command(cmd)

def estimate_completion_time():
    """估算完成时间"""
    cmd = '''cd /nas/factor_forecasting && L=$(ls -t logs/*run*.log | head -1) && 
    LATEST_ITER=$(tail -n 1 "$L" | grep -o "[0-9]*it" | head -1 | sed "s/it//") &&
    TIME_PER_ITER=$(tail -n 1 "$L" | grep -o "[0-9.]*s/it" | cut -d"s" -f1) &&
    echo "$LATEST_ITER,$TIME_PER_ITER"'''
    
    result = run_ssh_command(cmd)
    if result and "," in result:
        try:
            iter_count, time_per_iter = result.split(',')
            current_iter = int(iter_count)
            time_per = float(time_per_iter)
            
            # 基于数据量估算总iteration数
            estimated_total = 4200  # 基于299个文件和batch size的估算
            remaining = max(0, estimated_total - current_iter)
            remaining_time = remaining * time_per / 60  # 分钟
            
            return current_iter, estimated_total, remaining_time
        except:
            pass
    
    return None, None, None

def monitor_and_extract():
    """主函数：监控并提取correlation"""
    print("🔍 开始监控Epoch 0完成状态并提取correlation...")
    print("=" * 70)
    
    last_check_time = 0
    check_interval = 60  # 每分钟检查一次
    
    while True:
        try:
            current_time = time.time()
            
            if current_time - last_check_time >= check_interval:
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"\n[{timestamp}] 📊 训练状态检查")
                print("-" * 50)
                
                # 获取训练统计
                stats = get_training_statistics()
                if stats:
                    print(stats)
                
                # 估算完成时间
                current_iter, total_iter, remaining_min = estimate_completion_time()
                if current_iter and total_iter:
                    progress = (current_iter / total_iter) * 100
                    print(f"\n📈 进度: {current_iter}/{total_iter} ({progress:.1f}%)")
                    if remaining_min:
                        print(f"⏱️ 预计剩余: {remaining_min:.1f} 分钟")
                
                # 检查epoch完成
                validation_info, iter_info = check_epoch_completion()
                if validation_info and len(validation_info.strip()) > 0:
                    print(f"\n🎉 检测到验证阶段或Epoch完成!")
                    print(f"验证信息: {validation_info}")
                    break
                
                # 检查correlation输出
                corr_info = compute_correlation_from_logs()
                if corr_info and len(corr_info.strip()) > 0:
                    print(f"\n📊 发现correlation信息:")
                    print(corr_info)
                
                # 检查是否接近完成（基于iteration数）
                if current_iter and current_iter >= 4000:
                    print(f"\n⚠️ 接近预期完成点 (iteration {current_iter})")
                    print("继续监控verification阶段...")
                
                last_check_time = current_time
                print("=" * 50)
            
            time.sleep(15)  # 短间隔轮询
            
        except KeyboardInterrupt:
            print("\n🛑 监控中断")
            break
        except Exception as e:
            print(f"❌ 监控错误: {e}")
            time.sleep(30)
    
    # Epoch完成后的详细分析
    print("\n" + "=" * 70)
    print("📊 Epoch 0 训练完成分析")
    print("=" * 70)
    
    # 最终损失趋势
    losses = extract_loss_trend()
    if losses:
        print(f"\n📉 损失收敛分析:")
        print(f"最早损失: {losses[0]:.6f}")
        print(f"最终损失: {losses[-1]:.6f}")
        print(f"收敛幅度: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")
        
        # 计算收敛稳定性
        recent_losses = losses[-5:]
        if len(recent_losses) >= 5:
            volatility = sum(abs(recent_losses[i] - recent_losses[i-1]) for i in range(1, len(recent_losses))) / len(recent_losses)
            print(f"最近稳定性: {volatility:.6f} (越小越稳定)")
    
    # 检查最终correlation结果
    final_correlation = compute_correlation_from_logs()
    if final_correlation:
        print(f"\n📈 Correlation结果:")
        print(final_correlation)
    else:
        print("\n❌ 未发现直接的correlation输出")
        print("💡 检查是否需要运行验证阶段来计算correlation...")
    
    # 检查输出文件
    output_check = run_ssh_command('find /nas/factor_forecasting/outputs/ -name "*.json" -o -name "*.csv" -mmin -30 | head -5')
    if output_check:
        print(f"\n📁 最新输出文件:")
        print(output_check)
    
    print("\n🏁 监控分析完成!")

if __name__ == "__main__":
    monitor_and_extract()
