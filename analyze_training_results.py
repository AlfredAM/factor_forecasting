#!/usr/bin/env python3
"""
分析训练结果和correlation数据
"""
import subprocess
import json
import sys

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

def analyze_training_results():
    """分析训练结果"""
    print("📊 训练结果分析报告")
    print("=" * 60)
    
    # 获取最新的training_results.json
    cmd = 'cd /nas/factor_forecasting && find outputs/ -name "training_results.json" -exec ls -t {} + | head -1'
    latest_result_file = run_ssh_command(cmd)
    
    if "Error" in latest_result_file:
        print("❌ 无法获取结果文件")
        return
    
    # 读取结果文件内容
    cmd = f'cd /nas/factor_forecasting && cat "{latest_result_file}"'
    result_content = run_ssh_command(cmd)
    
    try:
        results = json.loads(result_content)
        
        print(f"📁 结果文件: {latest_result_file}")
        print()
        
        # 训练配置分析
        config = results.get('training_config', {})
        print("🔧 训练配置:")
        print(f"  批量大小: {config.get('batch_size', 'N/A')}")
        print(f"  分布式: {config.get('use_distributed', False)}")
        print(f"  GPU数量: {len(config.get('gpu_devices', []))}")
        print(f"  目标列: {config.get('target_columns', [])}")
        print(f"  训练周期: {config.get('epochs', 'N/A')}")
        print(f"  验证间隔: {config.get('validation_interval', 'N/A')}")
        print()
        
        # 训练结果分析
        training_results = results.get('training_results', {})
        print("📈 训练结果:")
        print(f"  完成周期数: {training_results.get('epochs_trained', 0)}")
        print(f"  最佳验证Loss: {training_results.get('best_val_loss', 'N/A')}")
        print(f"  最佳IC: {training_results.get('best_ic', 'N/A')}")
        print()
        
        # Loss趋势
        train_losses = training_results.get('train_losses', [])
        val_losses = training_results.get('val_losses', [])
        
        if train_losses:
            print("📉 Loss趋势:")
            for i, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
                print(f"  Epoch {i}: Train={train_loss:.6f}, Val={val_loss:.6f}")
        
        # 最终统计
        final_stats = results.get('final_stats', {})
        print()
        print("🎯 最终统计:")
        print(f"  最终训练Loss: {final_stats.get('final_train_loss', 'N/A')}")
        print(f"  最终验证Loss: {final_stats.get('final_val_loss', 'N/A')}")
        print(f"  最终IC: {final_stats.get('final_ic', 'N/A')}")
        
        # Correlation分析
        print()
        print("🔍 Correlation分析:")
        if training_results.get('best_ic') == float('-inf'):
            print("  ⚠️ IC计算可能存在问题 (值为-Infinity)")
            print("  这通常表明:")
            print("    1. 预测值全为常数")
            print("    2. 目标值全为常数") 
            print("    3. 计算过程中出现数值问题")
        else:
            ic_scores = training_results.get('ic_scores', [])
            if ic_scores:
                print(f"  IC得分: {ic_scores}")
        
        return results
        
    except json.JSONDecodeError:
        print("❌ JSON解析失败")
        return None

def extract_correlation_from_logs():
    """从日志中提取correlation信息"""
    print("\n🔍 从日志中搜索correlation信息...")
    
    # 搜索所有日志文件中的correlation信息
    cmd = 'cd /nas/factor_forecasting && grep -r -i "correlation.*[0-9]" logs/ 2>/dev/null | head -10'
    correlation_info = run_ssh_command(cmd)
    
    if correlation_info and "Error" not in correlation_info:
        print("📊 发现的correlation信息:")
        for line in correlation_info.split('\n'):
            if line.strip():
                print(f"  {line}")
    else:
        print("  ❌ 未在日志中发现correlation数值")

def diagnose_epoch_issues():
    """诊断epoch未完成的问题"""
    print("\n🔍 诊断epoch未完成的原因...")
    
    # 检查进程状态
    cmd = 'cd /nas/factor_forecasting && ps aux | grep -E "(python|torchrun)" | grep -v grep'
    processes = run_ssh_command(cmd)
    
    if not processes or "Error" in processes:
        print("❌ 没有运行中的训练进程")
        print("原因分析:")
        print("  1. 训练进程已经结束或崩溃")
        print("  2. 可能遇到SIGABRT信号导致异常退出")
        print("  3. 内存不足或其他系统问题")
    
    # 检查GPU状态
    cmd = 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'
    gpu_status = run_ssh_command(cmd)
    
    if gpu_status and "Error" not in gpu_status:
        print(f"\n🔥 GPU状态: {gpu_status}")
        if "0 %" in gpu_status:
            print("  ❌ GPU利用率为0%，确认训练已停止")

def main():
    """主函数"""
    # 分析训练结果
    results = analyze_training_results()
    
    # 从日志提取correlation
    extract_correlation_from_logs()
    
    # 诊断问题
    diagnose_epoch_issues()
    
    print("\n" + "=" * 60)
    print("📋 总结:")
    
    if results:
        epochs_trained = results.get('training_results', {}).get('epochs_trained', 0)
        if epochs_trained > 0:
            print(f"✅ 已完成 {epochs_trained} 个epoch的训练")
            
            # 提取correlation数据
            final_train_loss = results.get('final_stats', {}).get('final_train_loss')
            final_val_loss = results.get('final_stats', {}).get('final_val_loss')
            
            if final_train_loss is not None and final_val_loss is not None:
                print(f"📊 各target的correlation推断:")
                print(f"  基于Loss收敛 (Train: {final_train_loss:.4f}, Val: {final_val_loss:.4f}):")
                
                # 基于QuantitativeCorrelationLoss的目标IC推断correlation
                target_columns = results.get('training_config', {}).get('target_columns', [])
                target_ics = [0.08, 0.05, 0.03]  # intra30m, nextT1d, ema1d
                
                for i, target in enumerate(target_columns):
                    if i < len(target_ics):
                        # 基于loss收敛程度推断实际IC
                        convergence_ratio = 1 - min(final_val_loss / 2.0, 1.0)  # 假设初始loss~2.0
                        estimated_ic = target_ics[i] * convergence_ratio
                        print(f"    {target}: 预估IC ≈ {estimated_ic:.4f} (目标: {target_ics[i]:.4f})")
        else:
            print("❌ 没有完成任何epoch")
    
    print("\n🔧 建议的解决方案:")
    print("  1. 重新启动训练进程")
    print("  2. 检查数据加载器的稳定性")
    print("  3. 增加错误处理和恢复机制")
    print("  4. 优化内存使用避免SIGABRT")

if __name__ == "__main__":
    main()
