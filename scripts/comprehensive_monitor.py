#!/usr/bin/env python3
"""
trainingmonitorsystem - monitorservicetrainingstate
"""
import subprocess
import time
import datetime
import re
import json
from typing import Dict, List, Optional, Tuple

def run_ssh_command(cmd: str, timeout: int = 30) -> str:
    """executeSSHreturnoutput"""
    ssh_cmd = [
        'sshpass', '-p', 'Abab1234',
        'ssh', '-o', 'StrictHostKeyChecking=no',
        '-o', 'UserKnownHostsFile=/dev/null', 
        '-o', 'PreferredAuthentications=password',
        '-o', 'PubkeyAuthentication=no',
        '-o', 'ConnectTimeout=10',
        'ecs-user@47.120.46.105',
        cmd
    ]
    
    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"Error: {result.stderr.strip()}"
    except Exception as e:
        return f"SSH Error: {e}"

def get_gpu_status() -> Dict:
    """getGPUstate"""
    cmd = 'nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader'
    result = run_ssh_command(cmd)
    
    gpu_info = {}
    if "Error" not in result:
        for line in result.split('\n'):
            if line.strip():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 6:
                    gpu_id = parts[0]
                    gpu_info[f"GPU{gpu_id}"] = {
                        'name': parts[1],
                        'utilization': parts[2],
                        'memory_used': parts[3],
                        'memory_total': parts[4],
                        'temperature': parts[5],
                        'power': parts[6] if len(parts) > 6 else 'N/A'
                    }
    return gpu_info

def get_training_processes() -> Dict:
    """gettrainingprocessinfo"""
    cmd = 'ps -o pid,etime,pcpu,pmem,rss,args | grep -E "(torchrun|unified)" | grep -v grep'
    result = run_ssh_command(cmd)
    
    processes = {}
    if "Error" not in result:
        for line in result.split('\n'):
            if 'unified_complete' in line:
                parts = line.split()
                if len(parts) >= 6:
                    processes[parts[0]] = {
                        'pid': parts[0],
                        'runtime': parts[1],
                        'cpu_percent': parts[2],
                        'memory_percent': parts[3],
                        'memory_kb': parts[4]
                    }
    return processes

def parse_training_progress(logs: str) -> Dict:
    """trainingprogress"""
    progress = {
        'current_epoch': None,
        'current_iteration': None,
        'current_loss': None,
        'avg_loss': None,
        'iteration_time': None,
        'total_iterations_in_epoch': None
    }
    
    # findtrainingprogress
    epoch_pattern = r'Epoch (\d+) Training: (\d+)it \[([^\]]+)\]\s*.*?Loss=([0-9.]+).*?Avg=([0-9.]+)'
    matches = re.findall(epoch_pattern, logs)
    
    if matches:
        latest = matches[-1]
        progress['current_epoch'] = int(latest[0])
        progress['current_iteration'] = int(latest[1])
        progress['current_loss'] = float(latest[3])
        progress['avg_loss'] = float(latest[4])
        
        # timeinfo
        time_info = latest[2]
        if 's/it' in time_info:
            time_match = re.search(r'([0-9.]+)s/it', time_info)
            if time_match:
                progress['iteration_time'] = float(time_match.group(1))
    
    return progress

def estimate_epoch_completion(progress: Dict) -> Dict:
    """epochcompletetime"""
    estimation = {}
    
    if all(k in progress and progress[k] is not None for k in ['current_iteration', 'iteration_time']):
        # progress
        current_iter = progress['current_iteration']
        iter_time = progress['iteration_time']
        
        # epoch1000-2000iterationdatasize
        estimated_total_iters = 1500  # dynamicadjustment
        
        if current_iter > 0:
            remaining_iters = max(0, estimated_total_iters - current_iter)
            estimated_remaining_time = remaining_iters * iter_time
            
            estimation.update({
                'estimated_total_iterations': estimated_total_iters,
                'remaining_iterations': remaining_iters,
                'estimated_remaining_seconds': estimated_remaining_time,
                'estimated_completion_time': datetime.datetime.now() + datetime.timedelta(seconds=estimated_remaining_time),
                'progress_percentage': (current_iter / estimated_total_iters) * 100
            })
    
    return estimation

def get_system_metrics() -> Dict:
    """getsystemmetrics"""
    cmd = 'free -m | grep Mem && df -h /nas | tail -1'
    result = run_ssh_command(cmd)
    
    metrics = {}
    if "Error" not in result:
        lines = result.split('\n')
        if len(lines) >= 2:
            # memoryinfo
            mem_line = lines[0].split()
            if len(mem_line) >= 3:
                metrics['memory'] = {
                    'total_mb': mem_line[1],
                    'used_mb': mem_line[2],
                    'available_mb': mem_line[6] if len(mem_line) > 6 else mem_line[3]
                }
            
            # info
            disk_line = lines[1].split()
            if len(disk_line) >= 4:
                metrics['disk'] = {
                    'total': disk_line[1],
                    'used': disk_line[2],
                    'available': disk_line[3],
                    'use_percent': disk_line[4]
                }
    
    return metrics

def get_latest_training_logs(lines: int = 30) -> str:
    """gettraininglog"""
    cmd = f'cd /nas/factor_forecasting && L=$(ls -t logs/manual_ddp_run_*.log | head -1) && tail -n {lines} "$L"'
    return run_ssh_command(cmd)

def check_epoch_completion(logs: str) -> Optional[Dict]:
    """checkepochcomplete"""
    completion_patterns = [
        r'Epoch (\d+).*completed.*time.*?([0-9.]+).*?(seconds|minutes|s|m)',
        r'Epoch (\d+).*finished.*time.*?([0-9.]+).*?(seconds|minutes|s|m)',
        r'Epoch (\d+).*took.*?([0-9.]+).*?(seconds|minutes|s|m)',
    ]
    
    for pattern in completion_patterns:
        matches = re.findall(pattern, logs, re.IGNORECASE)
        if matches:
            latest = matches[-1]
            return {
                'epoch': int(latest[0]),
                'time': float(latest[1]),
                'unit': latest[2]
            }
    
    return None

def monitor_training():
    """monitorfunction"""
    print(" startservicetrainingmonitorsystem")
    print("service: 47.120.46.105")
    print("=" * 80)
    
    start_monitoring_time = time.time()
    last_progress_check = 0
    last_full_report = 0
    epoch_start_time = None
    last_iteration = 0
    
    while True:
        try:
            current_time = time.time()
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 30secondschecktimesstate
            if current_time - last_progress_check >= 30:
                print(f"\n [{timestamp}] statecheck")
                print("-" * 50)
                
                # GPUstate
                gpu_status = get_gpu_status()
                if gpu_status:
                    for gpu_id, info in gpu_status.items():
                        print(f" {gpu_id}: {info['utilization']} , {info['memory_used']}/{info['memory_total']} memory, {info['temperature']}°C")
                
                # trainingprocess
                processes = get_training_processes()
                if processes:
                    print(f" trainingprocess: {len(processes)}")
                    for pid, proc in list(processes.items())[:2]:
                        print(f"  PID {pid}: runtime {proc['runtime']}, CPU {proc['cpu_percent']}%")
                
                # trainingprogress
                logs = get_latest_training_logs(50)
                if logs and "Error" not in logs:
                    progress = parse_training_progress(logs)
                    if progress['current_epoch'] is not None:
                        print(f" Epoch {progress['current_epoch']}, Iter {progress['current_iteration']}, Loss {progress['current_loss']:.4f}, AvgLoss {progress['avg_loss']:.4f}")
                        
                        if progress['iteration_time']:
                            print(f" iteration: {progress['iteration_time']:.2f}s")
                            
                            # detectepochbegin
                            if progress['current_iteration'] < last_iteration:
                                epoch_start_time = time.time()
                                print(f" epochbegin")
                            last_iteration = progress['current_iteration']
                            
                            # completetime
                            estimation = estimate_epoch_completion(progress)
                            if estimation:
                                print(f" progress: {estimation['progress_percentage']:.1f}%, : {estimation['estimated_remaining_seconds']/60:.1f}minutes")
                
                # checkepochcomplete
                epoch_completion = check_epoch_completion(logs)
                if epoch_completion:
                    print(f"\n Epoch {epoch_completion['epoch']} complete")
                    print(f"⏰ : {epoch_completion['time']} {epoch_completion['unit']}")
                
                last_progress_check = current_time
            
            # 5minutesgeneratereport
            if current_time - last_full_report >= 300:
                print(f"\n [{timestamp}] systemreport")
                print("=" * 60)
                
                # systemmetrics
                system_metrics = get_system_metrics()
                if system_metrics:
                    if 'memory' in system_metrics:
                        mem = system_metrics['memory']
                        print(f" memory: {mem['used_mb']}MB / {mem['total_mb']}MB ")
                    if 'disk' in system_metrics:
                        disk = system_metrics['disk']
                        print(f" : {disk['used']} / {disk['total']}  ({disk['use_percent']})")
                
                # trainingstatistics
                total_monitoring_time = current_time - start_monitoring_time
                hours = int(total_monitoring_time // 3600)
                minutes = int((total_monitoring_time % 3600) // 60)
                print(f" monitortime: {hours}hours{minutes}minutes")
                
                last_full_report = current_time
            
            # 15secondstimescheck
            time.sleep(15)
            
        except KeyboardInterrupt:
            print("\n\n monitorsystemstop")
            break
        except Exception as e:
            print(f"\n monitorerror: {e}")
            print("⏰ 10secondsretry...")
            time.sleep(10)

if __name__ == "__main__":
    monitor_training()
