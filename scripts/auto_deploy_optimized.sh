#!/bin/bash
"""
自动化deploy和testingoptimization脚本
功能：
1. 上传所有optimizationfile到server
2. 自动testingoptimizationversion
3. 性能基准testing
4. 平滑切换training
"""

set -e  # 遇到error立即exit

# configurationvariable
REMOTE_HOST="8.155.163.64"
REMOTE_USER="ecs-user"
REMOTE_PASSWORD="Abab1234"
REMOTE_DIR="/nas/factor_forecasting"
LOCAL_DIR="/Users/scratch/Documents/My Code/Projects/factor_forecasting"

echo "=== 自动化optimizationdeploystart ==="
echo "time: $(date)"

# function：execute远程命令
remote_exec() {
    sshpass -p "$REMOTE_PASSWORD" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$REMOTE_USER@$REMOTE_HOST" "$1"
}

# function：上传file
upload_file() {
    local src="$1"
    local dst="$2"
    echo "上传: $src -> $dst"
    sshpass -p "$REMOTE_PASSWORD" scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$src" "$REMOTE_USER@$REMOTE_HOST:$dst"
}

# 1. 备份当前trainingstatus
echo "1. 备份当前trainingstatus..."
remote_exec "cd $REMOTE_DIR && mkdir -p backup/$(date +%Y%m%d_%H%M%S) && cp -r stream_tcn_attention_* backup/$(date +%Y%m%d_%H%M%S)/ 2>/dev/null || true"

# 2. 上传optimization脚本
echo "2. 上传optimization脚本..."
upload_file "$LOCAL_DIR/scripts/optimized_streaming_loader.py" "$REMOTE_DIR/scripts/"
upload_file "$LOCAL_DIR/scripts/optimized_tcn_attention_train.py" "$REMOTE_DIR/scripts/"
upload_file "$LOCAL_DIR/scripts/advanced_ddp_train.py" "$REMOTE_DIR/scripts/"
upload_file "$LOCAL_DIR/scripts/gpu_monitor.py" "$REMOTE_DIR/scripts/"

# 3. create性能testing脚本
echo "3. create性能testing脚本..."
cat > /tmp/performance_test.py << 'EOF'
#!/usr/bin/env python3
import torch
import time
import json
import sys
import os
from pathlib import Path

# add项目path
sys.path.insert(0, '/nas/factor_forecasting')
from scripts.optimized_streaming_loader import OptimizedStreamingDataLoader
from src.models.models import create_model

def test_data_loader_performance():
    """testingdata loading器性能"""
    print("testingdata loading器性能...")
    
    # 原始loadertesting
    sys.path.append('/nas/factor_forecasting/scripts')
    from advanced_tcn_train_improved import StreamingDataLoader as OriginalLoader
    
    # testing原始loader
    start_time = time.time()
    original_loader = OriginalLoader(
        data_path="/nas/feature_v2_10s",
        batch_size=256,
        sequence_length=20,
        max_files=5,
        logger=None
    )
    
    batches_original = 0
    for _ in range(10):
        batch = original_loader.get_next_batch()
        if batch is None:
            break
        batches_original += 1
    original_time = time.time() - start_time
    
    # testingoptimizationloader
    start_time = time.time()
    optimized_loader = OptimizedStreamingDataLoader(
        data_path="/nas/feature_v2_10s",
        batch_size=256,
        sequence_length=20,
        max_files=5,
        prefetch_workers=4,
        prefetch_queue_size=8,
        logger=None
    )
    
    batches_optimized = 0
    for _ in range(10):
        batch = optimized_loader.get_next_batch()
        if batch is None:
            break
        batches_optimized += 1
    optimized_time = time.time() - start_time
    optimized_loader.cleanup()
    
    improvement = (original_time - optimized_time) / original_time * 100 if original_time > 0 else 0
    
    results = {
        'original_time': original_time,
        'optimized_time': optimized_time,
        'improvement_percent': improvement,
        'batches_original': batches_original,
        'batches_optimized': batches_optimized
    }
    
    print(f"原始loader: {original_time:.2f}s ({batches_original} batches)")
    print(f"optimizationloader: {optimized_time:.2f}s ({batches_optimized} batches)")
    print(f"性能提升: {improvement:.1f}%")
    
    return results

def test_model_performance():
    """testingmodel性能"""
    print("testingmodel性能...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = {
        'model_type': 'tcn_attention',
        'num_factors': 100,
        'num_stocks': 5000,
        'd_model': 256,
        'num_heads': 8,
        'num_layers': 6,
        'sequence_length': 20,
        'target_columns': ['nextT1d']
    }
    
    model = create_model(config)
    model.to(device)
    model.eval()
    
    # 性能testing
    batch_sizes = [64, 128, 256, 512]
    results = {}
    
    for batch_size in batch_sizes:
        try:
            # createtestingdata
            features = torch.randn(batch_size, 20, 100, device=device)
            stock_ids = torch.randint(0, 5000, (batch_size, 20), device=device)
            
            # 预热
            for _ in range(5):
                with torch.no_grad():
                    _ = model(features, stock_ids)
            
            # 性能testing
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            for _ in range(10):
                with torch.no_grad():
                    _ = model(features, stock_ids)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 10
            throughput = batch_size / avg_time
            
            results[batch_size] = {
                'avg_time': avg_time,
                'throughput': throughput
            }
            
            print(f"Batch {batch_size}: {avg_time:.4f}s, {throughput:.1f} samples/s")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Batch {batch_size}: OOM")
                results[batch_size] = {'error': 'OOM'}
                break
            else:
                raise
    
    return results

def main():
    print("=== 性能基准testing ===")
    
    # testingdata loading器
    loader_results = test_data_loader_performance()
    
    # testingmodel
    model_results = test_model_performance()
    
    # saveresult
    benchmark_results = {
        'timestamp': time.time(),
        'data_loader': loader_results,
        'model': model_results
    }
    
    with open('/nas/factor_forecasting/benchmark_results.json', 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    
    print("基准testing complete，resultsave到 benchmark_results.json")
    
    # return是否应该continuedeploy
    if loader_results.get('improvement_percent', 0) > 10:
        print(" data loading器性能提升显著，suggestiondeploy")
        return True
    else:
        print("  data loading器性能提升不明显")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
EOF

upload_file "/tmp/performance_test.py" "$REMOTE_DIR/scripts/"

# 4. run性能testing
echo "4. run性能基准testing..."
if remote_exec "cd $REMOTE_DIR && source venv/bin/activate && python scripts/performance_test.py"; then
    echo " 性能testing通过"
else
    echo " 性能testingfailed，stopdeploy"
    exit 1
fi

# 5. create平滑切换脚本
echo "5. create平滑切换脚本..."
cat > /tmp/smooth_switch.sh << 'EOF'
#!/bin/bash
set -e

CURRENT_PID=$(ps aux | grep streaming_tcn_attention_train | grep -v grep | awk '{print $2}' | head -1)

if [ ! -z "$CURRENT_PID" ]; then
    echo "发现当前training进程: $CURRENT_PID"
    
    # 等待当前epochcomplete（最多等待10minutes）
    echo "等待当前epochcomplete..."
    for i in {1..60}; do
        sleep 10
        if ! ps -p $CURRENT_PID > /dev/null; then
            echo "training进程已自然end"
            break
        fi
        echo "等待中... ($i/60)"
    done
    
    # 如果还在run，优雅close
    if ps -p $CURRENT_PID > /dev/null; then
        echo "发送SIGTERM信号..."
        kill -TERM $CURRENT_PID
        sleep 30
        
        # 如果还在run，强制close
        if ps -p $CURRENT_PID > /dev/null; then
            echo "强制close进程..."
            kill -KILL $CURRENT_PID
        fi
    fi
fi

echo "startoptimization版training..."
cd /nas/factor_forecasting
source venv/bin/activate

# startoptimization版training
nohup python scripts/optimized_tcn_attention_train.py \
    --data-path /nas/feature_v2_10s \
    --sequence-length 20 \
    --batch-size 0 \
    --epochs 99999 \
    --auto-optimize \
    --prefetch-workers 6 \
    --prefetch-queue-size 12 \
    --resume auto \
    > optimized_training.log 2>&1 &

NEW_PID=$!
echo "新training进程start: $NEW_PID"
echo $NEW_PID > optimized_training.pid

# validatestartsuccessful
sleep 10
if ps -p $NEW_PID > /dev/null; then
    echo " optimization版trainingstartsuccessful"
    echo "PID: $NEW_PID"
    echo "log: optimized_training.log"
else
    echo " optimization版trainingstartfailed"
    exit 1
fi
EOF

upload_file "/tmp/smooth_switch.sh" "$REMOTE_DIR/"
remote_exec "chmod +x $REMOTE_DIR/smooth_switch.sh"

# 6. execute平滑切换
echo "6. execute平滑切换到optimization版training..."
if remote_exec "$REMOTE_DIR/smooth_switch.sh"; then
    echo " 平滑切换successful"
else
    echo " 平滑切换failed"
    exit 1
fi

# 7. 监控新trainingstatus
echo "7. 监控新trainingstatus..."
sleep 30

remote_exec "cd $REMOTE_DIR && echo '=== 新trainingstatus ===' && ps aux | grep optimized_tcn_attention_train | grep -v grep && echo '=== GPUstatus ===' && nvidia-smi | head -20 && echo '=== 最新log ===' && tail -n 10 optimized_training.log"

# 8. start性能监控
echo "8. start持续性能监控..."
remote_exec "cd $REMOTE_DIR && nohup python scripts/gpu_monitor.py --interval 30 --duration 7200 --training-log optimized_training.log > gpu_monitor.log 2>&1 &"

# 9. createDDP准备脚本
echo "9. 准备DDPtraining脚本..."
cat > /tmp/start_ddp.sh << 'EOF'
#!/bin/bash
set -e

echo "准备start高级DDPtraining..."
cd /nas/factor_forecasting
source venv/bin/activate

# check当前trainingstatus
CURRENT_PID=$(cat optimized_training.pid 2>/dev/null || echo "")
if [ ! -z "$CURRENT_PID" ] && ps -p $CURRENT_PID > /dev/null; then
    echo "stop当前optimizationtraining..."
    kill -TERM $CURRENT_PID
    sleep 30
fi

# startDDPtraining
echo "start高级DDPtraining..."
nohup python scripts/advanced_ddp_train.py \
    --data-path /nas/feature_v2_10s \
    --sequence-length 20 \
    --batch-size 64 \
    --epochs 99999 \
    --learning-rate 1e-4 \
    --prefetch-workers 6 \
    --prefetch-queue-size 12 \
    > ddp_training.log 2>&1 &

DDP_PID=$!
echo "DDPtrainingstart: $DDP_PID"
echo $DDP_PID > ddp_training.pid

sleep 10
if ps -p $DDP_PID > /dev/null; then
    echo " DDPtrainingstartsuccessful"
else
    echo " DDPtrainingstartfailed"
    cat ddp_training.log | tail -20
fi
EOF

upload_file "/tmp/start_ddp.sh" "$REMOTE_DIR/"
remote_exec "chmod +x $REMOTE_DIR/start_ddp.sh"

echo "=== 自动化deploycomplete ==="
echo "当前status："
remote_exec "cd $REMOTE_DIR && ps aux | grep -E '(optimized|ddp).*train' | grep -v grep"

echo ""
echo "可用命令："
echo "- viewoptimizationtraininglog: tail -f $REMOTE_DIR/optimized_training.log"
echo "- viewGPU监控: tail -f $REMOTE_DIR/gpu_monitor.log"
echo "- startDDPtraining: $REMOTE_DIR/start_ddp.sh"
echo "- view基准testingresult: cat $REMOTE_DIR/benchmark_results.json"

echo ""
echo " 自动化optimizationdeploysuccessfulcomplete！"
