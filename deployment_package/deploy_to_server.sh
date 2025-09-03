#!/bin/bash

# 自动化部署脚本 - 部署到新服务器
# 服务器: 8.216.35.79
# 目标: 彻底解决所有问题，实现稳定高效训练

SERVER="8.216.35.79"
USER="ecs-user"
PASSWORD="Abab1234"
REMOTE_DIR="/nas/factor_forecasting"

echo "=== 自动化部署到新服务器 ==="
echo "服务器: $SERVER"
echo "目标目录: $REMOTE_DIR"
echo

# 函数：安全连接服务器
safe_ssh() {
    sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PreferredAuthentications=password -o PubkeyAuthentication=no "$USER@$SERVER" "$1"
}

safe_scp() {
    sshpass -p "$PASSWORD" scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PreferredAuthentications=password -o PubkeyAuthentication=no "$1" "$USER@$SERVER:$2"
}

# 1. 测试服务器连接
echo "1. 测试服务器连接..."
if ! ping -c 1 $SERVER > /dev/null 2>&1; then
    echo "❌ 服务器网络不通"
    exit 1
fi

# 等待SSH服务可用
echo "2. 等待SSH服务..."
for i in {1..10}; do
    if safe_ssh "echo 'SSH连接成功'" > /dev/null 2>&1; then
        echo "✅ SSH连接正常"
        break
    else
        echo "等待SSH服务... ($i/10)"
        sleep 3
    fi
    if [ $i -eq 10 ]; then
        echo "❌ SSH连接失败"
        exit 1
    fi
done

# 3. 检查服务器硬件配置
echo "3. 检查服务器硬件配置..."
safe_ssh "cd $REMOTE_DIR && echo '=== 服务器硬件配置 ===' && echo 'CPU信息:' && lscpu | grep -E '(CPU\\(s\\)|Thread|Core|Socket)' && echo && echo '内存信息:' && free -h && echo && echo 'GPU信息:' && nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader && echo && echo '磁盘空间:' && df -h /nas"

# 4. 清理旧进程
echo "4. 清理所有旧训练进程..."
safe_ssh "cd $REMOTE_DIR && echo '=== 清理旧进程 ===' && pkill -f 'unified_complete' || echo '无unified_complete进程' && pkill -f 'torchrun.*unified' || echo '无torchrun进程' && pkill -f 'python.*training' || echo '无training进程' && nvidia-smi | grep python | awk '{print \$5}' | xargs -r kill -9 || echo '无GPU进程需清理' && sleep 5 && echo '✅ 进程清理完成'"

# 5. 激活虚拟环境并检查
echo "5. 激活虚拟环境并检查..."
safe_ssh "cd $REMOTE_DIR && source venv/bin/activate && python --version && echo '✅ 虚拟环境已激活'"

# 6. 上传修复后的代码
echo "6. 上传修复后的核心代码..."
safe_scp "src/unified_complete_training_v2.py" "$REMOTE_DIR/src/"
safe_scp "src/data_processing/adaptive_memory_manager.py" "$REMOTE_DIR/src/data_processing/"
safe_scp "src/models/advanced_tcn_attention.py" "$REMOTE_DIR/src/models/"
safe_scp "src/training/quantitative_loss.py" "$REMOTE_DIR/src/training/"
echo "✅ 核心代码已上传"

# 7. 安装/更新软件包
echo "7. 安装必要的软件包..."
safe_ssh "cd $REMOTE_DIR && source venv/bin/activate && pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch torchvision torchaudio numpy pandas scikit-learn pyyaml tqdm psutil pyarrow --upgrade"

# 8. 检查数据文件
echo "8. 检查数据文件状态..."
safe_ssh "cd $REMOTE_DIR && echo '=== 数据文件检查 ===' && ls -la /nas/feature_v2_10s/*.parquet 2>/dev/null | head -5 && echo '...' && echo '数据文件总数:' && ls /nas/feature_v2_10s/*.parquet 2>/dev/null | wc -l"

echo "✅ 部署准备完成，准备启动训练..."

