# Factor Forecasting Project

## 项目概述
先进的因子预测系统，使用深度学习技术进行金融因子预测。

## 核心特性
- 🚀 **4GPU分布式训练**: 充分利用多GPU资源
- 🧠 **TCN + Attention架构**: 先进的时序建模
- 📊 **实时监控**: 训练进度和相关性监控
- 🔧 **自适应内存管理**: 智能内存优化
- 📈 **滚动训练**: 支持时间序列滚动预测

## 模型架构
- **AdvancedFactorForecastingTCNAttentionModel**: 结合TCN和注意力机制
- **多目标预测**: intra30m, nextT1d, ema1d
- **量化损失函数**: 专门的金融预测损失

## 硬件要求
- **GPU**: 4x NVIDIA A10 (22GB显存)
- **内存**: 739GB RAM
- **CPU**: 128核心
- **存储**: 高速SSD存储

## 安装和使用

### 环境配置
```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 训练启动
```bash
# 4GPU分布式训练
torchrun --standalone --nproc_per_node=4 \
    unified_complete_training_v2_fixed.py \
    --config optimal_4gpu_config.yaml
```

### 监控系统
```bash
# 启动持续监控
python continuous_training_monitor.py
```

## 配置文件
- `optimal_4gpu_config.yaml`: 4GPU高性能配置
- `server_optimized_config.yaml`: 服务器优化配置

## 核心模块
- `src/models/`: 模型定义
- `src/data_processing/`: 数据处理和加载
- `src/training/`: 训练逻辑
- `src/monitoring/`: 监控和报告

## 性能指标
- **GPU利用率**: >90% (4GPU并行)
- **训练速度**: ~6s/iteration
- **内存效率**: 自适应批次大小
- **相关性报告**: 每2小时自动生成

## 技术栈
- **PyTorch**: 深度学习框架
- **CUDA**: GPU计算
- **Distributed Training**: 多GPU并行
- **Mixed Precision**: 混合精度训练
- **NCCL**: GPU通信后端

## 作者
AlfredAM - https://github.com/AlfredAM

## 许可证
Private Repository - All Rights Reserved
