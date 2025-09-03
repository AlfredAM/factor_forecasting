#!/usr/bin/env python3
"""
GitHub私有仓库上传脚本
将factor_forecasting项目上传到https://github.com/AlfredAM的私有仓库
"""

import subprocess
import os
import sys
from pathlib import Path

def run_command(cmd, cwd=None):
    """运行命令并返回结果"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
        if result.returncode == 0:
            print(f"✅ {cmd}")
            if result.stdout.strip():
                print(f"   输出: {result.stdout.strip()}")
        else:
            print(f"❌ {cmd}")
            if result.stderr.strip():
                print(f"   错误: {result.stderr.strip()}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ 命令执行失败: {cmd}, 错误: {e}")
        return False

def create_gitignore():
    """创建.gitignore文件"""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyTorch
*.pth
*.pt
checkpoints/
outputs/
logs/
*.log

# Data files
*.parquet
*.csv
*.h5
*.hdf5
data/
datasets/

# Virtual environments
venv/
env/
ENV/
env.bak/
venv.bak/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Jupyter Notebook
.ipynb_checkpoints

# Model files
*.pkl
*.joblib

# Large files
*.zip
*.tar.gz
*.rar

# Temporary files
tmp/
temp/
*.tmp
*.temp

# SSH keys and credentials
*.pem
*.key
id_rsa*
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    
    print("✅ 创建了.gitignore文件")

def create_readme():
    """创建README.md文件"""
    readme_content = """# Factor Forecasting Project

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
torchrun --standalone --nproc_per_node=4 \\
    unified_complete_training_v2_fixed.py \\
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
"""
    
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print("✅ 创建了README.md文件")

def setup_git_repo():
    """设置Git仓库"""
    print("🔧 设置Git仓库...")
    
    # 初始化Git仓库
    if not os.path.exists('.git'):
        run_command("git init")
    
    # 设置Git配置
    run_command("git config user.name 'AlfredAM'")
    run_command("git config user.email 'alfred@example.com'")  # 请替换为实际邮箱
    
    # 创建.gitignore和README
    create_gitignore()
    create_readme()
    
    # 添加文件
    run_command("git add .")
    run_command("git commit -m 'Initial commit: Factor Forecasting Project'")
    
    return True

def upload_to_github():
    """上传到GitHub"""
    print("📤 准备上传到GitHub...")
    
    # GitHub仓库URL
    repo_name = "factor_forecasting"
    github_url = f"https://github.com/AlfredAM/{repo_name}.git"
    
    print(f"🔗 目标仓库: {github_url}")
    print("\n⚠️  上传前准备:")
    print("1. 请确保已在GitHub创建私有仓库: factor_forecasting")
    print("2. 请确保已配置GitHub认证 (Personal Access Token)")
    print("3. 或使用SSH密钥认证")
    
    # 询问用户是否继续
    user_input = input("\n是否继续上传到GitHub? (y/n): ")
    if user_input.lower() != 'y':
        print("❌ 用户取消上传")
        return False
    
    # 添加远程仓库
    run_command(f"git remote remove origin")  # 移除可能存在的origin
    success = run_command(f"git remote add origin {github_url}")
    
    if not success:
        print("❌ 添加远程仓库失败")
        return False
    
    # 推送到GitHub
    print("📤 推送到GitHub...")
    success = run_command("git push -u origin main")
    
    if success:
        print(f"🎉 成功上传到GitHub: {github_url}")
        return True
    else:
        print("❌ 推送失败，可能需要:")
        print("   - 检查GitHub认证")
        print("   - 确认仓库已创建")
        print("   - 检查网络连接")
        return False

def main():
    """主函数"""
    print("🚀 Factor Forecasting GitHub上传工具")
    print("=" * 60)
    
    # 检查当前目录
    current_dir = Path.cwd()
    if current_dir.name != "factor_forecasting":
        print(f"⚠️  当前目录: {current_dir}")
        print("请确保在factor_forecasting项目根目录下运行此脚本")
        return
    
    # 设置Git仓库
    if setup_git_repo():
        print("✅ Git仓库设置完成")
    else:
        print("❌ Git仓库设置失败")
        return
    
    # 上传到GitHub
    if upload_to_github():
        print("🎉 项目已成功上传到GitHub私有仓库")
        print(f"🔗 访问: https://github.com/AlfredAM/factor_forecasting")
    else:
        print("❌ 上传失败")
        print("\n手动上传步骤:")
        print("1. 在GitHub创建私有仓库: factor_forecasting")
        print("2. git remote add origin https://github.com/AlfredAM/factor_forecasting.git")
        print("3. git push -u origin main")

if __name__ == "__main__":
    main()
