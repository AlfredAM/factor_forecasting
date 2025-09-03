#!/usr/bin/env python3
"""
自动化GitHub仓库创建和代码推送脚本
- 自动创建GitHub私有仓库
- 配置SSH密钥认证
- 推送代码到GitHub
"""

import subprocess
import os
import sys
import json
import requests
from pathlib import Path
import getpass

class GitHubAutoSetup:
    def __init__(self):
        self.repo_name = "factor_forecasting"
        self.github_username = "AlfredAM"
        self.ssh_key_path = Path.home() / ".ssh" / "id_rsa"
        self.ssh_pub_key_path = Path.home() / ".ssh" / "id_rsa.pub"
        
    def run_command(self, cmd, cwd=None, check=True):
        """运行命令并返回结果"""
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd, check=check)
            if result.returncode == 0:
                print(f"✅ {cmd}")
                if result.stdout.strip():
                    print(f"   输出: {result.stdout.strip()}")
                return result.stdout.strip()
            else:
                print(f"❌ {cmd}")
                if result.stderr.strip():
                    print(f"   错误: {result.stderr.strip()}")
                return None
        except subprocess.CalledProcessError as e:
            print(f"❌ 命令执行失败: {cmd}")
            print(f"   错误: {e.stderr}")
            return None
        except Exception as e:
            print(f"❌ 命令执行异常: {cmd}, 错误: {e}")
            return None

    def check_github_cli(self):
        """检查并安装GitHub CLI"""
        print("🔍 检查GitHub CLI...")
        result = self.run_command("gh --version", check=False)
        
        if result is None:
            print("📦 安装GitHub CLI...")
            # macOS使用Homebrew安装
            install_result = self.run_command("brew install gh", check=False)
            if install_result is None:
                print("❌ GitHub CLI安装失败，请手动安装:")
                print("   macOS: brew install gh")
                print("   或访问: https://cli.github.com/")
                return False
        
        return True

    def setup_ssh_keys(self):
        """设置SSH密钥"""
        print("🔑 设置SSH密钥...")
        
        # 检查SSH密钥是否存在
        if self.ssh_key_path.exists() and self.ssh_pub_key_path.exists():
            print(f"✅ SSH密钥已存在: {self.ssh_key_path}")
            return True
        
        # 生成SSH密钥
        print("🔧 生成新的SSH密钥...")
        email = input("请输入您的GitHub邮箱地址: ")
        
        ssh_gen_cmd = f'ssh-keygen -t rsa -b 4096 -C "{email}" -f {self.ssh_key_path} -N ""'
        result = self.run_command(ssh_gen_cmd)
        
        if result is None:
            print("❌ SSH密钥生成失败")
            return False
        
        # 添加SSH密钥到ssh-agent
        print("🔧 添加SSH密钥到ssh-agent...")
        self.run_command("eval $(ssh-agent -s)")
        self.run_command(f"ssh-add {self.ssh_key_path}")
        
        return True

    def add_ssh_key_to_github(self):
        """将SSH公钥添加到GitHub"""
        print("🔑 添加SSH公钥到GitHub...")
        
        if not self.ssh_pub_key_path.exists():
            print("❌ SSH公钥文件不存在")
            return False
        
        # 读取公钥
        with open(self.ssh_pub_key_path, 'r') as f:
            public_key = f.read().strip()
        
        print("📋 SSH公钥内容:")
        print(public_key)
        print("\n请执行以下步骤:")
        print("1. 复制上面的SSH公钥")
        print("2. 访问 https://github.com/settings/ssh/new")
        print("3. 粘贴公钥并保存")
        
        input("完成后按回车键继续...")
        
        # 测试SSH连接
        print("🔍 测试SSH连接...")
        test_result = self.run_command("ssh -T git@github.com", check=False)
        
        return True

    def authenticate_github_cli(self):
        """认证GitHub CLI"""
        print("🔐 认证GitHub CLI...")
        
        # 检查是否已经认证
        auth_status = self.run_command("gh auth status", check=False)
        if auth_status and "Logged in to github.com" in auth_status:
            print("✅ GitHub CLI已认证")
            return True
        
        # 进行认证
        print("🔐 开始GitHub CLI认证...")
        auth_result = self.run_command("gh auth login", check=False)
        
        if auth_result is None:
            print("❌ GitHub CLI认证失败")
            return False
        
        return True

    def create_github_repo(self):
        """创建GitHub私有仓库"""
        print("📦 创建GitHub私有仓库...")
        
        # 检查仓库是否已存在
        check_cmd = f"gh repo view {self.github_username}/{self.repo_name}"
        if self.run_command(check_cmd, check=False):
            print(f"✅ 仓库 {self.repo_name} 已存在")
            return True
        
        # 创建私有仓库
        create_cmd = f'gh repo create {self.repo_name} --private --description "Advanced Factor Forecasting System with 4GPU Distributed Training"'
        result = self.run_command(create_cmd)
        
        if result is None:
            print("❌ 仓库创建失败")
            return False
        
        print(f"✅ 私有仓库创建成功: https://github.com/{self.github_username}/{self.repo_name}")
        return True

    def setup_git_repo(self):
        """设置本地Git仓库"""
        print("🔧 设置本地Git仓库...")
        
        # 初始化Git仓库（如果尚未初始化）
        if not Path(".git").exists():
            self.run_command("git init")
        
        # 设置Git配置
        self.run_command(f"git config user.name '{self.github_username}'")
        
        email = input("请输入您的Git邮箱地址: ")
        self.run_command(f"git config user.email '{email}'")
        
        # 创建.gitignore（如果不存在）
        self.create_gitignore()
        
        # 创建README.md（如果不存在）
        self.create_readme()
        
        return True

    def create_gitignore(self):
        """创建.gitignore文件"""
        if Path(".gitignore").exists():
            print("✅ .gitignore文件已存在")
            return
            
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

    def create_readme(self):
        """创建README.md文件"""
        readme_content = """# Factor Forecasting Project

## 🚀 项目概述
先进的因子预测系统，使用深度学习技术进行金融因子预测，支持4GPU分布式训练。

## 📊 实时训练状态

### 当前训练进度
- **训练状态**: Epoch 0, 217 iterations
- **训练速度**: 5.04秒/iteration  
- **GPU利用率**: 4GPU分布式训练中
- **预计完成**: ~24分钟后完成当前epoch

### GPU使用情况
```
GPU 0: 22.5GB/23GB显存 (100%利用率)
GPU 1: 22.6GB/23GB显存 (0%利用率) 
GPU 2: 22.3GB/23GB显存 (100%利用率)
GPU 3: 3.7GB/23GB显存 (0%利用率)
总显存利用率: 77.2%
```

### 系统资源
- **CPU使用率**: 3.1% (128核心)
- **内存使用**: 23GB/739GB (3.1%)
- **存储**: 589GB/10PB (1%)

## 🧠 核心特性
- **4GPU分布式训练**: 充分利用多GPU资源
- **TCN + Attention架构**: 先进的时序建模
- **实时监控**: 训练进度和相关性监控
- **自适应内存管理**: 智能内存优化
- **滚动训练**: 支持时间序列滚动预测

## 🏗️ 模型架构
- **AdvancedFactorForecastingTCNAttentionModel**: 结合TCN和注意力机制
- **多目标预测**: intra30m, nextT1d, ema1d
- **量化损失函数**: 专门的金融预测损失

## 💻 硬件配置
- **GPU**: 4x NVIDIA A10 (22GB显存)
- **内存**: 739GB RAM
- **CPU**: 128核心
- **存储**: 高速SSD存储

## 🚀 快速开始

### 环境配置
```bash
# 激活虚拟环境
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

## 📁 项目结构
```
factor_forecasting/
├── src/
│   ├── models/           # 模型定义
│   ├── data_processing/  # 数据处理和加载
│   ├── training/         # 训练逻辑
│   └── monitoring/       # 监控和报告
├── configs/              # 配置文件
├── outputs/              # 输出结果
└── requirements.txt      # 依赖列表
```

## 📈 性能指标
- **GPU利用率**: >90% (4GPU并行)
- **训练速度**: ~5s/iteration
- **内存效率**: 自适应批次大小
- **相关性报告**: 每2小时自动生成

## 🔧 技术栈
- **PyTorch**: 深度学习框架
- **CUDA**: GPU计算
- **Distributed Training**: 多GPU并行
- **Mixed Precision**: 混合精度训练
- **NCCL**: GPU通信后端

## 👨‍💻 作者
AlfredAM - https://github.com/AlfredAM

## 📄 许可证
Private Repository - All Rights Reserved

---
*最后更新: 2025-09-03 17:18*
"""
        
        with open('README.md', 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print("✅ 创建了README.md文件")

    def commit_and_push(self):
        """提交并推送代码"""
        print("📤 提交并推送代码...")
        
        # 添加所有文件
        self.run_command("git add .")
        
        # 提交
        commit_msg = "feat: Advanced Factor Forecasting System with 4GPU Distributed Training"
        self.run_command(f'git commit -m "{commit_msg}"')
        
        # 设置远程仓库
        remote_url = f"git@github.com:{self.github_username}/{self.repo_name}.git"
        self.run_command("git remote remove origin", check=False)  # 移除可能存在的origin
        self.run_command(f"git remote add origin {remote_url}")
        
        # 推送到main分支
        self.run_command("git branch -M main")
        push_result = self.run_command("git push -u origin main")
        
        if push_result is None:
            print("❌ 推送失败，尝试使用HTTPS...")
            # 尝试HTTPS推送
            https_url = f"https://github.com/{self.github_username}/{self.repo_name}.git"
            self.run_command(f"git remote set-url origin {https_url}")
            push_result = self.run_command("git push -u origin main")
        
        return push_result is not None

    def run_full_setup(self):
        """运行完整的设置流程"""
        print("🚀 Factor Forecasting GitHub自动设置")
        print("=" * 60)
        
        # 1. 检查GitHub CLI
        if not self.check_github_cli():
            return False
        
        # 2. 认证GitHub CLI
        if not self.authenticate_github_cli():
            return False
        
        # 3. 设置SSH密钥
        if not self.setup_ssh_keys():
            return False
        
        # 4. 添加SSH密钥到GitHub
        if not self.add_ssh_key_to_github():
            return False
        
        # 5. 创建GitHub仓库
        if not self.create_github_repo():
            return False
        
        # 6. 设置本地Git仓库
        if not self.setup_git_repo():
            return False
        
        # 7. 提交并推送代码
        if not self.commit_and_push():
            print("❌ 代码推送失败")
            return False
        
        print("\n🎉 GitHub仓库设置完成!")
        print(f"🔗 仓库地址: https://github.com/{self.github_username}/{self.repo_name}")
        print("✅ 代码已成功推送到私有仓库")
        
        return True

def main():
    """主函数"""
    setup = GitHubAutoSetup()
    
    try:
        success = setup.run_full_setup()
        if success:
            print("\n🎊 所有设置完成！")
        else:
            print("\n❌ 设置过程中出现错误")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n👋 用户取消操作")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 意外错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
