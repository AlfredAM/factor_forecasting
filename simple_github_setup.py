#!/usr/bin/env python3
"""
简化的GitHub仓库设置脚本
不依赖GitHub CLI，使用Git和SSH直接操作
"""

import subprocess
import os
import sys
from pathlib import Path

class SimpleGitHubSetup:
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
            if hasattr(e, 'stderr') and e.stderr:
                print(f"   错误: {e.stderr}")
            return None
        except Exception as e:
            print(f"❌ 命令执行异常: {cmd}, 错误: {e}")
            return None

    def check_ssh_keys(self):
        """检查SSH密钥"""
        print("🔑 检查SSH密钥...")
        
        if self.ssh_key_path.exists() and self.ssh_pub_key_path.exists():
            print(f"✅ SSH密钥已存在: {self.ssh_key_path}")
            
            # 显示公钥
            with open(self.ssh_pub_key_path, 'r') as f:
                public_key = f.read().strip()
            print("\n📋 您的SSH公钥:")
            print(public_key)
            
            return True
        else:
            print("🔧 生成SSH密钥...")
            return self.generate_ssh_keys()

    def generate_ssh_keys(self):
        """生成SSH密钥"""
        email = input("请输入您的GitHub邮箱地址: ")
        
        # 创建.ssh目录
        ssh_dir = Path.home() / ".ssh"
        ssh_dir.mkdir(exist_ok=True, mode=0o700)
        
        # 生成SSH密钥
        ssh_gen_cmd = f'ssh-keygen -t rsa -b 4096 -C "{email}" -f {self.ssh_key_path} -N ""'
        result = self.run_command(ssh_gen_cmd)
        
        if result is None:
            print("❌ SSH密钥生成失败")
            return False
        
        # 设置权限
        self.ssh_key_path.chmod(0o600)
        self.ssh_pub_key_path.chmod(0o644)
        
        # 添加到ssh-agent
        print("🔧 添加SSH密钥到ssh-agent...")
        self.run_command("ssh-add -K ~/.ssh/id_rsa", check=False)
        
        # 显示公钥
        with open(self.ssh_pub_key_path, 'r') as f:
            public_key = f.read().strip()
        print("\n📋 新生成的SSH公钥:")
        print(public_key)
        
        return True

    def setup_git_config(self):
        """设置Git配置"""
        print("🔧 设置Git配置...")
        
        # 检查现有配置
        current_name = self.run_command("git config --global user.name", check=False)
        current_email = self.run_command("git config --global user.email", check=False)
        
        if current_name and current_email:
            print(f"✅ Git已配置: {current_name} <{current_email}>")
            return True
        
        # 设置Git配置
        name = input("请输入您的Git用户名: ") or self.github_username
        email = input("请输入您的Git邮箱: ")
        
        self.run_command(f'git config --global user.name "{name}"')
        self.run_command(f'git config --global user.email "{email}"')
        
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

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Model files
*.pkl
*.joblib

# Large files
*.zip
*.tar.gz

# Temporary files
tmp/
temp/
*.tmp

# Credentials
*.pem
*.key
id_rsa*
"""
        
        with open('.gitignore', 'w') as f:
            f.write(gitignore_content)
        
        print("✅ 创建了.gitignore文件")

    def create_readme(self):
        """创建README.md文件"""
        readme_content = f"""# Factor Forecasting Project

## 🚀 项目概述
先进的因子预测系统，使用深度学习技术进行金融因子预测，支持4GPU分布式训练。

## 📊 当前训练状态
- **训练进度**: Epoch 0, 217+ iterations
- **训练速度**: ~5秒/iteration  
- **GPU利用率**: 4GPU分布式训练
- **显存使用**: 77.2% (71GB/92GB)

## 🧠 核心特性
- **4GPU分布式训练**: 充分利用多GPU资源
- **TCN + Attention架构**: 先进的时序建模
- **实时监控**: 训练进度和相关性监控
- **自适应内存管理**: 智能内存优化

## 💻 硬件配置
- **GPU**: 4x NVIDIA A10 (22GB显存)
- **内存**: 739GB RAM
- **CPU**: 128核心

## 🚀 快速开始

### 环境配置
```bash
# 激活虚拟环境
source venv/bin/activate

# 安装依赖（使用清华镜像）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 训练启动
```bash
# 4GPU分布式训练
torchrun --standalone --nproc_per_node=4 \\
    unified_complete_training_v2_fixed.py \\
    --config optimal_4gpu_config.yaml
```

## 👨‍💻 作者
{self.github_username} - https://github.com/{self.github_username}

## 📄 许可证
Private Repository - All Rights Reserved
"""
        
        with open('README.md', 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print("✅ 创建了README.md文件")

    def setup_local_repo(self):
        """设置本地Git仓库"""
        print("🔧 设置本地Git仓库...")
        
        # 初始化Git仓库（如果尚未初始化）
        if not Path(".git").exists():
            self.run_command("git init")
            self.run_command("git branch -M main")
        
        # 创建必要文件
        self.create_gitignore()
        self.create_readme()
        
        return True

    def commit_changes(self):
        """提交更改"""
        print("📝 提交更改...")
        
        # 添加所有文件
        self.run_command("git add .")
        
        # 检查是否有更改
        status = self.run_command("git status --porcelain", check=False)
        if not status:
            print("✅ 没有需要提交的更改")
            return True
        
        # 提交
        commit_msg = "feat: Advanced Factor Forecasting System with 4GPU Distributed Training"
        result = self.run_command(f'git commit -m "{commit_msg}"')
        
        return result is not None

    def setup_remote_and_push(self):
        """设置远程仓库并推送"""
        print("📤 设置远程仓库并推送...")
        
        # 设置远程仓库
        remote_url = f"git@github.com:{self.github_username}/{self.repo_name}.git"
        
        # 移除现有的origin（如果存在）
        self.run_command("git remote remove origin", check=False)
        
        # 添加新的origin
        self.run_command(f"git remote add origin {remote_url}")
        
        # 推送到GitHub
        print("🚀 推送到GitHub...")
        push_result = self.run_command("git push -u origin main", check=False)
        
        if push_result is None:
            print("\n❌ SSH推送失败，可能的原因:")
            print("1. SSH密钥未添加到GitHub")
            print("2. 仓库不存在")
            print("3. 网络连接问题")
            
            print("\n📋 请按以下步骤操作:")
            print("1. 复制上面显示的SSH公钥")
            print("2. 访问 https://github.com/settings/ssh/new")
            print("3. 添加SSH密钥")
            print(f"4. 在GitHub创建私有仓库: {self.repo_name}")
            print("5. 重新运行推送命令")
            
            return False
        
        print(f"🎉 代码已成功推送到: https://github.com/{self.github_username}/{self.repo_name}")
        return True

    def run_setup(self):
        """运行完整设置"""
        print("🚀 Factor Forecasting 简化GitHub设置")
        print("=" * 60)
        
        try:
            # 1. 设置Git配置
            if not self.setup_git_config():
                return False
            
            # 2. 检查SSH密钥
            if not self.check_ssh_keys():
                return False
            
            print("\n⚠️  重要提示:")
            print("请确保已完成以下步骤:")
            print("1. 复制上面显示的SSH公钥")
            print("2. 访问 https://github.com/settings/ssh/new 添加SSH密钥")
            print(f"3. 在GitHub创建私有仓库: {self.repo_name}")
            
            input("\n完成上述步骤后，按回车键继续...")
            
            # 3. 设置本地仓库
            if not self.setup_local_repo():
                return False
            
            # 4. 提交更改
            if not self.commit_changes():
                return False
            
            # 5. 推送到GitHub
            if not self.setup_remote_and_push():
                print("\n🔧 手动推送命令:")
                print(f"git remote add origin git@github.com:{self.github_username}/{self.repo_name}.git")
                print("git push -u origin main")
                return False
            
            print("\n🎊 GitHub仓库设置完成!")
            return True
            
        except KeyboardInterrupt:
            print("\n\n👋 用户取消操作")
            return False
        except Exception as e:
            print(f"\n❌ 设置过程中出现错误: {e}")
            return False

def main():
    setup = SimpleGitHubSetup()
    setup.run_setup()

if __name__ == "__main__":
    main()
