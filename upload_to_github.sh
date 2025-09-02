#!/bin/bash

echo "=================================="
echo "Factor Forecasting Project"
echo "GitHub 私有仓库上传脚本"
echo "=================================="

# 检查Git状态
echo "=== 检查Git状态 ==="
git status

# 添加新文件
echo ""
echo "=== 添加项目报告 ==="
git add PROJECT_REPORT.md
git add upload_to_github.sh

# 创建新的提交
echo ""
echo "=== 创建提交 ==="
git commit -m "Add comprehensive project report and GitHub upload script

- Complete training analysis with 1115 iterations in 37 minutes
- IC correlation results: nextT1d=0.007, intra30m=-0.018, ema1d=-0.054
- Loss convergence: 2.68 → 0.14 (94.7% improvement)
- Server performance: Dual A10 GPUs, 2.08s/iteration
- Production-ready system with full monitoring"

# 显示远程仓库信息
echo ""
echo "=== 远程仓库信息 ==="
git remote -v

# 推送到GitHub的说明
echo ""
echo "=== GitHub上传说明 ==="
echo "1. 请先在GitHub上手动创建私有仓库："
echo "   - 访问: https://github.com/AlfredAM"
echo "   - 点击 'New repository'"
echo "   - 仓库名: factor_forecasting"
echo "   - 设置为 Private"
echo "   - 不要初始化README (我们已有本地版本)"
echo ""
echo "2. 然后执行以下命令推送代码："
echo "   git push -u origin master"
echo ""
echo "3. 如果需要认证，请使用GitHub Personal Access Token"
echo ""

# 显示项目统计
echo "=== 项目统计 ==="
echo "总文件数: $(find . -type f ! -path './.git/*' ! -path './venv/*' ! -path './data/*' | wc -l)"
echo "代码行数: $(find . -name '*.py' ! -path './.git/*' ! -path './venv/*' | xargs wc -l | tail -1)"
echo "配置文件: $(find . -name '*.yaml' -o -name '*.yml' ! -path './.git/*' ! -path './venv/*' | wc -l)"

echo ""
echo "=== 项目已准备就绪，可以推送到GitHub！ ==="
echo "仓库地址: https://github.com/AlfredAM/factor_forecasting"
echo ""
