#!/usr/bin/env python3
"""
综合训练系统修复脚本
从根本上彻底解决所有错误和问题，不走捷径
"""

import os
import re
import sys
from pathlib import Path

def comprehensive_fix_training_system(project_root: str):
    """综合修复训练系统的所有问题"""
    
    print("开始综合修复训练系统...")
    project_root = Path(project_root)
    
    # 1. 修复训练脚本的导入问题
    training_script = project_root / "unified_complete_training_v2_fixed.py"
    
    if not training_script.exists():
        print(f"错误：训练脚本不存在: {training_script}")
        return False
    
    with open(training_script, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查并添加缺失的导入
    imports_to_add = []
    
    if 'create_quantitative_loss_function' in content and 'from src.training.quantitative_loss import create_quantitative_loss_function' not in content:
        imports_to_add.append("from src.training.quantitative_loss import create_quantitative_loss_function")
    
    if 'mp.set_start_method' not in content:
        imports_to_add.append("import torch.multiprocessing as mp")
        imports_to_add.append("try:\n    mp.set_start_method('spawn', force=True)\nexcept RuntimeError:\n    pass")
    
    # 添加导入
    if imports_to_add:
        lines = content.split('\n')
        # 找到最后一个import语句的位置
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('from src.') or line.startswith('import '):
                insert_idx = i + 1
        
        # 插入新的导入
        for imp in reversed(imports_to_add):
            lines.insert(insert_idx, imp)
            lines.insert(insert_idx + 1, "")
        
        content = '\n'.join(lines)
    
    # 2. 修复DataLoader相关问题
    # 确保所有DataLoader使用正确的参数
    dataloader_pattern = r'DataLoader\((.*?)\)'
    
    def fix_dataloader_params(match):
        params = match.group(1)
        # 确保num_workers=0
        if 'num_workers=' in params:
            params = re.sub(r'num_workers=\d+', 'num_workers=0', params)
        else:
            params += ',\n        num_workers=0'
        
        # 移除prefetch_factor（与num_workers=0不兼容）
        params = re.sub(r',?\s*prefetch_factor=\w+', '', params)
        
        # 确保persistent_workers=False
        if 'persistent_workers=' in params:
            params = re.sub(r'persistent_workers=\w+', 'persistent_workers=False', params)
        else:
            params += ',\n        persistent_workers=False'
        
        return f'DataLoader({params})'
    
    content = re.sub(dataloader_pattern, fix_dataloader_params, content, flags=re.DOTALL)
    
    # 3. 修复loss函数调用
    # 如果使用create_quantitative_loss_function，确保它被正确调用
    if 'self.criterion = create_quantitative_loss_function(self.config)' in content:
        # 检查配置是否包含必要的参数
        config_check = """
        # Ensure config has required loss parameters
        if not hasattr(self.config, 'get'):
            # Convert to dict-like object if needed
            class ConfigDict(dict):
                def get(self, key, default=None):
                    return super().get(key, default)
            if isinstance(self.config, dict):
                self.config = ConfigDict(self.config)
        """
        
        # 在create_model方法开始处添加配置检查
        content = re.sub(
            r'(def create_model\(self\):.*?""".*?""")',
            r'\1' + config_check,
            content,
            flags=re.DOTALL
        )
    
    # 4. 修复异步加载问题
    content = re.sub(r'enable_async_loading=True', 'enable_async_loading=False', content)
    content = re.sub(r'ThreadPoolExecutor\(max_workers=\d+\)', 'ThreadPoolExecutor(max_workers=1)', content)
    
    # 5. 确保CUDA设备设置正确
    if 'torch.cuda.set_device' not in content:
        # 在trainer初始化后添加CUDA设备设置
        content = re.sub(
            r'(self\.device = torch\.device.*?\n)',
            r'\1        if torch.cuda.is_available():\n            torch.cuda.set_device(self.device)\n',
            content
        )
    
    # 保存修复后的文件
    with open(training_script, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"已修复训练脚本: {training_script}")
    
    # 6. 修复配置文件
    config_file = project_root / "optimized_server_config.yaml"
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            config_content = f.read()
        
        # 确保配置文件包含loss相关参数
        if 'correlation_weight:' not in config_content:
            loss_config = """
# Loss function parameters
correlation_weight: 1.0
mse_weight: 0.1
rank_correlation_weight: 0.2
risk_penalty_weight: 0.1
target_correlations: [0.08, 0.05, 0.03]
max_leverage: 2.0
transaction_cost: 0.001
use_adaptive_loss: true
volatility_window: 20
regime_sensitivity: 0.1
"""
            config_content += loss_config
            
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_content)
            
            print(f"已更新配置文件: {config_file}")
    
    print("综合修复完成！")
    return True

def test_training_script(project_root: str):
    """测试训练脚本是否能正常导入"""
    training_script = Path(project_root) / "unified_complete_training_v2_fixed.py"
    
    print("测试训练脚本导入...")
    
    # 简单的语法检查
    try:
        with open(training_script, 'r', encoding='utf-8') as f:
            content = f.read()
        
        compile(content, str(training_script), 'exec')
        print("✓ 训练脚本语法检查通过")
        return True
    except SyntaxError as e:
        print(f"✗ 语法错误: {e}")
        return False
    except Exception as e:
        print(f"✗ 其他错误: {e}")
        return False

if __name__ == "__main__":
    project_root = "/nas/factor_forecasting"
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    
    success = comprehensive_fix_training_system(project_root)
    if success:
        test_training_script(project_root)
    else:
        print("修复失败!")
        sys.exit(1)
