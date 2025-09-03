#!/usr/bin/env python3
"""
彻底修复Tensor.__contains__错误
问题根源：在损失函数中使用字符串检查tensor对象
"""

import os
import re
from pathlib import Path

def fix_tensor_contains_error():
    """修复所有可能的tensor.__contains__错误"""
    
    fixes_applied = []
    
    # 1. 修复 quantitative_loss.py 中的问题
    loss_file = Path("src/training/quantitative_loss.py")
    if loss_file.exists():
        with open(loss_file, 'r') as f:
            content = f.read()
        
        # 修复第77行的问题：检查targets是否为字典
        old_pattern = r'if target_name in predictions and target_name in targets:'
        new_pattern = 'if target_name in predictions and (isinstance(targets, dict) and target_name in targets):'
        
        if old_pattern in content:
            content = content.replace(old_pattern, new_pattern)
            fixes_applied.append("修复quantitative_loss.py中的tensor.__contains__问题")
        
        # 确保所有类似的检查都被修复
        content = re.sub(
            r'(\w+)\s+in\s+(targets)(?!\s*\.)',
            r'(isinstance(\2, dict) and \1 in \2)',
            content
        )
        
        with open(loss_file, 'w') as f:
            f.write(content)
        
        fixes_applied.append(f"更新了{loss_file}")
    
    # 2. 修复 trainer.py 中的类似问题
    trainer_file = Path("src/training/trainer.py")
    if trainer_file.exists():
        with open(trainer_file, 'r') as f:
            content = f.read()
        
        # 修复CorrelationLoss中的问题
        old_pattern = r'if target_name in predictions and target_name in targets:'
        new_pattern = 'if target_name in predictions and (isinstance(targets, dict) and target_name in targets):'
        
        if old_pattern in content:
            content = content.replace(old_pattern, new_pattern)
            fixes_applied.append("修复trainer.py中的tensor.__contains__问题")
        
        with open(trainer_file, 'w') as f:
            f.write(content)
        
        fixes_applied.append(f"更新了{trainer_file}")
    
    # 3. 修复其他可能的文件
    for file_path in Path("src").rglob("*.py"):
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            original_content = content
            
            # 查找并修复所有可能的tensor.__contains__问题
            patterns_to_fix = [
                (r'if\s+(\w+)\s+in\s+(targets)(?!\s*\.)', r'if isinstance(\2, dict) and \1 in \2'),
                (r'(\w+)\s+in\s+(predictions)(?!\s*\.)(?!\s*and)', r'(isinstance(\2, dict) and \1 in \2)'),
                (r'if\s+(\w+)\s+in\s+(batch\[.*?\])(?!\s*\.)', r'if isinstance(\2, dict) and \1 in \2'),
            ]
            
            for old_pattern, new_pattern in patterns_to_fix:
                content = re.sub(old_pattern, new_pattern, content)
            
            if content != original_content:
                with open(file_path, 'w') as f:
                    f.write(content)
                fixes_applied.append(f"修复了{file_path}中的潜在tensor.__contains__问题")
                
        except Exception as e:
            print(f"处理文件{file_path}时出错: {e}")
            continue
    
    # 4. 特别处理unified_complete_training_v2_fixed.py中的数据处理
    main_file = Path("src/unified_complete_training_v2_fixed.py")
    if main_file.exists():
        with open(main_file, 'r') as f:
            content = f.read()
        
        # 确保targets处理正确
        target_processing_fix = '''
                # Normalize target shapes to (batch,) when possible
                if isinstance(targets, dict):
                    normalized_targets = {}
                    for t_name, t_val in targets.items():
                        if t_val.dim() == 3 and t_val.size(-1) == 1:
                            t_val = t_val.squeeze(-1)
                        if t_val.dim() == 2:
                            # take last horizon step
                            t_val = t_val[:, -1]
                        normalized_targets[t_name] = t_val
                    targets = normalized_targets
                elif isinstance(targets, torch.Tensor):
                    # 如果targets是tensor，转换为字典格式
                    target_cols = self.config.get('target_columns', ['intra30m', 'nextT1d', 'ema1d'])
                    if targets.dim() == 2 and targets.size(1) == len(target_cols):
                        targets = {col: targets[:, i] for i, col in enumerate(target_cols)}
                    elif targets.dim() == 1:
                        # 假设只有一个目标
                        targets = {target_cols[0]: targets}
'''
        
        # 替换目标处理部分
        old_target_processing = r'# Normalize target shapes to (batch,) when possible\s*if isinstance\(targets, dict\):.*?targets = normalized_targets'
        content = re.sub(old_target_processing, target_processing_fix.strip(), content, flags=re.DOTALL)
        
        with open(main_file, 'w') as f:
            f.write(content)
        
        fixes_applied.append("修复了unified_complete_training_v2_fixed.py中的targets处理")
    
    # 5. 创建一个专门的类型检查函数
    utils_file = Path("src/utils/tensor_utils.py")
    utils_file.parent.mkdir(parents=True, exist_ok=True)
    
    utils_content = '''#!/usr/bin/env python3
"""
Tensor工具函数，避免__contains__错误
"""

import torch
from typing import Dict, Union, Any

def safe_dict_check(key: str, container: Union[Dict, torch.Tensor, Any]) -> bool:
    """
    安全地检查key是否在container中，避免tensor.__contains__错误
    
    Args:
        key: 要检查的键
        container: 容器对象
        
    Returns:
        bool: 如果key在container中返回True，否则返回False
    """
    if isinstance(container, dict):
        return key in container
    elif isinstance(container, torch.Tensor):
        return False  # tensor不应该用于key检查
    elif hasattr(container, '__contains__') and not isinstance(container, torch.Tensor):
        try:
            return key in container
        except (TypeError, RuntimeError):
            return False
    else:
        return False

def ensure_targets_dict(targets: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                       target_columns: list) -> Dict[str, torch.Tensor]:
    """
    确保targets是字典格式
    
    Args:
        targets: 目标值，可能是tensor或字典
        target_columns: 目标列名列表
        
    Returns:
        Dict[str, torch.Tensor]: 字典格式的目标值
    """
    if isinstance(targets, dict):
        return targets
    elif isinstance(targets, torch.Tensor):
        if targets.dim() == 2 and targets.size(1) == len(target_columns):
            return {col: targets[:, i] for i, col in enumerate(target_columns)}
        elif targets.dim() == 1:
            # 假设只有一个目标
            return {target_columns[0]: targets}
        else:
            # 默认处理
            return {target_columns[0]: targets.flatten()}
    else:
        raise TypeError(f"不支持的targets类型: {type(targets)}")
'''
    
    with open(utils_file, 'w') as f:
        f.write(utils_content)
    
    fixes_applied.append(f"创建了{utils_file}工具函数")
    
    return fixes_applied

if __name__ == "__main__":
    print("🔧 开始修复Tensor.__contains__错误...")
    
    fixes = fix_tensor_contains_error()
    
    print("✅ 修复完成！")
    print("应用的修复:")
    for fix in fixes:
        print(f"  - {fix}")
    
    print("\n🎯 修复说明:")
    print("1. 修复了损失函数中的字符串与tensor比较问题")
    print("2. 确保所有targets都正确转换为字典格式")
    print("3. 添加了类型检查以避免类似错误")
    print("4. 创建了安全的工具函数")
