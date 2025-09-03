#!/usr/bin/env python3
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
