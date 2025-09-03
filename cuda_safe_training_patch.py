#!/usr/bin/env python3
"""
CUDA安全训练补丁
彻底修复所有多进程和CUDA兼容性问题
"""

import re

def apply_cuda_safe_patches(file_path: str):
    """应用CUDA安全补丁"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. 在文件开头添加multiprocessing设置
    if 'mp.set_start_method' not in content:
        # 找到第一个import torch语句后插入
        lines = content.split('\n')
        insert_idx = 0
        for i, line in enumerate(lines):
            if 'import torch' in line and not line.strip().startswith('#'):
                insert_idx = i + 1
                break
        
        lines.insert(insert_idx, "")
        lines.insert(insert_idx + 1, "# Fix CUDA multiprocessing compatibility")
        lines.insert(insert_idx + 2, "import torch.multiprocessing as mp")
        lines.insert(insert_idx + 3, "try:")
        lines.insert(insert_idx + 4, "    mp.set_start_method('spawn', force=True)")
        lines.insert(insert_idx + 5, "except RuntimeError:")
        lines.insert(insert_idx + 6, "    pass")
        lines.insert(insert_idx + 7, "")
        
        content = '\n'.join(lines)
    
    # 2. 修复DataLoader参数
    content = re.sub(r'num_workers=\d+', 'num_workers=0', content)
    content = re.sub(r'persistent_workers=True', 'persistent_workers=False', content)
    content = re.sub(r'prefetch_factor=\d+', 'prefetch_factor=2', content)
    
    # 3. 修复变量赋值
    content = re.sub(r'(\w+) = max\(.*?self\.config\.get\(.*?num_workers.*?\).*?\)', 
                     r'\1 = 0  # Disabled for CUDA compatibility', content)
    
    # 4. 确保所有DataLoader都使用0个worker
    content = re.sub(r'DataLoader\((.*?)num_workers=\d+(.*?)\)', 
                     r'DataLoader(\1num_workers=0\2)', content, flags=re.DOTALL)
    
    # 5. 禁用异步加载
    content = re.sub(r'enable_async_loading=True', 'enable_async_loading=False', content)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"已应用CUDA安全补丁到: {file_path}")

if __name__ == "__main__":
    apply_cuda_safe_patches("/nas/factor_forecasting/unified_complete_training_v2_fixed.py")
