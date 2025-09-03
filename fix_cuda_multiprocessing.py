#!/usr/bin/env python3
"""
CUDA多进程问题彻底修复脚本
从根本上解决PyTorch CUDA与multiprocessing的兼容性问题
"""

import os
import sys
import re
from pathlib import Path

def fix_cuda_multiprocessing_issues(project_root: str):
    """彻底修复CUDA多进程问题"""
    
    print("开始修复CUDA多进程问题...")
    
    # 1. 修复训练脚本开头的multiprocessing设置
    training_script = Path(project_root) / "unified_complete_training_v2_fixed.py"
    if training_script.exists():
        with open(training_script, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 在开头添加multiprocessing设置
        if 'mp.set_start_method' not in content:
            lines = content.split('\n')
            # 找到import语句后插入
            insert_idx = 0
            for i, line in enumerate(lines):
                if line.startswith('import torch'):
                    insert_idx = i + 1
                    break
            
            lines.insert(insert_idx, "")
            lines.insert(insert_idx + 1, "# Fix CUDA multiprocessing issues")
            lines.insert(insert_idx + 2, "import torch.multiprocessing as mp")
            lines.insert(insert_idx + 3, "try:")
            lines.insert(insert_idx + 4, "    mp.set_start_method('spawn', force=True)")
            lines.insert(insert_idx + 5, "except RuntimeError:")
            lines.insert(insert_idx + 6, "    pass  # Already set")
            lines.insert(insert_idx + 7, "")
            
            with open(training_script, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            print(f"修复了训练脚本的multiprocessing设置")
    
    # 2. 修复所有数据加载器的worker设置
    src_dir = Path(project_root) / "src"
    for py_file in src_dir.rglob("*.py"):
        with open(py_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        modified = False
        
        # 禁用DataLoader的多进程
        if 'num_workers=' in content:
            content = re.sub(r'num_workers=\d+', 'num_workers=0', content)
            modified = True
        
        # 禁用ThreadPoolExecutor的多进程
        if 'ThreadPoolExecutor(max_workers=' in content:
            content = re.sub(r'ThreadPoolExecutor\(max_workers=\d+\)', 
                           'ThreadPoolExecutor(max_workers=1)', content)
            modified = True
        
        # 禁用异步加载
        if 'enable_async_loading=True' in content:
            content = content.replace('enable_async_loading=True', 'enable_async_loading=False')
            modified = True
        
        if modified:
            with open(py_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"修复了文件: {py_file}")
    
    # 3. 修复特定文件的executor设置
    streaming_loader = src_dir / "data_processing" / "optimized_streaming_loader.py"
    if streaming_loader.exists():
        with open(streaming_loader, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 确保executor为None或使用单线程
        content = re.sub(
            r'self\.executor = ThreadPoolExecutor\(max_workers=max_workers\) if enable_async_loading else None',
            'self.executor = None  # Disabled to avoid CUDA multiprocessing issues',
            content
        )
        
        with open(streaming_loader, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"修复了优化数据加载器")
    
    print("CUDA多进程问题修复完成!")

if __name__ == "__main__":
    project_root = "/nas/factor_forecasting"
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    
    fix_cuda_multiprocessing_issues(project_root)
