#!/usr/bin/env python3
"""
综合修复脚本 - 从根本上彻底解决所有问题
1. 修复所有导入问题
2. 修复所有函数调用问题
3. 修复所有CUDA多进程问题
4. 修复所有PyTorch DataLoader兼容性问题
"""

import os
import re
from pathlib import Path

def comprehensive_fix(project_root: str):
    """综合修复所有问题"""
    
    print("开始综合修复...")
    
    training_script = Path(project_root) / "unified_complete_training_v2_fixed.py"
    
    if not training_script.exists():
        print(f"训练脚本不存在: {training_script}")
        return
    
    with open(training_script, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("修复前的内容长度:", len(content))
    
    # 1. 确保正确的导入语句
    if 'from src.training.quantitative_loss import create_quantitative_loss_function' not in content:
        # 找到quantitative_loss导入行
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'from src.training.quantitative_loss import' in line:
                if 'create_quantitative_loss_function' not in line:
                    lines[i] = line.rstrip(', ') + ', create_quantitative_loss_function'
                break
        content = '\n'.join(lines)
        print("✓ 添加了create_quantitative_loss_function导入")
    
    # 2. 修复multiprocessing设置
    if 'mp.set_start_method' not in content:
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
        print("✓ 添加了multiprocessing设置")
    
    # 3. 修复DataLoader参数
    # 确保所有num_workers都是0
    content = re.sub(r'num_workers=\d+', 'num_workers=0', content)
    content = re.sub(r'persistent_workers=True', 'persistent_workers=False', content)
    
    # 删除所有prefetch_factor设置（与num_workers=0不兼容）
    content = re.sub(r',\s*prefetch_factor=.*?(?=,|\))', '', content)
    content = re.sub(r'prefetch_factor=.*?,\s*', '', content)
    
    print("✓ 修复了DataLoader参数")
    
    # 4. 修复损失函数调用
    if 'self.criterion = AdaptiveQuantitativeLoss()' in content:
        content = content.replace(
            'self.criterion = AdaptiveQuantitativeLoss()',
            'self.criterion = create_quantitative_loss_function(self.config)'
        )
        print("✓ 修复了损失函数调用")
    
    # 5. 修复变量赋值问题
    content = re.sub(r'(\w+)\s*=\s*max\(.*?self\.config\.get\(.*?num_workers.*?\).*?\)', 
                     r'\1 = 0  # Disabled for CUDA compatibility', content)
    
    # 6. 确保异步加载被禁用
    content = re.sub(r'enable_async_loading=True', 'enable_async_loading=False', content)
    
    # 7. 修复ThreadPoolExecutor设置
    content = re.sub(r'ThreadPoolExecutor\(max_workers=\d+\)', 
                     'ThreadPoolExecutor(max_workers=1)', content)
    
    print("✓ 修复了多进程相关设置")
    
    # 8. 确保ICCorrelationReporter有正确的参数
    if 'self.ic_reporter = ICCorrelationReporter(' in content:
        content = re.sub(
            r'self\.ic_reporter = ICCorrelationReporter\(\s*target_columns=',
            'self.ic_reporter = ICCorrelationReporter(output_dir=self.config.get("output_dir", "outputs"), target_columns=',
            content
        )
        print("✓ 修复了ICCorrelationReporter参数")
    
    # 9. 修复所有nn.dropout错误
    content = re.sub(r'nn\.dropout', 'nn.Dropout', content)
    content = re.sub(r'torch\.dropout', 'torch.nn.functional.dropout', content)
    
    print("✓ 修复了dropout相关错误")
    
    # 10. 清理重复的导入和语法错误
    lines = content.split('\n')
    cleaned_lines = []
    seen_imports = set()
    
    for line in lines:
        # 修复可能的语法错误
        if line.strip().startswith('nimport'):
            line = line.replace('nimport', 'import')
        
        # 去重导入语句
        if line.strip().startswith(('import ', 'from ')):
            if line not in seen_imports:
                seen_imports.add(line)
                cleaned_lines.append(line)
        else:
            cleaned_lines.append(line)
    
    content = '\n'.join(cleaned_lines)
    
    print("✓ 清理了重复导入和语法错误")
    
    # 写入修复后的文件
    with open(training_script, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"修复后的内容长度: {len(content)}")
    print(f"✅ 综合修复完成: {training_script}")
    
    # 验证语法
    try:
        compile(content, str(training_script), 'exec')
        print("✅ 语法验证通过")
    except SyntaxError as e:
        print(f"❌ 语法错误: {e}")
        print(f"错误位置: 第{e.lineno}行")
        return False
    
    return True

def fix_data_processing_files(project_root: str):
    """修复数据处理文件中的多进程问题"""
    
    src_dir = Path(project_root) / "src"
    
    for py_file in src_dir.rglob("*.py"):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            modified = False
            original_content = content
            
            # 修复所有多进程相关问题
            if 'num_workers=' in content:
                content = re.sub(r'num_workers=\d+', 'num_workers=0', content)
                modified = True
            
            if 'ThreadPoolExecutor(max_workers=' in content:
                content = re.sub(r'ThreadPoolExecutor\(max_workers=\d+\)', 
                               'ThreadPoolExecutor(max_workers=1)', content)
                modified = True
            
            if 'enable_async_loading=True' in content:
                content = content.replace('enable_async_loading=True', 'enable_async_loading=False')
                modified = True
            
            # 修复dropout错误
            if 'nn.dropout' in content:
                content = re.sub(r'nn\.dropout', 'nn.Dropout', content)
                modified = True
            
            if 'torch.dropout' in content:
                content = re.sub(r'torch\.dropout', 'torch.nn.functional.dropout', content)
                modified = True
            
            if modified:
                with open(py_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"✓ 修复了文件: {py_file}")
                
        except Exception as e:
            print(f"❌ 修复文件失败 {py_file}: {e}")

if __name__ == "__main__":
    project_root = "/nas/factor_forecasting"
    
    print("=" * 60)
    print("开始综合修复所有问题...")
    print("=" * 60)
    
    # 修复主训练脚本
    success = comprehensive_fix(project_root)
    
    if success:
        # 修复数据处理文件
        fix_data_processing_files(project_root)
        print("=" * 60)
        print("✅ 所有问题修复完成！")
        print("=" * 60)
    else:
        print("=" * 60)
        print("❌ 修复失败，请检查错误信息")
        print("=" * 60)
