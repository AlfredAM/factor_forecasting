#!/usr/bin/env python3
"""
语法修复脚本 - 修复DataLoader参数中的语法错误
"""

import re

def fix_syntax_errors(file_path: str):
    """修复语法错误"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("开始修复语法错误...")
    
    # 修复DataLoader参数中的语法错误
    # 移除多余的数字和逗号
    content = re.sub(r',\s*\d+\),\s*persistent_workers', ', persistent_workers', content)
    content = re.sub(r'pin_memory=.*?,\s*\d+\)', 'pin_memory=True)', content)
    
    # 修复可能的其他语法问题
    content = re.sub(r',\s*,', ',', content)  # 移除重复逗号
    content = re.sub(r',\s*\)', ')', content)  # 移除函数调用末尾的多余逗号
    
    # 确保DataLoader参数格式正确
    dataloader_pattern = r'DataLoader\((.*?)\)'
    
    def fix_dataloader_args(match):
        args = match.group(1)
        # 清理参数
        args = re.sub(r',\s*\d+(?=\s*,|\s*\))', '', args)  # 移除孤立的数字
        args = re.sub(r',\s*,', ',', args)  # 移除重复逗号
        args = re.sub(r',\s*$', '', args)  # 移除末尾逗号
        return f'DataLoader({args})'
    
    content = re.sub(dataloader_pattern, fix_dataloader_args, content, flags=re.DOTALL)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ 语法错误修复完成: {file_path}")
    
    # 验证语法
    try:
        compile(content, file_path, 'exec')
        print("✅ 语法验证通过")
        return True
    except SyntaxError as e:
        print(f"❌ 仍有语法错误: {e}")
        print(f"错误位置: 第{e.lineno}行")
        # 打印错误行附近的内容
        lines = content.split('\n')
        start = max(0, e.lineno - 3)
        end = min(len(lines), e.lineno + 2)
        for i in range(start, end):
            marker = ">>> " if i + 1 == e.lineno else "    "
            print(f"{marker}{i+1:3d}: {lines[i]}")
        return False

if __name__ == "__main__":
    fix_syntax_errors("/nas/factor_forecasting/unified_complete_training_v2_fixed.py")
