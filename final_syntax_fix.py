#!/usr/bin/env python3
"""
最终语法修复脚本
彻底修复所有语法错误
"""

import re

def fix_final_syntax_errors(file_path: str):
    """修复最终的语法错误"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复OptimizedStreamingDataLoader调用的语法错误
    # 错误的代码结构：
    # OptimizedStreamingDataLoader(
    #     data_dir=self.config.get('data_dir', '/nas/feature_v2_10s',
    #     num_workers=0,
    #     persistent_workers=False),
    #     memory_manager=...
    # )
    
    # 查找并修复这种错误的参数结构
    pattern = r'OptimizedStreamingDataLoader\(\s*data_dir=self\.config\.get\([^)]+\),\s*num_workers=\d+,\s*persistent_workers=\w+\),\s*memory_manager='
    
    if re.search(pattern, content, re.DOTALL):
        print("发现OptimizedStreamingDataLoader语法错误，正在修复...")
        
        # 重新构造正确的调用
        fixed_call = '''OptimizedStreamingDataLoader(
            data_dir=self.config.get('data_dir', '/nas/feature_v2_10s'),
            memory_manager='''
        
        content = re.sub(
            r'OptimizedStreamingDataLoader\(\s*data_dir=self\.config\.get\([^)]+\),\s*num_workers=\d+,\s*persistent_workers=\w+\),\s*memory_manager=',
            fixed_call,
            content,
            flags=re.DOTALL
        )
    
    # 修复其他可能的语法问题
    # 1. 修复多余的逗号
    content = re.sub(r',\s*\)', ')', content)
    
    # 2. 修复DataLoader参数
    content = re.sub(
        r'DataLoader\(\s*([^,]+),\s*batch_size=([^,]+),\s*num_workers=0,\s*persistent_workers=False([^)]*)\)',
        r'DataLoader(\1, batch_size=\2, num_workers=0, persistent_workers=False\3)',
        content,
        flags=re.DOTALL
    )
    
    # 3. 确保所有函数调用的括号匹配
    # 简单检查括号平衡
    def check_parentheses_balance(text):
        count = 0
        for char in text:
            if char == '(':
                count += 1
            elif char == ')':
                count -= 1
        return count == 0
    
    if not check_parentheses_balance(content):
        print("警告：检测到括号不平衡，请手动检查")
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"语法修复完成: {file_path}")

def validate_python_syntax(file_path: str) -> bool:
    """验证Python语法"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        compile(content, file_path, 'exec')
        print(f"✓ 语法验证通过: {file_path}")
        return True
    except SyntaxError as e:
        print(f"✗ 语法错误 {file_path}: 行{e.lineno}: {e.msg}")
        print(f"  错误位置: {e.text}")
        return False

if __name__ == "__main__":
    file_path = "/nas/factor_forecasting/unified_complete_training_v2_fixed.py"
    
    print("开始最终语法修复...")
    fix_final_syntax_errors(file_path)
    
    print("验证语法...")
    if validate_python_syntax(file_path):
        print("所有语法错误已修复！")
    else:
        print("仍有语法错误需要修复")
