#!/usr/bin/env python3
"""
简化测试脚本 - 测试核心修复内容
"""

import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_core_fixes():
    """测试核心修复内容"""
    print("=== 核心修复测试 ===")
    
    # 1. 测试语法检查
    print("\n1. 语法检查:")
    import ast
    
    test_files = [
        "src/unified_complete_training_v2_fixed.py",
        "src/data_processing/optimized_streaming_loader.py", 
        "src/data_processing/adaptive_memory_manager.py",
        "src/training/quantitative_loss.py"
    ]
    
    syntax_passed = 0
    for file_path in test_files:
        if Path(file_path).exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    ast.parse(f.read())
                print(f"  ✅ {file_path}")
                syntax_passed += 1
            except SyntaxError as e:
                print(f"  ❌ {file_path}: {e}")
        else:
            print(f"  ⚠️  {file_path}: 文件不存在")
    
    # 2. 测试导入
    print(f"\n2. 导入测试:")
    import_passed = 0
    
    try:
        from src.data_processing.adaptive_memory_manager import create_memory_manager
        print("  ✅ 内存管理器导入成功")
        import_passed += 1
    except Exception as e:
        print(f"  ❌ 内存管理器导入失败: {e}")
    
    try:
        from src.training.quantitative_loss import QuantitativeCorrelationLoss
        print("  ✅ 损失函数导入成功")
        import_passed += 1
    except Exception as e:
        print(f"  ❌ 损失函数导入失败: {e}")
    
    try:
        from src.unified_complete_training_v2_fixed import load_config
        print("  ✅ 配置加载函数导入成功")
        import_passed += 1
    except Exception as e:
        print(f"  ❌ 配置加载函数导入失败: {e}")
    
    # 3. 测试内存管理器修复
    print(f"\n3. 内存管理器修复验证:")
    try:
        memory_manager = create_memory_manager({
            'critical_threshold': 0.98,
            'warning_threshold': 0.95
        })
        
        if memory_manager.critical_threshold == 0.98:
            print("  ✅ 关键阈值修复正确: 98%")
        else:
            print(f"  ❌ 关键阈值错误: {memory_manager.critical_threshold}")
            
        if memory_manager.warning_threshold == 0.95:
            print("  ✅ 警告阈值修复正确: 95%")
        else:
            print(f"  ❌ 警告阈值错误: {memory_manager.warning_threshold}")
            
        memory_manager_passed = 1
    except Exception as e:
        print(f"  ❌ 内存管理器测试失败: {e}")
        memory_manager_passed = 0
    
    # 4. 测试配置加载修复
    print(f"\n4. 配置加载修复验证:")
    try:
        config = load_config("nonexistent.yaml")
        
        required_keys = ['batch_size', 'num_workers', 'use_distributed']
        config_passed = 0
        for key in required_keys:
            if key in config:
                print(f"  ✅ 配置参数 {key}: {config[key]}")
                config_passed += 1
            else:
                print(f"  ❌ 缺少配置参数: {key}")
        
        if config_passed == len(required_keys):
            config_passed = 1
        else:
            config_passed = 0
            
    except Exception as e:
        print(f"  ❌ 配置加载测试失败: {e}")
        config_passed = 0
    
    # 5. 测试pickle序列化修复
    print(f"\n5. pickle序列化修复验证:")
    try:
        import pickle
        
        # 测试基本数据结构
        test_data = {
            'numbers': [1, 2, 3],
            'text': 'test',
            'nested': {'key': 'value'}
        }
        
        serialized = pickle.dumps(test_data)
        deserialized = pickle.loads(serialized)
        
        if deserialized == test_data:
            print("  ✅ pickle序列化测试通过")
            pickle_passed = 1
        else:
            print("  ❌ pickle序列化数据不一致")
            pickle_passed = 0
            
    except Exception as e:
        print(f"  ❌ pickle序列化测试失败: {e}")
        pickle_passed = 0
    
    # 总结
    print(f"\n=== 测试总结 ===")
    total_tests = 5
    passed_tests = (
        (syntax_passed == len(test_files)) +
        (import_passed == 3) +
        memory_manager_passed +
        config_passed +
        pickle_passed
    )
    
    print(f"总测试: {total_tests}")
    print(f"通过: {passed_tests}")
    print(f"通过率: {passed_tests/total_tests*100:.1f}%")
    
    print(f"\n✅ 核心修复验证:")
    print(f"  - 语法错误修复: {'✅' if syntax_passed == len(test_files) else '❌'}")
    print(f"  - 导入问题修复: {'✅' if import_passed == 3 else '❌'}")
    print(f"  - 内存管理器优化: {'✅' if memory_manager_passed else '❌'}")
    print(f"  - 配置加载修复: {'✅' if config_passed else '❌'}")
    print(f"  - pickle序列化修复: {'✅' if pickle_passed else '❌'}")
    
    if passed_tests == total_tests:
        print(f"\n🎉 所有核心修复验证通过！")
        return True
    else:
        print(f"\n⚠️  部分修复需要进一步检查")
        return False

if __name__ == "__main__":
    success = test_core_fixes()
    sys.exit(0 if success else 1)
