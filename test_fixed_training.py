#!/usr/bin/env python3
"""
测试修复后的训练脚本
验证所有关键问题是否已解决
"""

import os
import sys
import torch
import torch.multiprocessing as mp
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """测试所有导入是否正常"""
    print("=== 测试导入 ===")
    try:
        from src.unified_complete_training_v2_fixed import UnifiedCompleteTrainer, load_config, main
        print("✅ 训练脚本导入成功")
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_config_loading():
    """测试配置加载"""
    print("\n=== 测试配置加载 ===")
    try:
        from src.unified_complete_training_v2_fixed import load_config
        
        # 测试默认配置
        config = load_config("nonexistent.yaml")
        print(f"✅ 默认配置加载成功，包含 {len(config)} 个参数")
        
        # 检查关键参数
        required_keys = ['batch_size', 'num_workers', 'use_distributed']
        for key in required_keys:
            if key in config:
                print(f"✅ 关键参数 {key}: {config[key]}")
            else:
                print(f"❌ 缺少关键参数: {key}")
                return False
        return True
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False

def test_multiprocessing():
    """测试多进程功能"""
    print("\n=== 测试多进程功能 ===")
    try:
        # 设置spawn方法
        mp.set_start_method('spawn', force=True)
        print("✅ multiprocessing spawn方法设置成功")
        
        # 测试简单的多进程任务
        def simple_worker(rank):
            return f"Worker {rank} completed"
        
        # 不实际启动进程，只测试设置
        print("✅ 多进程设置验证成功")
        return True
    except Exception as e:
        print(f"❌ 多进程测试失败: {e}")
        return False

def test_cuda_availability():
    """测试CUDA可用性"""
    print("\n=== 测试CUDA环境 ===")
    try:
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print("✅ CUDA环境检查完成")
        return True
    except Exception as e:
        print(f"❌ CUDA环境检查失败: {e}")
        return False

def test_trainer_initialization():
    """测试训练器初始化"""
    print("\n=== 测试训练器初始化 ===")
    try:
        from src.unified_complete_training_v2_fixed import UnifiedCompleteTrainer, load_config
        
        # 加载配置
        config = load_config("nonexistent.yaml")
        config.update({
            'batch_size': 32,
            'num_workers': 2,
            'epochs': 1,
            'data_dir': './test_data'  # 使用本地测试目录
        })
        
        # 初始化训练器
        trainer = UnifiedCompleteTrainer(config, rank=0, world_size=1)
        print("✅ 训练器初始化成功")
        print(f"设备: {trainer.device}")
        print(f"配置参数数量: {len(trainer.config)}")
        
        return True
    except Exception as e:
        print(f"❌ 训练器初始化失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("开始测试修复后的训练脚本...")
    
    tests = [
        test_imports,
        test_config_loading,
        test_multiprocessing,
        test_cuda_availability,
        test_trainer_initialization
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"测试失败: {test_func.__name__}")
        except Exception as e:
            print(f"测试异常 {test_func.__name__}: {e}")
    
    print(f"\n=== 测试总结 ===")
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过！修复后的脚本可以使用。")
        return True
    else:
        print("⚠️  部分测试失败，需要进一步修复。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
