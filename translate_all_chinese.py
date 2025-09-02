#!/usr/bin/env python3
"""
Translate all Chinese comments, docstrings, and logs to English in the project.
Remove emojis from all files.
"""
import os
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

# Translation dictionary for common Chinese phrases
TRANSLATIONS = {
    # Common terms
    "": "data",
    "": "model", 
    "": "training",
    "": "test",
    "": "validation",
    "": "configuration",
    "": "parameters",
    "": "features",
    "": "target",
    "": "prediction",
    "": "loss",
    "": "optimization",
    "": "learning rate",
    "": "batch",
    "": "sequence",
    "": "time",
    "": "step",
    "": "iteration",
    "": "epoch",
    "": "accuracy",
    "": "precision",
    "": "recall",
    "": "metric",
    "": "evaluation",
    "": "result",
    "": "output",
    "": "input",
    "": "file",
    "": "directory",
    "": "path",
    "": "load",
    "": "save",
    "": "process",
    "": "generate",
    "": "create",
    "": "initialize",
    "": "setup",
    "": "start",
    "": "run",
    "": "execute",
    "": "complete",
    "": "error",
    "": "exception",
    "": "warning",
    "": "info",
    "": "log",
    "": "monitor",
    "": "status",
    "": "progress",
    "": "success",
    "": "failure",
    
    # Specific project terms
    "": "factor forecasting",
    "": "attention mechanism", 
    "Transformer": "Transformer",
    "TCN": "TCN",
    "": "neural network",
    "": "deep learning",
    "": "machine learning",
    "": "time series",
    "": "rolling window",
    "": "distributed training",
    "GPU": "multi-GPU",
    "": "parallel computing",
    "": "memory management",
    "": "data loader",
    "": "dataset",
    "": "streaming processing",
    "": "async loading",
    "": "cache",
    "": "checkpoint",
    "": "model saving",
    "": "model loading",
    "": "inference",
    "": "quantization",
    "": "correlation",
    "IC": "IC",
    "": "risk management",
    "": "backtesting",
    "": "out-of-sample",
    "": "in-sample",
    
    # Phrases and common expressions
    "": "Initialize data loader",
    "": "Create model",
    "": "Setup distributed training",
    "": "Start training",
    "": "Save checkpoint",
    "": "Load checkpoint",
    "": "Calculate loss",
    "": "Update parameters",
    "": "Validate model",
    "": "Test model",
    "": "Generate predictions",
    "": "Evaluate performance",
    "": "Monitor training",
    "": "Log information",
    "": "Process data",
    "": "Load data",
    "": "Preprocessing",
    "": "Postprocessing",
    "": "Data cleaning",
    "": "Feature engineering",
    "": "Feature extraction",
    "": "Data augmentation",
    "": "Normalization",
    "": "Normalization",
    "": "Anomaly detection",
    "": "Error handling",
    "": "Exception handling",
    "": "Resource cleanup",
    "": "Memory cleanup",
    "": "Cache cleanup",
    "": "Process management",
    "": "Thread management",
    "GPU": "GPU management",
    "": "GPU memory management",
    
    # File and project specific
    "": "Unified Complete Training System",
    "": "Quantitative Correlation Loss",
    "": "Optimized Streaming Data Loader",
    "": "Adaptive Memory Manager",
    "TCN": "Advanced TCN Attention Model",
    "": "Yearly Rolling Trainer",
    "IC": "IC Correlation Reporter",
    "": "Risk Manager",
    "": "Model Benchmarker",
    "": "Quantitative Metrics Calculator",
}

def remove_emojis(text: str) -> str:
    """Remove emojis from text."""
    # Unicode ranges for emojis
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-a
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

def translate_chinese_text(text: str) -> str:
    """Translate Chinese text to English using the translation dictionary."""
    # Remove emojis first
    text = remove_emojis(text)
    
    # Sort by length (longest first) to handle phrases before individual words
    sorted_translations = sorted(TRANSLATIONS.items(), key=lambda x: len(x[0]), reverse=True)
    
    for chinese, english in sorted_translations:
        text = text.replace(chinese, english)
    
    return text

def translate_python_file(file_path: Path) -> bool:
    """Translate Chinese content in a Python file to English."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Remove emojis from the entire content
        content = remove_emojis(content)
        
        # Translate Chinese text in comments and strings
        lines = content.split('\n')
        translated_lines = []
        
        for line in lines:
            # Check if line contains Chinese characters
            if re.search(r'[\u4e00-\u9fff]', line):
                # Translate the line
                translated_line = translate_chinese_text(line)
                translated_lines.append(translated_line)
            else:
                translated_lines.append(line)
        
        translated_content = '\n'.join(translated_lines)
        
        # Only write if content changed
        if translated_content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(translated_content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def translate_yaml_file(file_path: Path) -> bool:
    """Translate Chinese content in YAML files."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Remove emojis and translate Chinese
        content = remove_emojis(content)
        content = translate_chinese_text(content)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def translate_project(project_root: str):
    """Translate all files in the project."""
    project_path = Path(project_root)
    
    # File patterns to process
    patterns = [
        "**/*.py",
        "**/*.yaml", 
        "**/*.yml",
        "**/*.md",
        "**/*.txt",
        "**/*.sh"
    ]
    
    translated_files = []
    
    for pattern in patterns:
        for file_path in project_path.glob(pattern):
            # Skip virtual environment and other ignore directories
            if any(part in str(file_path) for part in ['venv', '.git', '__pycache__', '.pytest_cache']):
                continue
            
            print(f"Processing: {file_path}")
            
            if file_path.suffix == '.py':
                if translate_python_file(file_path):
                    translated_files.append(str(file_path))
            elif file_path.suffix in ['.yaml', '.yml']:
                if translate_yaml_file(file_path):
                    translated_files.append(str(file_path))
            else:
                # For other text files
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    original_content = content
                    content = remove_emojis(content)
                    content = translate_chinese_text(content)
                    
                    if content != original_content:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        translated_files.append(str(file_path))
                        
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    print(f"\nTranslation completed!")
    print(f"Modified files: {len(translated_files)}")
    for file_path in translated_files:
        print(f"  - {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate Chinese to English in project files")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    
    args = parser.parse_args()
    translate_project(args.project_root)
