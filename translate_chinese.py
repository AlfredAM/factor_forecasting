#!/usr/bin/env python3
"""
Batch translation script to convert Chinese comments and messages to English
and remove emoji icons from the codebase.
"""

import os
import re
import fnmatch
from pathlib import Path

# Translation dictionary for common Chinese terms
TRANSLATION_MAP = {
    # Technical terms
    "data": "data",
    "model": "model",
    "training": "training",
    "testing": "testing",
    "validate": "validation",
    "prediction": "prediction",
    "features": "feature",
    "target": "target",
    "configuration": "configuration",
    "parameter": "parameter",
    "optimization": "optimization",
    "learning": "learning",
    "batch": "batch",
    "size": "size",
    "path": "path",
    "file": "file",
    "directory": "directory",
    "log": "log",
    "error": "error",
    "warning": "warning",
    "info": "info",
    "debug": "debug",
    "start": "start",
    "end": "end",
    "complete": "complete",
    "successful": "success",
    "failed": "failure",
    "initialize": "initialize",
    "create": "create",
    "remove": "delete",
    "update": "update",
    "save": "save",
    "load": "load",
    "process": "process",
    "generate": "generate",
    "calculate": "calculate",
    "check": "check",
    "validate": "validate",
    "input": "input",
    "output": "output",
    "result": "result",
    "status": "status",
    "progress": "progress",
    "time": "time",
    "memory": "memory",
    "GPU": "GPU",
    "CPU": "CPU",
    "device": "device",
    "server": "server",
    "client": "client",
    "connections": "connection",
    "network": "network",
    "interface": "interface",
    "API": "API",
    "database": "database",
    "table": "table",
    "field": "field",
    "index": "index",
    "query": "query",
    "sort": "sort",
    "filter": "filter",
    "search": "search",
    "match": "match",
    "replace": "replace",
    "split": "split",
    "merge": "merge",
    "split": "split",
    "combine": "combine",
    "convert": "convert",
    "mapping": "mapping",
    "format": "format",
    "type": "type",
    "class": "class",
    "function": "function",
    "method": "method",
    "variable": "variable",
    "constant": "constant",
    "object": "object",
    "instance": "instance",
    "attribute": "attribute",
    "value": "value",
    "key": "key",
    "list": "list",
    "dictionary": "dictionary",
    "set": "set",
    "array": "array",
    "matrix": "matrix",
    "vector": "vector",
    "scalar": "scalar",
    "dimension": "dimension",
    "shape": "shape",
    "length": "length",
    "width": "width",
    "height": "height",
    "depth": "depth",
    "layers": "layer",
    "level": "level",
    "level": "level",
    "weight": "weight",
    "bias": "bias",
    "activation": "activation",
    "loss": "loss",
    "gradient": "gradient",
    "learning rate": "learning rate",
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "score": "score",
    "score": "score",
    "metric": "metric",
    "evaluation": "evaluation",
    "comparison": "comparison",
    "analysis": "analysis",
    "statistics": "statistics",
    "correlation": "correlation",
    "regression": "regression",
    "classification": "classification",
    "clustering": "clustering",
    "anomaly": "anomaly",
    "normal": "normal",
    "standard": "standard",
    "default": "default",
    "custom": "custom",
    "user": "user",
    "system": "system",
    "application": "application",
    "program": "program",
    "software": "software",
    "hardware": "hardware",
    "platform": "platform",
    "environment": "environment",
    "version": "version",
    "update": "update",
    "upgrade": "upgrade",
    "install": "install",
    "uninstall": "uninstall",
    "deploy": "deploy",
    "release": "release",
    "build": "build",
    "compile": "compile",
    "run": "run",
    "execute": "execute",
    "start": "start",
    "stop": "stop",
    "pause": "pause",
    "continue": "continue",
    "restart": "restart",
    "reset": "reset",
    "clear": "clear",
    "clean": "clean",
    "organize": "organize",
    "sort": "sort",
    
    # Phrases and sentences
    "initialization complete": "initialization complete",
    "start training": "start training",
    "training complete": "training complete", 
    "start testing": "start testing",
    "testing complete": "testing complete",
    "data loading": "data loading",
    "model saving": "model saving",
    "checkpoint": "checkpoint",
    "best model": "best model",
    "current model": "current model",
    "training loss": "training loss",
    "validation loss": "validation loss",
    "test loss": "test loss",
    "learning rate": "learning rate",
    "batch size": "batch size",
    "training epochs": "training epochs",
    "validation epochs": "validation epochs",
    "early stopping": "early stopping",
    "overfitting": "overfitting",
    "underfitting": "underfitting",
    "regularization": "regularization",
    "normalization": "normalization",
    "standardization": "standardization",
    "feature engineering": "feature engineering",
    "feature selection": "feature selection",
    "data preprocessing": "data preprocessing",
    "data cleaning": "data cleaning",
    "data analysis": "data analysis",
    "data visualization": "data visualization",
    "model evaluation": "model evaluation",
    "model selection": "model selection",
    "hyperparameter": "hyperparameter",
    "hyperparameter tuning": "hyperparameter tuning",
    "cross validation": "cross validation",
    "grid search": "grid search",
    "random search": "random search",
    "bayesian optimization": "bayesian optimization",
    "ensemble learning": "ensemble learning",
    "deep learning": "deep learning",
    "machine learning": "machine learning",
    "artificial intelligence": "artificial intelligence",
    "neural network": "neural network",
    "convolutional neural network": "convolutional neural network",
    "recurrent neural network": "recurrent neural network",
    "long short-term memory": "long short-term memory",
    "attention mechanism": "attention mechanism",
    "Transformer": "Transformer",
    "encoder": "encoder",
    "decoder": "decoder",
    "embedding": "embedding",
    "word embedding": "word embedding",
    "positional encoding": "positional encoding",
    "multi-head attention": "multi-head attention",
    "self-attention": "self-attention",
    "residual connection": "residual connection",
    "layer normalization": "layer normalization",
    "batch normalization": "batch normalization",
    "dropout": "dropout",
    "activation function": "activation function",
    "loss function": "loss function",
    "optimizer": "optimizer",
    "gradient descent": "gradient descent",
    "stochastic gradient descent": "stochastic gradient descent",
    "Adam optimizer": "Adam optimizer",
    "learning rate scheduling": "learning rate scheduling",
    "momentum": "momentum",
    "weight decay": "weight decay",
    "gradient clipping": "gradient clipping",
    "batch normalization": "batch normalization",
    "instance normalization": "instance normalization",
    "group normalization": "group normalization",
    
    # Log messages
    "processing": "processing",
    "completed": "completed",
    "failed": "failed",
    "successful": "successful",
    "error": "error",
    "warning": "warning",
    "note": "note",
    "tip": "tip",
    "suggestion": "suggestion",
    "recommend": "recommend",
    "not recommended": "not recommended",
    "skipped": "skipped",
    "ignored": "ignored",
    "not supported": "not supported",
    "not supported yet": "not supported yet",
    "coming soon": "coming soon",
    
    # Time related
    "seconds": "seconds",
    "minutes": "minutes", 
    "hours": "hours",
    "days": "days",
    "weeks": "weeks",
    "months": "months",
    "years": "years",
    "milliseconds": "milliseconds",
    "microseconds": "microseconds",
    "nanoseconds": "nanoseconds",
    
    # Units
    "": "",
    "times": "times",
    "rounds": "rounds",
    "steps": "steps",
    "batches": "batches",
    "samples": "samples",
    "features": "features",
    "dimensions": "dimensions",
    "layers": "layers",
    "nodes": "nodes",
    "connections": "connections",
    
    # Actions
    "add": "add",
    "remove": "remove", 
    "modify": "modify",
    "edit": "edit",
    "copy": "copy",
    "move": "move",
    "rename": "rename",
    "view": "view",
    "display": "display",
    "hide": "hide",
    "open": "open",
    "close": "close",
    "select": "select",
    "cancel": "cancel",
    "confirm": "confirm",
    "submit": "submit",
    "retry": "retry",
    "skip": "skip",
    "continue": "continue",
    "return": "return",
    "exit": "exit",
}

def remove_emojis(text):
    """Remove emoji icons from text"""
    # Emoji patterns
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002700-\U000027BF"  # dingbats
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"  # dingbats
        "\u3030"
        "]+", flags=re.UNICODE)
    
    return emoji_pattern.sub(r'', text)

def translate_chinese_terms(text):
    """Translate Chinese terms to English"""
    result = text
    
    # Sort by length (longest first) to avoid partial replacements
    sorted_terms = sorted(TRANSLATION_MAP.items(), key=lambda x: len(x[0]), reverse=True)
    
    for chinese, english in sorted_terms:
        # Use word boundary for better matching
        pattern = re.compile(re.escape(chinese))
        result = pattern.sub(english, result)
    
    return result

def process_file(file_path):
    """Process a single file to translate Chinese and remove emojis"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove emojis first
        content = remove_emojis(content)
        
        # Translate Chinese terms
        content = translate_chinese_terms(content)
        
        # Write back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Processed: {file_path}")
        return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def should_process_file(file_path):
    """Check if file should be processed"""
    # Skip certain files and directories
    skip_patterns = [
        '*/venv/*',
        '*/__pycache__/*',
        '*.pyc',
        '*/.git/*',
        '*/node_modules/*',
        '*/dist/*',
        '*/build/*',
        '*.egg-info/*'
    ]
    
    file_str = str(file_path)
    for pattern in skip_patterns:
        if fnmatch.fnmatch(file_str, pattern):
            return False
    
    # Only process text files
    text_extensions = ['.py', '.yaml', '.yml', '.md', '.txt', '.sh', '.json', '.cfg', '.ini']
    return any(file_path.suffix.lower() == ext for ext in text_extensions)

def main():
    """Main function to process all files"""
    project_root = Path('.')
    processed_count = 0
    error_count = 0
    
    print("Starting batch translation of Chinese comments and removal of emojis...")
    
    # Find all files to process
    for file_path in project_root.rglob('*'):
        if file_path.is_file() and should_process_file(file_path):
            if process_file(file_path):
                processed_count += 1
            else:
                error_count += 1
    
    print(f"\nTranslation complete!")
    print(f"Processed files: {processed_count}")
    print(f"Errors: {error_count}")

if __name__ == "__main__":
    main()
