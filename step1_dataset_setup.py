#!/usr/bin/env python3

import os
import sys
from datasets import load_dataset
import pandas as pd
import numpy as np

def check_apple_silicon():
    import platform
    import subprocess
    
    print("🔍 Checking system configuration...")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python version: {sys.version}")
    
    if platform.machine() == 'arm64':
        print("✅ Running on Apple Silicon (M1/M2/M3)")
    else:
        print("⚠️  Not on Apple Silicon - some optimizations may not apply")
    
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        print(f"Available RAM: {memory_gb:.1f} GB")
    except ImportError:
        print("psutil not available - can't check memory")

def install_dependencies():
    print("\n📦 Installing dependencies...")
    
    os.system("pip install torch torchvision torchaudio")
    
    os.system("pip install transformers datasets tokenizers")
    os.system("pip install numpy pandas tqdm scikit-learn matplotlib seaborn")
    
    print("✅ Dependencies installed successfully!")

def load_squad_dataset():
    print("\n📚 Loading SQuAD v1.1 dataset...")
    
    dataset = load_dataset("squad")
    
    print(f"✅ Dataset loaded successfully!")
    print(f"Dataset structure: {dataset}")
    
    train_size = len(dataset['train'])
    validation_size = len(dataset['validation'])
    
    print(f"\n📊 Dataset Statistics:")
    print(f"Training examples: {train_size:,}")
    print(f"Validation examples: {validation_size:,}")
    print(f"Total examples: {train_size + validation_size:,}")
    
    return dataset

def explore_sample_data(dataset):
    print("\n🔍 Exploring sample data...")
    
    sample_examples = dataset['train'].select(range(3))
    
    print("\n📝 Sample Training Examples:")
    print("=" * 80)
    
    for i, example in enumerate(sample_examples, 1):
        print(f"\nExample {i}:")
        print(f"Context: {example['context'][:200]}...")
        print(f"Question: {example['question']}")
        print(f"Answer: {example['answers']['text'][0]}")
        print(f"Answer start: {example['answers']['answer_start'][0]}")
        print("-" * 40)
    
    val_sample = dataset['validation'].select(range(2))
    
    print("\n📝 Sample Validation Examples:")
    print("=" * 80)
    
    for i, example in enumerate(val_sample, 1):
        print(f"\nValidation Example {i}:")
        print(f"Context: {example['context'][:200]}...")
        print(f"Question: {example['question']}")
        print(f"Answer: {example['answers']['text'][0]}")
        print(f"Answer start: {example['answers']['answer_start'][0]}")
        print("-" * 40)

def analyze_dataset_features(dataset):
    print("\n📈 Dataset Feature Analysis:")
    print("=" * 80)
    
    train_features = dataset['train'].features
    print(f"Training set features: {list(train_features.keys())}")
    
    train_answers = dataset['train']['answers']
    answer_lengths = [len(ans['text'][0]) for ans in train_answers]
    
    print(f"\nAnswer length statistics:")
    print(f"Mean answer length: {np.mean(answer_lengths):.1f} characters")
    print(f"Median answer length: {np.median(answer_lengths):.1f} characters")
    print(f"Min answer length: {np.min(answer_lengths)} characters")
    print(f"Max answer length: {np.max(answer_lengths)} characters")
    
    question_lengths = [len(q) for q in dataset['train']['question']]
    print(f"\nQuestion length statistics:")
    print(f"Mean question length: {np.mean(question_lengths):.1f} characters")
    print(f"Median question length: {np.median(question_lengths):.1f} characters")
    print(f"Min question length: {np.min(question_lengths)} characters")
    print(f"Max question length: {np.max(question_lengths)} characters")
    
    context_lengths = [len(c) for c in dataset['train']['context']]
    print(f"\nContext length statistics:")
    print(f"Mean context length: {np.mean(context_lengths):.1f} characters")
    print(f"Median context length: {np.median(context_lengths):.1f} characters")
    print(f"Min context length: {np.min(context_lengths)} characters")
    print(f"Max context length: {np.max(context_lengths)} characters")

def main():
    print("🚀 Step 1: Setting up QA System Environment and Loading SQuAD v1.1 Dataset")
    print("=" * 80)
    
    check_apple_silicon()
    
    dataset = load_squad_dataset()
    
    explore_sample_data(dataset)
    
    analyze_dataset_features(dataset)
    
    print("\n✅ Step 1 completed successfully!")
    print("\n📋 Summary:")
    print("- Environment checked for Apple Silicon compatibility")
    print("- SQuAD v1.1 dataset loaded successfully")
    print("- Sample data explored and displayed")
    print("- Dataset statistics and features analyzed")
    print("\n🎯 Ready for Step 2!")

if __name__ == "__main__":
    main() 