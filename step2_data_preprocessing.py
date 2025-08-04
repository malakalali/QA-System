#!/usr/bin/env python3

import os
import sys
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from tqdm import tqdm

def load_tokenizer_and_dataset():
    print("🔧 Loading tokenizer and dataset...")
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    print(f"✅ Tokenizer loaded: {tokenizer.__class__.__name__}")
    
    dataset = load_dataset("squad")
    print(f"✅ Dataset loaded: {len(dataset['train']):,} training, {len(dataset['validation']):,} validation examples")
    
    return tokenizer, dataset

def find_answer_positions(context, answer_text, answer_start):
    start_positions = []
    end_positions = []
    
    context_lower = context.lower()
    answer_lower = answer_text.lower()
    
    start_idx = 0
    while True:
        pos = context_lower.find(answer_lower, start_idx)
        if pos == -1:
            break
        start_positions.append(pos)
        end_positions.append(pos + len(answer_text))
        start_idx = pos + 1
    
    if not start_positions:
        start_positions = [answer_start]
        end_positions = [answer_start + len(answer_text)]
    
    return start_positions, end_positions

def convert_char_positions_to_token_positions(tokenized_example, char_start, char_end):
    offset_mapping = tokenized_example["offset_mapping"][0]
    
    token_start = None
    token_end = None
    
    for token_idx, (start_char, end_char) in enumerate(offset_mapping):
        if start_char == 0 and end_char == 0:
            continue
            
        if start_char <= char_start < end_char:
            token_start = token_idx
        if start_char < char_end <= end_char:
            token_end = token_idx
    
    if token_start is None:
        token_start = 0
    if token_end is None:
        token_end = len(offset_mapping) - 1
    
    return token_start, token_end

def preprocess_function(examples, tokenizer, max_length=384):
    questions = examples["question"]
    contexts = examples["context"]
    answers = examples["answers"]
    
    input_ids = []
    attention_masks = []
    start_positions = []
    end_positions = []
    
    for i, (question, context, answer) in enumerate(zip(questions, contexts, answers)):
        answer_text = answer["text"][0]
        answer_start = answer["answer_start"][0]
        
        start_positions_char, end_positions_char = find_answer_positions(
            context, answer_text, answer_start
        )
        
        tokenized = tokenizer(
            question,
            context,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        
        token_start, token_end = convert_char_positions_to_token_positions(
            tokenized, start_positions_char[0], end_positions_char[0]
        )
        
        token_start = max(0, min(token_start, max_length - 1))
        token_end = max(0, min(token_end, max_length - 1))
        
        input_ids.append(tokenized["input_ids"][0])
        attention_masks.append(tokenized["attention_mask"][0])
        start_positions.append(token_start)
        end_positions.append(token_end)
    
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "start_positions": torch.tensor(start_positions),
        "end_positions": torch.tensor(end_positions)
    }

def process_dataset_split(dataset_split, tokenizer, max_length=384, batch_size=1000):
    print(f"🔄 Processing {len(dataset_split):,} examples...")
    
    processed_examples = []
    
    for i in tqdm(range(0, len(dataset_split), batch_size), desc="Processing batches"):
        batch = dataset_split.select(range(i, min(i + batch_size, len(dataset_split))))
        
        processed_batch = preprocess_function(batch, tokenizer, max_length)
        
        for j in range(len(batch)):
            processed_examples.append({
                "input_ids": processed_batch["input_ids"][j].tolist(),
                "attention_mask": processed_batch["attention_mask"][j].tolist(),
                "start_positions": processed_batch["start_positions"][j].item(),
                "end_positions": processed_batch["end_positions"][j].item()
            })
    
    return processed_examples

def analyze_tokenization_results(dataset, tokenizer):
    print("\n📊 Tokenization Analysis:")
    print("=" * 50)
    
    sample_examples = dataset['train'].select(range(5))
    
    for i, example in enumerate(sample_examples):
        question = example['question']
        context = example['context']
        answer = example['answers']['text'][0]
        
        tokenized = tokenizer(
            question,
            context,
            truncation=True,
            padding="max_length",
            max_length=384,
            return_tensors="pt"
        )
        
        decoded = tokenizer.decode(tokenized['input_ids'][0], skip_special_tokens=True)
        
        print(f"\nExample {i+1}:")
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print(f"Input length: {len(tokenized['input_ids'][0])} tokens")
        print(f"First 100 chars of decoded: {decoded[:100]}...")

def create_processed_dataset(tokenizer, dataset, max_length=384):
    print("\n🔧 Creating processed dataset...")
    
    print("Processing training set...")
    train_processed = process_dataset_split(
        dataset['train'], tokenizer, max_length, batch_size=1000
    )
    
    print("Processing validation set...")
    val_processed = process_dataset_split(
        dataset['validation'], tokenizer, max_length, batch_size=1000
    )
    
    from datasets import Dataset, DatasetDict
    
    processed_dataset = DatasetDict({
        'train': Dataset.from_list(train_processed),
        'validation': Dataset.from_list(val_processed)
    })
    
    print(f"✅ Processed dataset created:")
    print(f"   Training: {len(processed_dataset['train']):,} examples")
    print(f"   Validation: {len(processed_dataset['validation']):,} examples")
    
    return processed_dataset

def verify_dataset_structure(processed_dataset):
    print("\n🔍 Verifying dataset structure...")
    
    train_example = processed_dataset['train'][0]
    print("Training example structure:")
    for key, value in train_example.items():
        if isinstance(value, list):
            print(f"  {key}: list of length {len(value)}")
        else:
            print(f"  {key}: {type(value).__name__} = {value}")
    
    val_example = processed_dataset['validation'][0]
    print("\nValidation example structure:")
    for key, value in val_example.items():
        if isinstance(value, list):
            print(f"  {key}: list of length {len(value)}")
        else:
            print(f"  {key}: {type(value).__name__} = {value}")
    
    assert all(key in processed_dataset['train'][0] for key in ['input_ids', 'attention_mask', 'start_positions', 'end_positions']), "Missing required fields"
    print("✅ Dataset structure verified!")

def save_processed_dataset(processed_dataset, save_path="processed_squad_dataset"):
    print(f"\n💾 Saving processed dataset to {save_path}...")
    processed_dataset.save_to_disk(save_path)
    print("✅ Dataset saved successfully!")

def main():
    print("🚀 Step 2: Data Preprocessing and Tokenization for SQuAD v1.1")
    print("=" * 80)
    
    tokenizer, dataset = load_tokenizer_and_dataset()
    
    analyze_tokenization_results(dataset, tokenizer)
    
    processed_dataset = create_processed_dataset(tokenizer, dataset, max_length=384)
    
    verify_dataset_structure(processed_dataset)
    
    save_processed_dataset(processed_dataset)
    
    print("\n✅ Step 2 completed successfully!")
    print("\n📋 Summary:")
    print("- BERT tokenizer loaded and configured")
    print("- Dataset tokenized with max_length=384")
    print("- Answer positions mapped to token indices")
    print("- Processed dataset created with required fields")
    print("- Dataset saved for Step 3")
    print("\n🎯 Ready for Step 3 (Model Training)!")

if __name__ == "__main__":
    main() 