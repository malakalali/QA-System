#!/usr/bin/env python3

import os
import sys
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    RobertaForQuestionAnswering,
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
from tqdm import tqdm

os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"

def check_colab_environment():
    try:
        import google.colab
        print("✅ Running in Google Colab")
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            device = torch.device("cpu")
            print("⚠️ No GPU available, using CPU")
        
        return device
    except ImportError:
        print("⚠️ Not running in Google Colab")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device

def load_and_preprocess_dataset():
    print("🔧 Loading SQuAD dataset...")
    
    dataset = load_dataset("squad")
    print(f"✅ Dataset loaded successfully!")
    print(f"   Training: {len(dataset['train']):,} examples")
    print(f"   Validation: {len(dataset['validation']):,} examples")
    
    return dataset

def load_tokenizer():
    print("🔧 Loading RoBERTa tokenizer...")
    
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    print(f"✅ Tokenizer loaded: {tokenizer.__class__.__name__}")
    
    return tokenizer

def find_answer_positions(context, answer_text, answer_start):
    answer_end = answer_start + len(answer_text)
    
    if context[answer_start:answer_end] != answer_text:
        start_pos = context.find(answer_text)
        if start_pos != -1:
            answer_start = start_pos
            answer_end = start_pos + len(answer_text)
        else:
            return None, None
    
    return answer_start, answer_end

def preprocess_function(examples, tokenizer, max_length=384):
    questions = [q.strip() for q in examples["question"]]
    contexts = [c.strip() for c in examples["context"]]
    
    tokenized_examples = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=max_length,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors=None
    )
    
    offset_mapping = tokenized_examples.pop("offset_mapping")
    
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    
    start_positions = []
    end_positions = []
    
    for i, sample_id in enumerate(sample_mapping):
        answer = examples["answers"][sample_id]
        answer_text = answer["text"][0]
        answer_start = answer["answer_start"][0]
        
        start_char, end_char = find_answer_positions(
            contexts[sample_id], answer_text, answer_start
        )
        
        if start_char is None or end_char is None:
            start_positions.append(0)
            end_positions.append(0)
            continue
        
        sequence_ids = tokenized_examples.sequence_ids(i)
        
        context_start = 0
        context_end = len(sequence_ids) - 1
        while sequence_ids[context_start] != 1:
            context_start += 1
        while sequence_ids[context_end] != 1:
            context_end -= 1
        
        start_position = 0
        end_position = 0
        
        for token_idx, (start, end) in enumerate(offset_mapping[i]):
            if start <= start_char < end:
                start_position = token_idx
            if start < end_char <= end:
                end_position = token_idx
        
        if start_position == 0 or end_position == 0 or start_position > end_position:
            start_position = 0
            end_position = 0
        
        start_positions.append(start_position)
        end_positions.append(end_position)
    
    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions
    
    return tokenized_examples

def create_processed_dataset(tokenizer, dataset, max_length=384):
    print("\n🔧 Creating processed dataset...")
    
    print("Processing training set...")
    train_processed = dataset["train"].map(
        lambda examples: preprocess_function(examples, tokenizer, max_length),
        batched=True,
        batch_size=1000,
        remove_columns=dataset["train"].column_names
    )
    
    print("Processing validation set...")
    val_processed = dataset["validation"].map(
        lambda examples: preprocess_function(examples, tokenizer, max_length),
        batched=True,
        batch_size=1000,
        remove_columns=dataset["validation"].column_names
    )
    
    print(f"✅ Processed dataset created:")
    print(f"   Training: {len(train_processed):,} examples")
    print(f"   Validation: {len(val_processed):,} examples")
    
    return {"train": train_processed, "validation": val_processed}

def load_model_and_tokenizer():
    print("🔧 Loading RoBERTa model and tokenizer...")
    
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    print(f"✅ Tokenizer loaded: {tokenizer.__class__.__name__}")
    
    model = RobertaForQuestionAnswering.from_pretrained("roberta-base")
    print(f"✅ Model loaded: {model.__class__.__name__}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    return model, tokenizer

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    start_preds = predictions[0]
    end_preds = predictions[1]
    
    start_labels = labels[0]
    end_labels = labels[1]
    
    start_preds = np.argmax(start_preds, axis=1)
    end_preds = np.argmax(end_preds, axis=1)
    
    start_accuracy = accuracy_score(start_labels, start_preds)
    end_accuracy = accuracy_score(end_labels, end_preds)
    
    start_f1 = precision_recall_fscore_support(start_labels, start_preds, average='weighted')[2]
    end_f1 = precision_recall_fscore_support(end_labels, end_preds, average='weighted')[2]
    
    exact_matches = np.sum((start_preds == start_labels) & (end_preds == end_labels))
    exact_match_rate = exact_matches / len(start_labels)
    
    return {
        'start_accuracy': start_accuracy,
        'end_accuracy': end_accuracy,
        'start_f1': start_f1,
        'end_f1': end_f1,
        'exact_match_rate': exact_match_rate,
        'avg_f1': (start_f1 + end_f1) / 2
    }

def create_data_collator(tokenizer):
    return DataCollatorWithPadding(tokenizer=tokenizer)

def setup_improved_training_arguments():
    print("🔧 Setting up improved training arguments...")
    
    training_args = TrainingArguments(
        output_dir="./qa_model_roberta_output",
        run_name="qa_roberta_improved_training",
        num_train_epochs=4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_steps=500,
        logging_dir="./logs",
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="avg_f1",
        greater_is_better=True,
        report_to=[],
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        push_to_hub=False,
        gradient_accumulation_steps=2,
        fp16=True,
        optim="adamw_torch",
    )
    
    print("✅ Improved training arguments configured:")
    print(f"   Model: RoBERTa-base (stronger than BERT-base)")
    print(f"   Epochs: {training_args.num_train_epochs}")
    print(f"   Batch size: {training_args.per_device_train_batch_size}")
    print(f"   Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    print(f"   Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"   Learning rate: {training_args.learning_rate}")
    print(f"   Mixed precision: {training_args.fp16}")
    print(f"   Warmup steps: {training_args.warmup_steps}")
    
    return training_args

def estimate_training_time(dataset, training_args):
    total_examples = len(dataset["train"])
    batch_size = training_args.per_device_train_batch_size
    gradient_accumulation_steps = getattr(training_args, 'gradient_accumulation_steps', 1)
    effective_batch_size = batch_size * gradient_accumulation_steps
    
    steps_per_epoch = total_examples // effective_batch_size
    total_steps = steps_per_epoch * training_args.num_train_epochs
    
    print(f"\n⏱️ Training Time Estimate:")
    print(f"   Total examples: {total_examples:,}")
    print(f"   Effective batch size: {effective_batch_size}")
    print(f"   Steps per epoch: ~{steps_per_epoch:,}")
    print(f"   Total steps: ~{total_steps:,}")
    print(f"   Estimated time: ~{total_steps * 1.5 // 60} minutes (with GPU)")
    print("=" * 50)

def create_trainer(model, tokenizer, dataset, training_args):
    print("🔧 Creating Trainer...")
    
    data_collator = create_data_collator(tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    print("✅ Trainer created successfully!")
    return trainer

def train_model(trainer):
    print("\n🚀 Starting improved model training...")
    print("=" * 50)
    
    train_result = trainer.train()
    
    print("\n✅ Training completed!")
    print(f"Training loss: {train_result.training_loss:.4f}")
    
    return train_result

def evaluate_model(trainer):
    print("\n🔍 Evaluating model...")
    
    eval_results = trainer.evaluate()
    
    print("\n📊 Evaluation Results:")
    print("=" * 30)
    for key, value in eval_results.items():
        if key != "eval_loss":
            print(f"{key}: {value:.4f}")
    
    return eval_results

def save_model_and_tokenizer(trainer, tokenizer, save_path="./qa_model_roberta"):
    print(f"\n💾 Saving model to {save_path}...")
    
    trainer.save_model(save_path)
    
    tokenizer.save_pretrained(save_path)
    
    training_info = {
        "model_name": "roberta-base",
        "dataset": "squad_v1.1",
        "max_length": 384,
        "training_args": trainer.args.to_dict(),
        "improvements": {
            "stronger_model": "RoBERTa-base (vs BERT-base)",
            "more_epochs": "4 epochs (vs 3)",
            "lower_learning_rate": "2e-5 (vs 3e-5)",
            "optimized_warmup": "500 steps (vs 1000)"
        }
    }
    
    with open(f"{save_path}/training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)
    
    print("✅ Model and tokenizer saved successfully!")

def test_model_on_sample(model, tokenizer, dataset):
    print("\n🧪 Testing model on sample questions...")
    
    sample_examples = dataset["validation"].select(range(3))
    
    model.eval()
    
    for i, example in enumerate(sample_examples):
        input_ids = torch.tensor(example["input_ids"]).unsqueeze(0)
        attention_mask = torch.tensor(example["attention_mask"]).unsqueeze(0)
        
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        start_pred = torch.argmax(outputs.start_logits, dim=1).item()
        end_pred = torch.argmax(outputs.end_logits, dim=1).item()
        
        start_label = example["start_positions"]
        end_label = example["end_positions"]
        
        decoded_input = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        print(f"\nSample {i+1}:")
        print(f"Input: {decoded_input[:100]}...")
        print(f"Predicted start: {start_pred}, Actual start: {start_label}")
        print(f"Predicted end: {end_pred}, Actual end: {end_label}")
        print(f"Correct: {start_pred == start_label and end_pred == end_label}")

def main():
    print("🚀 Step 3: Improved Model Training for Question Answering (Google Colab)")
    print("=" * 80)
    
    device = check_colab_environment()
    
    print("🔧 Creating processed dataset from scratch...")
    raw_dataset = load_and_preprocess_dataset()
    
    tokenizer = load_tokenizer()
    
    dataset = create_processed_dataset(tokenizer, raw_dataset)
    
    model, tokenizer = load_model_and_tokenizer()
    
    model = model.to(device)
    print(f"✅ Model moved to {device}")
    
    training_args = setup_improved_training_arguments()
    
    estimate_training_time(dataset, training_args)
    
    trainer = create_trainer(model, tokenizer, dataset, training_args)
    
    train_result = train_model(trainer)
    
    eval_results = evaluate_model(trainer)
    
    save_model_and_tokenizer(trainer, tokenizer)
    
    test_model_on_sample(model, tokenizer, dataset)
    
    print("\n✅ Improved Step 3 completed successfully!")
    print("\n📋 Summary:")
    print("- RoBERTa-base model fine-tuned on SQuAD v1.1")
    print("- Improved training settings applied")
    print("- Training completed with evaluation metrics")
    print("- Model saved to ./qa_model_roberta/")
    print("- Sample predictions tested")
    print("\n🎯 Ready for Step 4 (Model Inference) with improved model!")

if __name__ == "__main__":
    main() 