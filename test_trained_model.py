#!/usr/bin/env python3

import torch
from transformers import AutoTokenizer, RobertaForQuestionAnswering
import json

def load_trained_model(model_path="./qa_model_roberta_output/checkpoint-11072"):
    print(f"🔧 Loading trained model from {model_path}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"✅ Tokenizer loaded: {tokenizer.__class__.__name__}")
        
        model = RobertaForQuestionAnswering.from_pretrained(model_path)
        print(f"✅ Model loaded: {model.__class__.__name__}")
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU")
    
    return device

def predict_answer(model, tokenizer, question, context, device):
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        max_length=384,
        truncation=True,
        padding=True
    )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    
    start_idx = torch.argmax(start_logits, dim=1).item()
    end_idx = torch.argmax(end_logits, dim=1).item()
    
    input_ids = inputs["input_ids"][0]
    answer_tokens = input_ids[start_idx:end_idx + 1]
    
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    
    start_confidence = torch.softmax(start_logits, dim=1).max().item()
    end_confidence = torch.softmax(end_logits, dim=1).max().item()
    avg_confidence = (start_confidence + end_confidence) / 2
    
    return answer, start_idx, end_idx, avg_confidence

def format_answer(question, context, answer, confidence):
    print(f"\n🎯 Question: {question}")
    print(f"📖 Context: {context[:200]}...")
    print(f"✅ Answer: {answer}")
    print(f"📊 Confidence: {confidence:.2%}")
    print("-" * 50)

def test_sample_questions(model, tokenizer, device):
    print("\n🧪 Testing model on sample questions...")
    
    test_cases = [
        {
            "question": "What is the capital of France?",
            "context": "Paris is the capital and largest city of France. It is known as the City of Light and is famous for the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral."
        },
        {
            "question": "Who wrote Romeo and Juliet?",
            "context": "William Shakespeare was an English playwright and poet. He wrote many famous plays including Romeo and Juliet, Hamlet, and Macbeth."
        },
        {
            "question": "What is the largest planet in our solar system?",
            "context": "Jupiter is the largest planet in our solar system. It is a gas giant and has a Great Red Spot, which is a massive storm that has been raging for hundreds of years."
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}:")
        answer, start_idx, end_idx, confidence = predict_answer(
            model, tokenizer, test_case["question"], test_case["context"], device
        )
        format_answer(test_case["question"], test_case["context"], answer, confidence)

def interactive_mode(model, tokenizer, device):
    print("\n🎮 Interactive Mode - Test your own questions!")
    print("Type 'quit' to exit")
    print("=" * 50)
    
    while True:
        try:
            question = input("\n❓ Enter your question: ").strip()
            if question.lower() == 'quit':
                break
            
            if not question:
                print("⚠️ Please enter a question.")
                continue
            
            context = input("📖 Enter the context: ").strip()
            if not context:
                print("⚠️ Please enter a context.")
                continue
            
            answer, start_idx, end_idx, confidence = predict_answer(
                model, tokenizer, question, context, device
            )
            format_answer(question, context, answer, confidence)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    print("🚀 Testing Trained RoBERTa Question Answering Model")
    print("=" * 60)
    
    device = setup_device()
    
    model, tokenizer = load_trained_model()
    
    if model is None or tokenizer is None:
        print("❌ Failed to load model. Please check the model path.")
        return
    
    model = model.to(device)
    print(f"✅ Model moved to {device}")
    
    test_sample_questions(model, tokenizer, device)
    
    interactive_mode(model, tokenizer, device)
    
    print("\n✅ Model testing completed!")

if __name__ == "__main__":
    main() 