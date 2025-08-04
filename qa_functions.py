#!/usr/bin/env python3

import torch
from transformers import AutoTokenizer, RobertaForQuestionAnswering

class QAModel:
    
    def __init__(self, model_path="./qa_model_roberta_output/checkpoint-11072"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = None
        self._load_model()
    
    def _load_model(self):
        try:
            print(f"🔧 Loading model from {self.model_path}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            print(f"✅ Tokenizer loaded: {self.tokenizer.__class__.__name__}")
            
            self.model = RobertaForQuestionAnswering.from_pretrained(self.model_path)
            print(f"✅ Model loaded: {self.model.__class__.__name__}")
            
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print("✅ Using Apple Silicon GPU (MPS)")
            else:
                self.device = torch.device("cpu")
                print("⚠️ Using CPU")
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print("✅ Model ready for inference!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def answer_question(self, question, context):
        try:
            inputs = self.tokenizer(
                question,
                context,
                return_tensors="pt",
                max_length=384,
                truncation=True,
                padding=True
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            
            start_idx = torch.argmax(start_logits, dim=1).item()
            end_idx = torch.argmax(end_logits, dim=1).item()
            
            input_ids = inputs["input_ids"][0]
            answer_tokens = input_ids[start_idx:end_idx + 1]
            
            answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
            
            start_confidence = torch.softmax(start_logits, dim=1).max().item()
            end_confidence = torch.softmax(end_logits, dim=1).max().item()
            avg_confidence = (start_confidence + end_confidence) / 2
            
            if avg_confidence >= 0.8:
                confidence_level = "high"
            elif avg_confidence >= 0.6:
                confidence_level = "medium"
            else:
                confidence_level = "low"
            
            return {
                "answer": answer,
                "confidence": avg_confidence,
                "confidence_level": confidence_level,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_confidence": start_confidence,
                "end_confidence": end_confidence,
                "question": question,
                "context_length": len(context),
                "answer_length": len(answer)
            }
            
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            return {
                "answer": None,
                "confidence": 0.0,
                "confidence_level": "error",
                "error": str(e)
            }

def answer_question(question, context, model_path="./qa_model_roberta_output/checkpoint-11072"):
    if not hasattr(answer_question, '_qa_model'):
        answer_question._qa_model = QAModel(model_path)
    
    result = answer_question._qa_model.answer_question(question, context)
    
    return result["answer"]

if __name__ == "__main__":
    qa_model = QAModel()
    
    test_cases = [
        {
            "question": "What is the capital of France?",
            "context": "Paris is the capital and largest city of France. It is known as the City of Light and is famous for the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral."
        },
        {
            "question": "Who wrote Romeo and Juliet?",
            "context": "William Shakespeare was an English playwright and poet. He wrote many famous plays including Romeo and Juliet, Hamlet, and Macbeth."
        }
    ]
    
    print("\n🧪 Testing the QA model:")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}:")
        print(f"Question: {test_case['question']}")
        print(f"Context: {test_case['context'][:100]}...")
        
        result = qa_model.answer_question(test_case['question'], test_case['context'])
        
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']:.1%} ({result['confidence_level']})")
        print("-" * 30) 