#!/usr/bin/env python3

import streamlit as st
import torch
from transformers import AutoTokenizer, RobertaForQuestionAnswering
import time

st.set_page_config(
    page_title="Question Answering App",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        font-size: 1.1rem;
    }
    .stButton > button:hover {
        background-color: #0d5aa7;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        model_path = "./qa_model_roberta_output/checkpoint-11072"
        
        with st.spinner("🔧 Loading model and tokenizer..."):
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            model = RobertaForQuestionAnswering.from_pretrained(model_path)
            
            if torch.cuda.is_available():
                device = torch.device("cuda")
                st.success(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
                st.success("✅ Using Apple Silicon GPU (MPS)")
            else:
                device = torch.device("cpu")
                st.warning("⚠️ Using CPU")
            
            model = model.to(device)
            model.eval()
            
            return model, tokenizer, device
            
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None, None

def predict_answer(model, tokenizer, question, context, device):
    try:
        inputs = tokenizer(
            question,
            context,
            return_tensors="pt",
            max_length=384,
            truncation=True,
            padding=True
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
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
        
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
        return None, None, None, None

def get_confidence_color(confidence):
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.6:
        return "confidence-medium"
    else:
        return "confidence-low"

def main():
    st.markdown('<h1 class="main-header">🤖 Question Answering App</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Powered by your trained RoBERTa model</p>', unsafe_allow_html=True)
    
    model, tokenizer, device = load_model()
    
    if model is None:
        st.error("❌ Failed to load model. Please check if the model files exist in ./qa_model_roberta_output/checkpoint-11072/")
        return
    
    with st.sidebar:
        st.header("📊 Model Information")
        st.info(f"**Model:** RoBERTa-base fine-tuned on SQuAD")
        st.info(f"**Device:** {device}")
        st.info(f"**Parameters:** 124M")
        
        st.header("💡 Tips")
        st.markdown("""
        - Make sure your question is clear and specific
        - Provide enough context for the model to find the answer
        - The model works best with factual questions
        - Longer contexts are automatically truncated to 384 tokens
        """)
        
        st.header("📝 Sample Questions")
        sample_questions = [
            "What is the capital of France?",
            "Who wrote Romeo and Juliet?",
            "What is the largest planet in our solar system?",
            "When was the Declaration of Independence signed?",
            "What is the chemical symbol for gold?"
        ]
        
        for i, question in enumerate(sample_questions, 1):
            st.text(f"{i}. {question}")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📖 Context")
        context = st.text_area(
            "Enter the context/passage where the answer can be found:",
            height=200,
            placeholder="Paste or type the context here...",
            help="This is the text that contains the information needed to answer your question."
        )
    
    with col2:
        st.header("❓ Question")
        question = st.text_input(
            "Enter your question:",
            placeholder="What is...?",
            help="Ask a specific question that can be answered from the context."
        )
    
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🔍 Get Answer", type="primary"):
            if not context.strip():
                st.error("❌ Please enter some context.")
            elif not question.strip():
                st.error("❌ Please enter a question.")
            else:
                with st.spinner("🤔 Thinking..."):
                    time.sleep(0.5)
                    
                    answer, start_idx, end_idx, confidence = predict_answer(
                        model, tokenizer, question, context, device
                    )
                    
                    if answer is not None:
                        st.markdown("---")
                        st.header("✅ Answer")
                        
                        confidence_color = get_confidence_color(confidence)
                        st.markdown(f"""
                        <div class="answer-box">
                            <h3>🎯 Answer: {answer}</h3>
                            <p><span class="{confidence_color}">📊 Confidence: {confidence:.1%}</span></p>
                            <p><small>📍 Answer span: tokens {start_idx} to {end_idx}</small></p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("📋 Additional Information"):
                            st.write(f"**Question:** {question}")
                            st.write(f"**Context length:** {len(context)} characters")
                            st.write(f"**Answer length:** {len(answer)} characters")
                            
                            if confidence >= 0.8:
                                st.success("🎉 High confidence answer!")
                            elif confidence >= 0.6:
                                st.warning("⚠️ Medium confidence answer. Consider providing more context.")
                            else:
                                st.error("❌ Low confidence answer. The model may not have found a good answer.")
                    else:
                        st.error("❌ Could not generate an answer. Please try again with different input.")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with ❤️ using Streamlit and your trained RoBERTa model</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 