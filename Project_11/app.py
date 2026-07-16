# -*- coding: utf-8 -*-
"""
Streamlit Web Application: Deep Learning Movie Review Sentiment Hub
"""

import streamlit as st
import pandas as pd
import torch
from transformers import pipeline
from sklearn.metrics import accuracy_score, f1_score

# 1. Page Configuration
st.set_page_config(
    page_title="LLM Sentiment Hub",
    page_icon="🎬",
    layout="wide"
)

# 2. Cache Model Weights to Prevent Reloading on Every Interaction
@st.cache_resource
def load_nlp_pipeline():
    return pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

# Initialize model pipeline quietly
classifier = load_nlp_pipeline()

# 3. Sophisticated Success Banner Header
st.markdown("""
<div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 22px; border-radius: 12px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.15); font-family: 'Segoe UI', sans-serif; margin-bottom: 25px;">
    <h2 style="color: #ffffff; margin: 0; font-weight: 600; letter-spacing: 1px;">🚀 Neural Inference Engine Active</h2>
    <p style="color: #dcdde1; margin: 8px 0 0 0; font-size: 14px;">Production-Ready Sentiment Hub Powered by DistilBERT & Streamlit</p>
</div>
""", unsafe_html=True)

# 4. Data Ingestion Selection
st.markdown("### 📦 Data Ingestion Control")
data_source = st.radio("Choose Data Source:", ["Use Sample Reviews Data", "Upload Custom Movie Reviews CSV"])

if data_source == "Upload Custom Movie Reviews CSV":
    uploaded_file = st.file_uploader("Upload your review dataset (Must include 'Review' and 'Class' columns)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.info("Awaiting file upload. Using default baseline sample dataset in the interim.")
        df = pd.DataFrame({
            'Review': [
                "KGF 2 is an absolute masterpiece! Rocky Bhai's screen presence is unreal, and the background music gives goosebumps.",
                "The action sequences were great, but the pacing felt incredibly rushed in the second half. Editing could be much better.",
                "An overhyped, loud headache of a movie. Too much slow motion and senseless violence with zero substance."
            ],
            'Class': ["POSITIVE", "NEGATIVE", "NEGATIVE"]
        })
else:
    df = pd.DataFrame({
        'Review': [
            "KGF 2 is an absolute masterpiece! Rocky Bhai's screen presence is unreal, and the background music gives goosebumps.",
            "The action sequences were great, but the pacing felt incredibly rushed in the second half. Editing could be much better.",
            "An overhyped, loud headache of a movie. Too much slow motion and senseless violence with zero substance."
        ],
        'Class': ["POSITIVE", "NEGATIVE", "NEGATIVE"]
    })

reviews = df['Review'].tolist()
real_labels = df['Class'].tolist()

# 5. Batch Inference Execution
predicted_labels = classifier(reviews)

# 6. Performance Evaluation Dashboard Calculations via Scikit-Learn (Local Math)
references = [1 if str(label).upper() == "POSITIVE" else 0 for label in real_labels]
predictions = [1 if label['label'] == "POSITIVE" else 0 for label in predicted_labels]

# High-reliability metric calculation (handles edge cases smoothly)
accuracy_result = accuracy_score(references, predictions)
f1_result = f1_score(references, predictions, average='macro', zero_division=0)

# 7. Render Executive Dashboard
dashboard_html = f"""
<div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 16px; padding: 25px; font-family: 'Segoe UI', sans-serif; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.05); margin-bottom: 30px;">
    <h3 style="margin-top: 0; color: #0f172a; font-weight: 600; text-align: center; border-bottom: 2px solid #e2e8f0; padding-bottom: 15px; letter-spacing: 0.5px;">📈 Pipeline Performance Metrics</h3>
    <div style="display: flex; justify-content: space-around; margin-top: 25px; gap: 20px;">
        <div style="background: white; border-radius: 12px; padding: 20px; text-align: center; width: 45%; box-shadow: 0 4px 12px rgba(0,0,0,0.02); border-top: 5px solid #3b82f6;">
            <span style="font-size: 12px; text-transform: uppercase; color: #64748b; font-weight: 600; letter-spacing: 0.5px; display: block; margin-bottom: 5px;">Model Accuracy</span>
            <h1 style="margin: 0; color: #1e293b; font-size: 38px; font-weight: 700;">{accuracy_result * 100:.1f}%</h1>
        </div>
        <div style="background: white; border-radius: 12px; padding: 20px; text-align: center; width: 45%; box-shadow: 0 4px 12px rgba(0,0,0,0.02); border-top: 5px solid #8b5cf6;">
            <span style="font-size: 12px; text-transform: uppercase; color: #64748b; font-weight: 600; letter-spacing: 0.5px; display: block; margin-bottom: 5px;">Macro F1-Score</span>
            <h1 style="margin: 0; color: #1e293b; font-size: 38px; font-weight: 700;">{f1_result:.3f}</h1>
        </div>
    </div>
</div>
"""
st.markdown(dashboard_html, unsafe_html=True)

# 8. Main Visual Cards Container Setup
st.markdown("### 🔍 Deep Learning Inference Matrix")
for i, (review, prediction, label) in enumerate(zip(reviews, predicted_labels, real_labels)):
    pred_text = prediction['label']
    confidence = prediction['score'] * 100
    
    badge_color = "#2ecc71" if pred_text == "POSITIVE" else "#e74c3c"
    border_status = "#2ecc71" if str(pred_text).upper() == str(label).upper() else "#f39c12"
    
    card_html = f"""
    <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-left: 6px solid {border_status}; border-radius: 8px; padding: 18px; margin-bottom: 15px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); font-family: 'Segoe UI', sans-serif;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <span style="font-size: 11px; font-weight: bold; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px;">Review Frame #{i+1}</span>
            <div>
                <span style="background-color: {badge_color}; color: white; padding: 4px 12px; border-radius: 20px; font-size: 11px; font-weight: 600; margin-right: 6px;">PREDICTED: {pred_text}</span>
                <span style="background-color: #f1f5f9; color: #475569; padding: 4px 12px; border-radius: 20px; font-size: 11px; font-weight: 600;">GROUND TRUTH: {label}</span>
            </div>
        </div>
        <p style="color: #1e293b; font-size: 14px; margin: 0 0 12px 0; line-height: 1.6; font-style: italic;">"{review}"</p>
        <div style="font-size: 12px; color: #64748b; text-align: right; border-top: 1px solid #f1f5f9; padding-top: 8px;">
            Mathematical Certainty: <strong style="color: #2563eb; font-size: 13px;">{confidence:.2f}%</strong>
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_html=True)

# 9. Dynamic Interactive Sandbox
st.markdown("### 🧪 Real-Time Testing Sandbox")
user_input = st.text_input("Type your own custom review to challenge the model:", 
                            value="KGF 2 is an amazing movie with powerful action and excellent performance.")

if st.button("⚡ Trigger Neural Evaluation", use_container_width=True):
    live_prediction = classifier(user_input)[0]
    live_sentiment = live_prediction['label']
    live_confidence = live_prediction['score'] * 100
    theme_color = "#10ac84" if live_sentiment == "POSITIVE" else "#ee5253"
    
    playground_ui = f"""
    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; font-family: 'Segoe UI', sans-serif; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-top: 15px;">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <span style="font-size: 12px; color: #64748b; display: block; font-weight: 500; text-transform: uppercase;">Classification Output</span>
                <span style="font-size: 26px; font-weight: 700; color: {theme_color};">{live_sentiment}</span>
            </div>
            <div style="text-align: right;">
                <span style="font-size: 12px; color: #64748b; display: block; font-weight: 500; text-transform: uppercase;">Model Assurance</span>
                <span style="font-size: 22px; font-weight: 700; color: #0f172a;">{live_confidence:.2f}%</span>
            </div>
        </div>
    </div>
    """
    st.markdown(playground_ui, unsafe_html=True)

# 10. Pipeline Footer
st.markdown("""
<div style="text-align: center; margin-top: 25px; font-family: 'Segoe UI', sans-serif;">
    <hr style="border: 0; border-top: 1px dashed #cbd5e1; margin-bottom: 20px;">
    <span style="background: transparent; color: #94a3b8; font-size: 12px; font-weight: 600; letter-spacing: 2.5px; text-transform: uppercase;">✨ End of Streamlit Dashboard Runtime ✨</span>
</div>
""", unsafe_html=True)
