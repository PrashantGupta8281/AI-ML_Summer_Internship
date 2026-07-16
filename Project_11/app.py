# -*- coding: utf-8 -*-
"""
Analyzing_KGF_2_movie_review_with_LLMs_GenAIs.ipynb
Sophisticated & Visual-Enhanced Production Version
"""

# ==========================================
# 1. DEPENDENCY INSTALLATION & INITIALIZATION
# ==========================================
#%%capture
#!pip install transformers evaluate sentencepiece sacremoses pandas torch

import transformers
import pandas as pd
import torch
from transformers import pipeline
import evaluate
from IPython.display import display, HTML

# Suppress raw installation text and render a high-end success banner
display(HTML("""
<div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 22px; border-radius: 12px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.15); font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin-bottom: 25px;">
    <h2 style="color: #ffffff; margin: 0; font-weight: 600; letter-spacing: 1px;">🚀 Execution Environment Initialized</h2>
    <p style="color: #dcdde1; margin: 8px 0 0 0; font-size: 14px;">Hugging Face Transformers Engine Active — Engine Version: <span style="color: #00cbc6; font-weight: bold;">""" + transformers.__version__ + """</span></p>
</div>
"""))

# ==========================================
# 2. DATA INGESTION & PIPELINE CONFIGURATION
# ==========================================
display(HTML("""
<div style="border-left: 5px solid #2a5298; padding-left: 15px; margin: 20px 0 10px 0;">
    <h3 style="color: #2a5298; margin: 0; font-family: 'Segoe UI', sans-serif; font-weight: 600;">📦 Data Ingestion & Model Loading</h3>
    <p style="color: #7f8c8d; margin: 4px 0 0 0; font-size: 13px;">Spinning up DistilBERT base architecture for optimized token classification...</p>
</div>
"""))

# Initialize the fine-tuned sentiment analysis pipeline
classifier = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

# --- DATA GENERATION ACCORDING TO DATASET STRUCTURING ---
# Note: For your live environment, uncomment the line below to ingest from your local path:
# df = pd.read_csv("/content/movie_reviews.csv", delimiter=",")
# -------------------------------------------------------
data = {
    'Review': [
        "KGF 2 is an absolute masterpiece! Rocky Bhai's screen presence is unreal, and the background music gives goosebumps.",
        "The action sequences were great, but the pacing felt incredibly rushed in the second half. Editing could be much better.",
        "An overhyped, loud headache of a movie. Too much slow motion and senseless violence with zero substance."
    ],
    'Class': ["POSITIVE", "NEGATIVE", "NEGATIVE"]
}
df = pd.DataFrame(data)

reviews = df['Review'].tolist()
real_labels = df['Class'].tolist()


# ==========================================
# 3. BATCH INFERENCE & VISUAL DATA GRID
# ==========================================
predicted_labels = classifier(reviews)

display(HTML("<h3 style='color: #2c3e50; font-family: \"Segoe UI\", sans-serif; margin: 25px 0 15px 0; font-weight: 600;'>🔍 Deep Learning Inference Matrix</h3>"))

for i, (review, prediction, label) in enumerate(zip(reviews, predicted_labels, real_labels)):
    pred_text = prediction['label']
    confidence = prediction['score'] * 100
    
    # Dynamic accent assignment depending on verification status
    badge_color = "#2ecc71" if pred_text == "POSITIVE" else "#e74c3c"
    border_status = "#2ecc71" if pred_text == label else "#f39c12"
    
    card_html = f"""
    <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-left: 6px solid {border_status}; border-radius: 8px; padding: 18px; margin-bottom: 15px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); font-family: 'Segoe UI', sans-serif;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <span style="font-size: 11px; font-weight: bold; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px;">Review Analysis Frame #{i+1}</span>
            <div>
                <span style="background-color: {badge_color}; color: white; padding: 4px 12px; border-radius: 20px; font-size: 11px; font-weight: 600; margin-right: 6px; letter-spacing: 0.3px;">PREDICTED: {pred_text}</span>
                <span style="background-color: #f1f5f9; color: #475569; padding: 4px 12px; border-radius: 20px; font-size: 11px; font-weight: 600; letter-spacing: 0.3px;">GROUND TRUTH: {label}</span>
            </div>
        </div>
        <p style="color: #1e293b; font-size: 14px; margin: 0 0 12px 0; line-height: 1.6; font-style: italic;">"{review}"</p>
        <div style="font-size: 12px; color: #64748b; text-align: right; border-top: 1px solid #f1f5f9; padding-top: 8px;">
            Mathematical Certainty: <strong style="color: #2563eb; font-size: 13px;">{confidence:.2f}%</strong>
        </div>
    </div>
    """
    display(HTML(card_html))


# ==========================================
# 4. EXECUTIVE PERFORMANCE METRICS DASHBOARD
# ==========================================
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

references = [1 if label == "POSITIVE" else 0 for label in real_labels]
predictions = [1 if label['label'] == "POSITIVE" else 0 for label in predicted_labels]

accuracy_result = accuracy.compute(references=references, predictions=predictions)['accuracy']
f1_result = f1.compute(references=references, predictions=predictions)['f1']

dashboard_html = f"""
<div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 16px; padding: 25px; font-family: 'Segoe UI', sans-serif; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.05); max-width: 700px; margin: 30px auto;">
    <h3 style="margin-top: 0; color: #0f172a; font-weight: 600; text-align: center; border-bottom: 2px solid #e2e8f0; padding-bottom: 15px; letter-spacing: 0.5px;">📈 Pipeline Performance Dashboard</h3>
    
    <div style="display: flex; justify-content: space-around; margin-top: 25px; gap: 20px;">
        <!-- Accuracy Widget -->
        <div style="background: white; border-radius: 12px; padding: 20px; text-align: center; width: 45%; box-shadow: 0 4px 12px rgba(0,0,0,0.02); border-top: 5px solid #3b82f6;">
            <span style="font-size: 12px; text-transform: uppercase; color: #64748b; font-weight: 600; letter-spacing: 0.5px; display: block; margin-bottom: 5px;">Model Accuracy</span>
            <h1 style="margin: 0; color: #1e293b; font-size: 38px; font-weight: 700;">{accuracy_result * 100:.1f}%</h1>
        </div>
        
        <!-- F1-Score Widget -->
        <div style="background: white; border-radius: 12px; padding: 20px; text-align: center; width: 45%; box-shadow: 0 4px 12px rgba(0,0,0,0.02); border-top: 5px solid #8b5cf6;">
            <span style="font-size: 12px; text-transform: uppercase; color: #64748b; font-weight: 600; letter-spacing: 0.5px; display: block; margin-bottom: 5px;">Macro F1-Score</span>
            <h1 style="margin: 0; color: #1e293b; font-size: 38px; font-weight: 700;">{f1_result:.3f}</h1>
        </div>
    </div>
</div>
"""
display(HTML(dashboard_html))


# ==========================================
# 5. LIVE PLAYGROUND / CUSTOM PREDICTION UI
# ==========================================
sample_review = "KGF 2 is an amazing movie with powerful action and excellent performance."

# Generate real-time inference on targeted runtime string
sentiment_result = classifier(sample_review)
sentiment = sentiment_result[0]['label']
confidence_score = sentiment_result[0]['score'] * 100
theme_color = "#10ac84" if sentiment == "POSITIVE" else "#ee5253"

playground_ui = f"""
<div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 25px; font-family: 'Segoe UI', sans-serif; max-width: 700px; margin: 30px auto; box-shadow: 0 20px 25px -5px rgba(0,0,0,0.05), 0 10px 10px -5px rgba(0,0,0,0.04);">
    <div style="font-size: 11px; font-weight: bold; color: #64748b; text-transform: uppercase; margin-bottom: 6px; letter-spacing: 1px;">Live Demonstration Sandbox</div>
    <h4 style="margin: 0 0 18px 0; color: #0f172a; font-size: 18px; font-weight: 600;">🧪 Target String Playground</h4>
    
    <div style="background: #f8fafc; border-radius: 8px; padding: 16px; margin-bottom: 22px; border-left: 4px solid #3b82f6;">
        <span style="font-size: 11px; font-weight: bold; color: #64748b; display: block; margin-bottom: 6px; text-transform: uppercase;">Active Testing Value:</span>
        <span style="color: #1e293b; font-size: 14px; font-style: italic; font-weight: 500;">"{sample_review}"</span>
    </div>
    
    <!-- Beautiful Creative Call-to-Action Component -->
    <div style="text-align: center; margin-bottom: 25px;">
        <button style="background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); color: white; border: none; padding: 12px 35px; font-size: 13px; font-weight: 600; border-radius: 30px; cursor: pointer; box-shadow: 0 10px 15px -3px rgba(124, 58, 237, 0.3); transition: transform 0.2s ease, box-shadow 0.2s ease; letter-spacing: 0.5px;">
            ⚡ Trigger Neural Evaluation
        </button>
    </div>
    
    <!-- Output Card Metrics Structuring -->
    <div style="border-top: 1px solid #e2e8f0; padding-top: 20px; display: flex; align-items: center; justify-content: space-between;">
        <div>
            <span style="font-size: 12px; color: #64748b; display: block; font-weight: 500; text-transform: uppercase; margin-bottom: 2px;">Classification</span>
            <span style="font-size: 24px; font-weight: 700; color: {theme_color}; letter-spacing: 0.5px;">{sentiment}</span>
        </div>
        <div style="text-align: right;">
            <span style="font-size: 12px; color: #64748b; display: block; font-weight: 500; text-transform: uppercase; margin-bottom: 2px;">Model Assurance</span>
            <span style="font-size: 20px; font-weight: 700; color: #0f172a;">{confidence_score:.2f}%</span>
        </div>
    </div>
</div>
"""
display(HTML(playground_ui))

# ==========================================
# 6. PIPELINE FINALE
# ==========================================
display(HTML("""
<div style="text-align: center; margin: 40px 0 20px 0; font-family: 'Segoe UI', sans-serif;">
    <hr style="border: 0; border-top: 1px dashed #cbd5e1; margin-bottom: 20px;">
    <span style="background: #ffffff; padding: 0 15px; color: #94a3b8; font-size: 12px; font-weight: 600; letter-spacing: 2.5px; text-transform: uppercase;">✨ End of Analysis Pipeline ✨</span>
</div>
"""))
