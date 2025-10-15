import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

st.set_page_config(
    page_title="AI Tools & Frameworks Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .footer {
        text-align: center;
        padding: 2rem 0;
        margin-top: 3rem;
        border-top: 2px solid #e5e7eb;
        color: #6b7280;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🤖 AI Tools & Frameworks Dashboard</h1><p>Multi-Model Machine Learning Platform</p></div>', unsafe_allow_html=True)

st.markdown("""
### 📊 Project Overview
This interactive dashboard demonstrates three powerful AI frameworks working together:
- **Scikit-learn**: Classical ML with Decision Trees (Iris Classification)
- **TensorFlow**: Deep Learning with CNNs (MNIST Digit Recognition)
- **spaCy + TextBlob**: Natural Language Processing (Sentiment & Entity Analysis)
""")

tabs = st.tabs(["🪴 Iris Classifier", "🔢 MNIST Digit Recognition", "💬 NLP Sentiment & Entities", "🔍 Explainability Dashboard"])

with tabs[0]:
    from iris_classifier import render_iris_classifier
    render_iris_classifier()

with tabs[1]:
    from mnist_classifier import render_mnist_classifier
    render_mnist_classifier()

with tabs[2]:
    from nlp_analyzer import render_nlp_analyzer
    render_nlp_analyzer()

with tabs[3]:
    from explainability import render_explainability
    render_explainability()

st.markdown('<div class="footer">Developed by <strong>Team Aimtech7</strong> | AI Tools & Frameworks Assignment 2025</div>', unsafe_allow_html=True)
