# AI Tools & Frameworks Dashboard

A comprehensive Streamlit web application showcasing three powerful AI frameworks: Scikit-learn, TensorFlow, and spaCy.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![License](https://img.shields.io/badge/License-Educational-green)

## Quick Start

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python setup_models.py
streamlit run app.py
```

See [QUICKSTART.md](QUICKSTART.md) for detailed instructions.

## Features

### 🪴 Iris Classifier (Scikit-learn)
- Interactive species prediction using Decision Tree
- Input custom flower measurements
- View prediction confidence with visualizations
- Feature importance analysis

### 🔢 MNIST Digit Recognition (TensorFlow)
- Draw digits on an interactive canvas
- Upload handwritten digit images
- CNN-powered prediction with confidence scores
- Real-time image preprocessing

### 💬 NLP Sentiment & Entity Analysis (spaCy + TextBlob)
- Sentiment analysis with polarity and subjectivity scores
- Named Entity Recognition (NER)
- Pre-loaded example reviews
- Visual entity distribution

### 🔍 Model Explainability
- Feature importance visualization
- LIME explanations for individual predictions
- SHAP analysis for global interpretability
- Understand AI decision-making

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download spaCy language model:
```bash
python -m spacy download en_core_web_sm
```

3. Train the models (optional):
```bash
cd notebooks
python train_iris_model.py
python train_mnist_model.py
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Project Structure

```
.
├── app.py                      # Main Streamlit application
├── iris_classifier.py          # Iris classification module
├── mnist_classifier.py         # MNIST digit recognition module
├── nlp_analyzer.py            # NLP sentiment & entity analysis
├── explainability.py          # Model explainability dashboard
├── requirements.txt           # Python dependencies
├── models/
│   ├── iris_decision_tree.pkl # Trained Iris model
│   └── mnist_cnn.h5          # Trained MNIST CNN model
├── notebooks/
│   ├── train_iris_model.py   # Iris model training script
│   └── train_mnist_model.py  # MNIST model training script
└── screenshots/              # Application screenshots
```

## Technologies

- **Streamlit**: Interactive web application framework
- **Scikit-learn**: Classical machine learning (Decision Trees)
- **TensorFlow/Keras**: Deep learning (CNNs)
- **spaCy**: Industrial-strength NLP
- **TextBlob**: Simple sentiment analysis
- **LIME**: Local model explanations
- **SHAP**: SHapley Additive exPlanations
- **Matplotlib/Seaborn**: Data visualization

## Deployment

### Streamlit Cloud

1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Deploy with one click

Ensure all model files are included or retrain models in the cloud environment.

## Documentation

- [QUICKSTART.md](QUICKSTART.md) - Get started in 3 minutes
- [FEATURES.md](FEATURES.md) - Detailed feature documentation
- [SETUP.md](SETUP.md) - Installation and troubleshooting
- [DEPLOYMENT.md](DEPLOYMENT.md) - Deploy to production
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - Code architecture
- [ARCHITECTURE.md](ARCHITECTURE.md) - System design diagrams

## Screenshots

### Dashboard
Beautiful, professional interface with tab-based navigation.

### Iris Classifier
Interactive predictions with confidence visualization.

### MNIST Recognition
Draw digits or upload images for real-time recognition.

### NLP Analysis
Sentiment analysis and named entity recognition.

### Explainability
LIME and SHAP visualizations for model interpretation.

## Project Stats

- **Total Files**: 22
- **Lines of Code**: ~1,200+
- **Documentation Pages**: 7
- **Models Implemented**: 3
- **AI Frameworks**: 5

## Team

**Developed by Team Aimtech7**
AI Tools & Frameworks Assignment 2025

## License

This project is created for educational purposes.
