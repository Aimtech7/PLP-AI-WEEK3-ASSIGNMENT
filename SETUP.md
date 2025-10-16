# Setup Instructions

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download spaCy Language Model

```bash
python -m spacy download en_core_web_sm
```

### 3. Train the Models

Run the setup script to train both models:

```bash
python setup_models.py
```

This will:
- Train the Iris Decision Tree model (~1 minute)
- Train the MNIST CNN model (~5-10 minutes)
- Save models to the `models/` directory

### 4. Run the Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Manual Model Training

If you prefer to train models individually:

### Iris Model

```bash
cd notebooks
python train_iris_model.py
```

### MNIST Model

```bash
cd notebooks
python train_mnist_model.py
```

## Troubleshooting

### spaCy Model Not Found

If you get a spaCy model error:

```bash
python -m spacy download en_core_web_sm
```

### TensorFlow Issues

For TensorFlow installation issues on Apple Silicon (M1/M2):

```bash
pip install tensorflow-macos
pip install tensorflow-metal
```

### Canvas Drawing Not Working

Ensure `streamlit-drawable-canvas` is installed:

```bash
pip install streamlit-drawable-canvas
```

### LIME/SHAP Errors

Install explainability libraries:

```bash
pip install lime shap
```

## Deployment to Streamlit Cloud

1. Push your code to GitHub (exclude model files if too large)
2. Add `setup_models.py` to your deployment
3. In Streamlit Cloud settings, set the Python version to 3.9+
4. Models will be trained on first run or add them to your repo if small enough

### Option 1: Include Pre-trained Models
- If models are < 100MB, commit them to the repo
- Models will be available immediately

### Option 2: Train on Startup
- Add a startup script in Streamlit Cloud
- Run `python setup_models.py` before starting the app

## System Requirements

- Python 3.8 or higher
- 2GB RAM minimum (4GB+ recommended for MNIST training)
- ~500MB disk space for dependencies
- ~100MB for trained models

## Dependencies Overview

| Package | Purpose |
|---------|---------|
| streamlit | Web application framework |
| scikit-learn | Iris classifier |
| tensorflow | MNIST CNN |
| spacy | NLP entity recognition |
| textblob | Sentiment analysis |
| lime | Model explanations |
| shap | Model interpretability |
| streamlit-drawable-canvas | Draw digits |
| matplotlib, seaborn | Visualizations |

## Performance Tips

1. **First Load**: Models are cached after first load
2. **Canvas Drawing**: Works best on desktop browsers
3. **SHAP Analysis**: Can take 30-60 seconds for computation
4. **Memory**: Close other applications if training MNIST locally

## Support

For issues or questions, contact Team Aimtech7.
