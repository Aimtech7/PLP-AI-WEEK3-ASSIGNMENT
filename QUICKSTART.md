# Quick Start Guide

Get the AI Tools & Frameworks Dashboard running in 3 minutes!

## Prerequisites

- Python 3.8 or higher
- pip package manager
- 2GB+ RAM
- Internet connection

## Installation (3 Steps)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Download Language Model

```bash
python -m spacy download en_core_web_sm
```

### Step 3: Train Models

```bash
python setup_models.py
```

This will take 5-10 minutes. Go grab a coffee!

## Run the App

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

## What You Get

### 🪴 Iris Classifier
Try these values:
- Sepal Length: 5.1
- Sepal Width: 3.5
- Petal Length: 1.4
- Petal Width: 0.2

Click "Predict Species" → See it classify as **Setosa**!

### 🔢 MNIST Digit Recognition
- Click the "Draw on Canvas" option
- Draw a digit (0-9) in the black box
- Click "Predict Digit"
- Watch the AI recognize your handwriting!

### 💬 NLP Analysis
- Select "Positive Review" from the dropdown
- Click "Analyze Text"
- See sentiment scores and detected entities!

### 🔍 Explainability
- Go to "Feature Importance" tab
- See which features matter most
- Try LIME for individual explanations
- Generate SHAP analysis for deep insights

## Troubleshooting

### "Model not found"
Run: `python setup_models.py`

### "spaCy model not found"
Run: `python -m spacy download en_core_web_sm`

### "Canvas not drawing"
Install: `pip install streamlit-drawable-canvas`

### Still having issues?
Run the diagnostic: `python test_imports.py`

## Quick Demo Script

1. **Iris Tab**: Predict a flower species
2. **MNIST Tab**: Draw the number "3"
3. **NLP Tab**: Analyze the mixed review
4. **Explainability Tab**: Generate LIME explanation

Total demo time: ~2 minutes

## Next Steps

- Read [FEATURES.md](FEATURES.md) for detailed documentation
- See [DEPLOYMENT.md](DEPLOYMENT.md) to deploy online
- Check [SETUP.md](SETUP.md) for advanced configuration

## One-Line Setup (Unix/Mac)

```bash
chmod +x quickstart.sh && ./quickstart.sh
```

## Alternative: Skip Model Training

If you want to test the interface without waiting:
1. Run `streamlit run app.py`
2. Models will show "not found" messages
3. UI will still work, showing the interface

Train models later when ready!

---

**Need Help?** Contact Team Aimtech7

**Ready to go?** Run: `streamlit run app.py` 🚀
