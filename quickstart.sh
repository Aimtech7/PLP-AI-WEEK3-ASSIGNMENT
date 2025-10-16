#!/bin/bash

echo "============================================================"
echo "AI Tools & Frameworks Dashboard - Quick Start"
echo "============================================================"

echo ""
echo "Step 1: Installing dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Error installing dependencies. Trying minimal requirements..."
    pip install -r requirements-minimal.txt
fi

echo ""
echo "Step 2: Downloading spaCy language model..."
python -m spacy download en_core_web_sm

echo ""
echo "Step 3: Training models..."
python setup_models.py

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "Setup Complete!"
    echo "============================================================"
    echo ""
    echo "Start the app with:"
    echo "  streamlit run app.py"
    echo ""
else
    echo ""
    echo "⚠ Model training failed. You can still run the app, but"
    echo "  some features may not work until models are trained."
    echo ""
    echo "Try running manually:"
    echo "  python setup_models.py"
    echo ""
fi
