# AI Tools & Frameworks Dashboard - Features

## Overview

This Streamlit application demonstrates three major AI frameworks in a single, unified dashboard.

---

## 🪴 Iris Classifier (Scikit-learn)

**Purpose**: Classify iris flowers into three species based on petal and sepal measurements.

**Features**:
- ✅ Interactive input sliders for 4 measurements
- ✅ Real-time species prediction (Setosa, Versicolor, Virginica)
- ✅ Confidence probability visualization
- ✅ Feature importance chart
- ✅ Decision Tree model explanation

**User Interaction**:
1. Enter sepal length, sepal width, petal length, petal width
2. Click "Predict Species"
3. View predicted species with confidence scores
4. Explore feature importance

**Technical Details**:
- Model: Decision Tree Classifier
- Dataset: Iris dataset (150 samples, 3 classes)
- Framework: scikit-learn
- Accuracy: ~96-98%

---

## 🔢 MNIST Digit Recognition (TensorFlow)

**Purpose**: Recognize handwritten digits (0-9) using deep learning.

**Features**:
- ✅ Interactive canvas for drawing digits
- ✅ Image upload support
- ✅ Real-time digit prediction
- ✅ Confidence distribution across all digits
- ✅ Preprocessed image visualization

**User Interaction**:
1. Choose input method (draw or upload)
2. Draw a digit on the canvas OR upload an image
3. Click "Predict Digit"
4. View prediction with confidence scores

**Technical Details**:
- Model: Convolutional Neural Network (CNN)
- Architecture: 2 Conv layers + 2 Dense layers
- Dataset: MNIST (70,000 images)
- Accuracy: ~99%

---

## 💬 NLP Sentiment & Entity Analysis (spaCy + TextBlob)

**Purpose**: Analyze text for sentiment and extract named entities.

**Features**:
- ✅ Sentiment analysis (positive/negative/neutral)
- ✅ Polarity and subjectivity scores
- ✅ Named Entity Recognition (NER)
- ✅ Entity type classification
- ✅ Pre-loaded example reviews
- ✅ Word/character/sentence counts

**User Interaction**:
1. Enter text or select an example review
2. Click "Analyze Text"
3. View sentiment classification with scores
4. Explore detected entities (people, organizations, locations, dates, etc.)

**Technical Details**:
- NER Framework: spaCy (en_core_web_sm)
- Sentiment: TextBlob
- Supported entities: PERSON, ORG, GPE, DATE, MONEY, PRODUCT, EVENT, etc.

---

## 🔍 Explainability Dashboard (LIME & SHAP)

**Purpose**: Understand how AI models make predictions.

**Features**:
- ✅ Feature importance visualization
- ✅ LIME (Local Interpretable Model-agnostic Explanations)
- ✅ SHAP (SHapley Additive exPlanations)
- ✅ Individual prediction explanations
- ✅ Global feature importance

**Tabs**:

### 1. Feature Importance
- Shows which features matter most to the model
- Bar chart of importance scores
- Available for Iris model

### 2. LIME Explanation
- Explains individual predictions
- Shows feature contributions (positive/negative)
- Interactive: adjust inputs and see explanations
- Helps understand specific decisions

### 3. SHAP Analysis
- Game theory-based explanations
- Summary plots across all samples
- Mean absolute SHAP values
- Global model interpretation

**User Interaction**:
1. Select model to explain
2. For LIME: adjust input values and generate explanation
3. For SHAP: click to generate global analysis
4. Interpret visualizations

---

## Technical Architecture

### File Structure
```
app.py                  # Main application with tabs
iris_classifier.py      # Iris module
mnist_classifier.py     # MNIST module
nlp_analyzer.py        # NLP module
explainability.py      # XAI module
models/                # Trained model files
notebooks/             # Training scripts
```

### Design Patterns
- **Modular Design**: Each feature in separate file
- **Caching**: Models loaded once and cached
- **Responsive Layout**: Two-column layouts, tabs
- **Error Handling**: Graceful degradation if models missing

### UI/UX Features
- Clean, professional design
- Blue gradient header
- Emoji icons for visual hierarchy
- Expandable info sections
- Interactive visualizations
- Progress indicators
- Team footer

---

## Use Cases

### Education
- Learn about different ML frameworks
- Understand model explainability
- Interactive demonstrations

### Prototyping
- Quick ML model deployment
- Compare different approaches
- Showcase AI capabilities

### Research
- Test model interpretability techniques
- Compare LIME vs SHAP
- Analyze NLP performance

---

## Deployment Options

### Local Development
```bash
streamlit run app.py
```

### Streamlit Cloud
- One-click deployment
- Automatic updates from Git
- Free for public apps

### Docker
```dockerfile
FROM python:3.9
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN python setup_models.py
CMD streamlit run app.py
```

---

## Future Enhancements

Potential additions:
- [ ] More ML models (Random Forest, XGBoost)
- [ ] Image classification beyond digits
- [ ] Real-time video analysis
- [ ] Model comparison tools
- [ ] Export predictions to CSV
- [ ] User authentication
- [ ] API endpoints
- [ ] Mobile optimization

---

## Credits

**Developed by Team Aimtech7**

AI Tools & Frameworks Assignment 2025

**Frameworks Used**:
- Streamlit
- Scikit-learn
- TensorFlow/Keras
- spaCy
- TextBlob
- LIME
- SHAP
- Matplotlib/Seaborn
- NumPy/Pandas
