# Project Structure

## Directory Layout

```
ai-tools-dashboard/
├── app.py                          # Main application entry point
├── iris_classifier.py              # Iris classification module
├── mnist_classifier.py             # MNIST digit recognition module
├── nlp_analyzer.py                 # NLP analysis module
├── explainability.py               # Model explainability module
├── setup_models.py                 # Automated model training script
├── test_imports.py                 # Dependency verification script
├── quickstart.sh                   # Quick setup script (Unix/Mac)
│
├── requirements.txt                # Python dependencies (versioned)
├── requirements-minimal.txt        # Python dependencies (minimal)
├── .gitignore                      # Git ignore rules
│
├── README.md                       # Project overview and quick start
├── SETUP.md                        # Detailed setup instructions
├── FEATURES.md                     # Feature documentation
├── DEPLOYMENT.md                   # Deployment guide
├── PROJECT_STRUCTURE.md            # This file
│
├── .streamlit/
│   └── config.toml                 # Streamlit configuration
│
├── models/
│   ├── iris_decision_tree.pkl      # Trained Iris model (generated)
│   └── mnist_cnn.h5                # Trained MNIST model (generated)
│
├── notebooks/
│   ├── train_iris_model.py         # Iris training script
│   └── train_mnist_model.py        # MNIST training script
│
└── screenshots/                    # Screenshots for documentation
```

---

## File Descriptions

### Core Application Files

#### `app.py`
**Purpose**: Main application entry point
- Sets up Streamlit page configuration
- Creates tab-based navigation
- Imports and renders all modules
- Displays header and footer
- **Lines**: ~80

#### `iris_classifier.py`
**Purpose**: Iris species classification
- Loads scikit-learn Decision Tree model
- Provides input interface for flower measurements
- Displays predictions with confidence scores
- Shows feature importance visualization
- **Lines**: ~150

#### `mnist_classifier.py`
**Purpose**: Handwritten digit recognition
- Loads TensorFlow CNN model
- Implements drawable canvas interface
- Supports image upload
- Preprocesses images to 28x28 grayscale
- Displays confidence distribution
- **Lines**: ~180

#### `nlp_analyzer.py`
**Purpose**: Natural language processing
- Performs sentiment analysis with TextBlob
- Extracts named entities with spaCy
- Provides example reviews
- Visualizes entity distribution
- Shows annotated text
- **Lines**: ~200

#### `explainability.py`
**Purpose**: Model interpretability
- Feature importance for Decision Trees
- LIME explanations for individual predictions
- SHAP analysis for global interpretability
- Interactive visualizations
- **Lines**: ~280

---

### Setup and Configuration Files

#### `setup_models.py`
**Purpose**: Automated model training
- Trains Iris Decision Tree
- Trains MNIST CNN
- Saves models to `models/` directory
- Displays training progress and metrics
- **Lines**: ~100

#### `test_imports.py`
**Purpose**: Dependency verification
- Tests all required imports
- Reports missing packages
- Provides installation guidance
- **Lines**: ~70

#### `quickstart.sh`
**Purpose**: One-command setup (Unix/Mac)
- Installs dependencies
- Downloads spaCy model
- Trains models
- **Lines**: ~35

#### `.streamlit/config.toml`
**Purpose**: Streamlit configuration
- Sets theme colors
- Configures server settings
- **Lines**: ~10

---

### Model Training Scripts

#### `notebooks/train_iris_model.py`
**Purpose**: Train and save Iris model
- Loads Iris dataset
- Trains Decision Tree
- Evaluates accuracy
- Saves to `models/iris_decision_tree.pkl`
- **Lines**: ~45

#### `notebooks/train_mnist_model.py`
**Purpose**: Train and save MNIST model
- Loads MNIST dataset
- Creates CNN architecture
- Trains for 10 epochs
- Saves to `models/mnist_cnn.h5`
- **Lines**: ~75

---

### Documentation Files

#### `README.md`
**Purpose**: Project overview
- Quick start guide
- Feature highlights
- Installation steps
- Team credits

#### `SETUP.md`
**Purpose**: Detailed setup instructions
- Step-by-step installation
- Troubleshooting guide
- System requirements
- Dependency overview

#### `FEATURES.md`
**Purpose**: Feature documentation
- Detailed feature descriptions
- User interaction flows
- Technical details
- Use cases

#### `DEPLOYMENT.md`
**Purpose**: Deployment guide
- Streamlit Cloud deployment
- Docker deployment
- AWS EC2 deployment
- Performance optimization
- Cost estimates

---

## Code Organization

### Module Pattern

Each feature module follows this pattern:

```python
import streamlit as st
import required_libraries

@st.cache_resource
def load_model():
    """Load and cache the model"""
    pass

def render_module_name():
    """Main rendering function"""
    st.header("Module Title")

    # Input section
    col1, col2 = st.columns(2)
    with col1:
        # User inputs
        pass

    with col2:
        # Results display
        pass

    # Additional information
    with st.expander("Info"):
        pass
```

### Naming Conventions

- **Files**: `snake_case.py`
- **Functions**: `snake_case()`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_CASE`
- **Private functions**: `_leading_underscore()`

### Import Order

1. Standard library imports
2. Third-party imports
3. Streamlit imports
4. Local imports

---

## Data Flow

### Iris Classification Flow
```
User Input → iris_classifier.py → Model Prediction → Visualization → Display
     ↓
(sepal/petal measurements) → DecisionTreeClassifier → probabilities → bar chart
```

### MNIST Recognition Flow
```
Canvas/Upload → mnist_classifier.py → Preprocessing → CNN Prediction → Display
     ↓
(image) → grayscale + resize → normalize → TensorFlow model → digit + confidence
```

### NLP Analysis Flow
```
Text Input → nlp_analyzer.py → spaCy + TextBlob → Results → Visualization
     ↓
(review text) → NER + sentiment → entities + scores → tables + charts
```

### Explainability Flow
```
Model Selection → explainability.py → Analysis Tool → Interpretation → Display
     ↓
(Iris model) → LIME/SHAP → feature contributions → interactive plots
```

---

## State Management

### Caching Strategy

- **Models**: Cached with `@st.cache_resource` (never reloads)
- **Data**: Cached with `@st.cache_data` (reloads on change)
- **Session State**: Used for canvas drawings and user inputs

### Performance Considerations

1. Models loaded once on first access
2. Expensive computations cached
3. Visualizations generated on-demand
4. No global state pollution

---

## Extension Points

### Adding New Models

1. Create new module file: `new_model.py`
2. Implement `render_new_model()` function
3. Import in `app.py`
4. Add new tab in main interface

### Adding New Explainability Methods

1. Add new tab in `explainability.py`
2. Implement analysis function
3. Add visualization code
4. Update documentation

### Adding Database Integration

1. Add database connection in separate module
2. Store predictions for analytics
3. Implement user authentication
4. Add data export features

---

## Testing Strategy

### Manual Testing Checklist

- [ ] All models load correctly
- [ ] Iris predictions work
- [ ] MNIST canvas draws correctly
- [ ] MNIST image upload works
- [ ] NLP sentiment analysis accurate
- [ ] NER detects entities
- [ ] LIME explanations generate
- [ ] SHAP analysis completes
- [ ] All visualizations render
- [ ] Error handling works

### Automated Testing (Future)

```python
# tests/test_models.py
def test_iris_model_loads():
    model = load_iris_model()
    assert model is not None

def test_iris_prediction():
    model = load_iris_model()
    result = model.predict([[5.1, 3.5, 1.4, 0.2]])
    assert result[0] in [0, 1, 2]
```

---

## Dependencies Graph

```
app.py
├── iris_classifier.py
│   ├── scikit-learn
│   ├── numpy
│   └── matplotlib
│
├── mnist_classifier.py
│   ├── tensorflow
│   ├── PIL
│   ├── opencv-python
│   └── streamlit-drawable-canvas
│
├── nlp_analyzer.py
│   ├── spacy
│   ├── textblob
│   └── pandas
│
└── explainability.py
    ├── lime
    ├── shap
    └── All above models
```

---

## Version History

### v1.0.0 (Current)
- Initial release
- Iris classifier with Decision Tree
- MNIST digit recognition with CNN
- NLP sentiment and entity analysis
- LIME and SHAP explainability
- Complete documentation

### Future Versions

#### v1.1.0 (Planned)
- [ ] Additional ML models (Random Forest, SVM)
- [ ] Model comparison tools
- [ ] Export predictions to CSV
- [ ] Enhanced visualizations

#### v1.2.0 (Planned)
- [ ] User authentication
- [ ] Database integration
- [ ] API endpoints
- [ ] Mobile optimization

---

## Contributing Guidelines

### Code Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to functions
- Keep functions focused and small

### Git Workflow
```bash
git checkout -b feature/new-feature
# Make changes
git commit -m "Add: new feature description"
git push origin feature/new-feature
# Create pull request
```

### Commit Messages
- **Add**: New features
- **Fix**: Bug fixes
- **Update**: Improvements to existing features
- **Docs**: Documentation changes
- **Refactor**: Code restructuring

---

## Resources

### Documentation Links
- [Streamlit Docs](https://docs.streamlit.io)
- [Scikit-learn Docs](https://scikit-learn.org/stable/)
- [TensorFlow Docs](https://www.tensorflow.org/api_docs)
- [spaCy Docs](https://spacy.io/api)
- [LIME GitHub](https://github.com/marcotcr/lime)
- [SHAP Docs](https://shap.readthedocs.io/)

### Team Resources
- Project Repository: [GitHub Link]
- Issue Tracker: [GitHub Issues]
- Documentation: [Project Wiki]

---

**Maintained by Team Aimtech7**
**Last Updated**: October 2025
