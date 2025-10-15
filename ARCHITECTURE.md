# System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT WEB APP                        │
│                        (app.py)                             │
└─────────────────────────────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Tab 1      │    │   Tab 2      │    │   Tab 3      │
│   Iris       │    │   MNIST      │    │   NLP        │
│ Classifier   │    │  Digit       │    │  Analyzer    │
└──────────────┘    └──────────────┘    └──────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Scikit-learn │    │ TensorFlow   │    │ spaCy +      │
│ Decision     │    │ CNN Model    │    │ TextBlob     │
│ Tree         │    │ (MNIST)      │    │              │
└──────────────┘    └──────────────┘    └──────────────┘
        │                                        │
        └────────────────┬───────────────────────┘
                        ▼
            ┌──────────────────────┐
            │   Tab 4              │
            │   Explainability     │
            │   (LIME + SHAP)      │
            └──────────────────────┘
```

## Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       USER INTERFACE                         │
├─────────────────────────────────────────────────────────────┤
│  Navigation Tabs  │  Input Controls  │  Visualization       │
├─────────────────────────────────────────────────────────────┤
│                    APPLICATION LAYER                         │
├────────────┬────────────┬────────────┬────────────┬─────────┤
│   Iris     │   MNIST    │    NLP     │  Explain   │  Utils  │
│  Module    │  Module    │   Module   │  Module    │         │
└────────────┴────────────┴────────────┴────────────┴─────────┘
       │            │            │            │
       ▼            ▼            ▼            ▼
┌─────────────────────────────────────────────────────────────┐
│                       MODEL LAYER                            │
├────────────┬────────────┬────────────┬────────────┬─────────┤
│   Sklearn  │  TensorFlow│   spaCy    │    LIME    │  SHAP   │
│   Models   │   Models   │   Models   │  Explainer │ Explainer│
└────────────┴────────────┴────────────┴────────────┴─────────┘
```

## Data Flow Diagram

### Iris Classification

```
User Input
    │
    ├─► Sepal Length ─┐
    ├─► Sepal Width  ─┤
    ├─► Petal Length ─┼─► [4 features] ─► Decision Tree Model
    └─► Petal Width  ─┘                          │
                                                 ▼
                                    ┌────────────────────┐
                                    │  Predict Species   │
                                    │  Calculate Probs   │
                                    └────────────────────┘
                                             │
                        ┌────────────────────┼────────────────┐
                        ▼                    ▼                ▼
                  Prediction          Confidence        Feature
                  (Species)           (Bar Chart)       Importance
```

### MNIST Digit Recognition

```
User Input
    │
    ├─► Draw Canvas ─┐
    │                ├─► [Image Data] ─► Preprocess
    └─► Upload File ─┘                       │
                                             ▼
                                  ┌──────────────────┐
                                  │  Resize 28×28    │
                                  │  Grayscale       │
                                  │  Normalize       │
                                  └──────────────────┘
                                             │
                                             ▼
                                    CNN Model (TensorFlow)
                                             │
                        ┌────────────────────┼──────────────┐
                        ▼                    ▼              ▼
                  Digit (0-9)         Confidence      Preprocessed
                                    Distribution      Image Display
```

### NLP Analysis

```
Text Input
    │
    ├─► Custom Text ──┐
    │                 ├─► [Raw Text]
    └─► Example ──────┘       │
                              ├──────────┬──────────┐
                              ▼          ▼          ▼
                        TextBlob     spaCy      Stats
                              │          │          │
                              ▼          ▼          ▼
                        Sentiment    Named      Word/Char
                        Analysis    Entities    Counts
                              │          │          │
                    ┌─────────┼──────────┼──────────┘
                    ▼         ▼          ▼
            Polarity    Entity List   Quick
            Subjectivity  + Types     Stats
```

## Module Interaction

```
app.py (Main)
    │
    ├─► imports iris_classifier.py
    │       │
    │       ├─► loads models/iris_decision_tree.pkl
    │       └─► uses sklearn, matplotlib
    │
    ├─► imports mnist_classifier.py
    │       │
    │       ├─► loads models/mnist_cnn.h5
    │       └─► uses tensorflow, PIL, canvas
    │
    ├─► imports nlp_analyzer.py
    │       │
    │       └─► uses spacy, textblob
    │
    └─► imports explainability.py
            │
            ├─► uses LIME
            ├─► uses SHAP
            └─► references iris_classifier + mnist_classifier
```

## Caching Strategy

```
┌─────────────────────────────────────────────────────────┐
│                   First Request                         │
├─────────────────────────────────────────────────────────┤
│  Load Model ─► Cache with @st.cache_resource            │
│  (Slow)        (Stored in memory)                       │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                 Subsequent Requests                     │
├─────────────────────────────────────────────────────────┤
│  Retrieve from Cache ─► Instant Access                  │
│  (Fast)                 (No reload needed)              │
└─────────────────────────────────────────────────────────┘
```

## Training Pipeline

```
setup_models.py
    │
    ├─► Train Iris Model
    │       │
    │       ├─► Load iris dataset
    │       ├─► Split train/test
    │       ├─► Train Decision Tree
    │       ├─► Evaluate accuracy
    │       └─► Save to models/iris_decision_tree.pkl
    │
    └─► Train MNIST Model
            │
            ├─► Load MNIST dataset
            ├─► Preprocess images
            ├─► Build CNN architecture
            ├─► Train (10 epochs)
            ├─► Evaluate accuracy
            └─► Save to models/mnist_cnn.h5
```

## Deployment Flow

```
┌──────────────┐
│ Local Dev    │
└──────┬───────┘
       │
       ├─► git push
       │
       ▼
┌──────────────┐
│ GitHub Repo  │
└──────┬───────┘
       │
       ├─► Connect
       │
       ▼
┌────────────────┐
│ Streamlit      │
│ Cloud          │
└────────┬───────┘
       │
       ├─► Deploy
       │
       ▼
┌────────────────┐
│ Live App       │
│ (Public URL)   │
└────────────────┘
```

## Technology Stack Layers

```
┌─────────────────────────────────────────────────────────┐
│                  PRESENTATION LAYER                     │
├─────────────────────────────────────────────────────────┤
│  Streamlit │ HTML/CSS │ JavaScript (Canvas)             │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                   BUSINESS LOGIC                        │
├─────────────────────────────────────────────────────────┤
│  Python │ NumPy │ Pandas │ Matplotlib │ Seaborn         │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    MODEL LAYER                          │
├─────────────────────────────────────────────────────────┤
│  Scikit-learn │ TensorFlow │ spaCy │ TextBlob           │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                 EXPLAINABILITY LAYER                    │
├─────────────────────────────────────────────────────────┤
│  LIME │ SHAP │ Feature Importance                       │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    DATA LAYER                           │
├─────────────────────────────────────────────────────────┤
│  Iris Dataset │ MNIST Dataset │ Trained Models (.pkl/.h5)│
└─────────────────────────────────────────────────────────┘
```

## Request-Response Cycle

```
1. User Opens App
        │
        ▼
2. Streamlit Loads app.py
        │
        ▼
3. Render Header & Tabs
        │
        ▼
4. User Selects Tab (e.g., Iris)
        │
        ▼
5. Import iris_classifier.py
        │
        ▼
6. Load Model (cached)
        │
        ▼
7. Render Input Form
        │
        ▼
8. User Enters Values
        │
        ▼
9. User Clicks "Predict"
        │
        ▼
10. Model.predict(features)
        │
        ▼
11. Generate Visualizations
        │
        ▼
12. Display Results
```

## Security Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    USER INPUT                           │
├─────────────────────────────────────────────────────────┤
│  Validation │ Sanitization │ Type Checking              │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                 PROCESSING LAYER                        │
├─────────────────────────────────────────────────────────┤
│  Bounded Inputs │ Error Handling │ Try-Except Blocks    │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                   MODEL LAYER                           │
├─────────────────────────────────────────────────────────┤
│  Pre-trained Models │ No User Code Execution            │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                   OUTPUT                                │
├─────────────────────────────────────────────────────────┤
│  Safe Rendering │ No Injection │ Escaped HTML           │
└─────────────────────────────────────────────────────────┘
```

## Performance Optimization

```
┌────────────────────────────────────┐
│   Optimization Strategies          │
├────────────────────────────────────┤
│                                    │
│  1. Model Caching                  │
│     └─► @st.cache_resource         │
│                                    │
│  2. Lazy Loading                   │
│     └─► Import on demand           │
│                                    │
│  3. Efficient Computation          │
│     └─► NumPy vectorization        │
│                                    │
│  4. Image Optimization             │
│     └─► Resize before processing   │
│                                    │
│  5. Progressive Loading            │
│     └─► Tabs load separately       │
│                                    │
└────────────────────────────────────┘
```

## Scalability Considerations

```
Current Architecture:
  Single Instance
  In-Memory Caching
  Synchronous Processing

Scaling Options:
  ├─► Horizontal: Multiple instances + Load Balancer
  ├─► Vertical: Larger instance (more RAM/CPU)
  ├─► Caching: Redis for shared cache
  ├─► Queue: Celery for async tasks
  └─► Database: PostgreSQL for persistence
```

---

**Architecture designed by Team Aimtech7**
**Last Updated**: October 2025
