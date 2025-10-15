# 🧠 AI Tools & Frameworks Assignment

This project demonstrates both **theoretical and practical mastery** of key AI frameworks — TensorFlow, PyTorch, Scikit-learn, and spaCy — through real-world tasks.  
It combines **classical ML**, **deep learning**, **NLP**, **ethical reflection**, and **deployment** with advanced additions like **XAI** and **cloud-hosted dashboards**.

---

## 📘 Project Overview

### 🎯 Objective
To evaluate understanding and proficiency in:
- Applying and comparing AI tools/frameworks.
- Implementing ethical and explainable AI systems.
- Building deployable and interactive AI applications.

### 👥 Team Members
| Name | Role | Task |
|------|------|------|
| Member 1 | Theory & Docs | Theoretical Q&A |
| Member 2 | Classical ML | Iris Decision Tree |
| Member 3 | Deep Learning | MNIST CNN |
| Member 4 | NLP | spaCy NER & Sentiment |
| Member 5 | XAI & Deployment | Explainability + Streamlit Cloud Hosting |

---

## 🧩 PART 1: Theoretical Understanding (40%)

### **Q1: Explain the primary differences between TensorFlow and PyTorch. When would you choose one over the other?**

| Feature | TensorFlow | PyTorch |
|----------|-------------|----------|
| **Developer** | Google Brain | Facebook AI |
| **Computation Graph** | Static (TF1), Eager (TF2) | Dynamic (Define-by-Run) |
| **Ease of Debugging** | Moderate | Easier (Pythonic) |
| **Deployment** | TensorFlow Lite, TF Serving | TorchServe, ONNX |
| **When to Choose** | Production deployment | Research, prototyping |

✅ **Summary:**  
Use **TensorFlow** for production-ready scalable systems;  
Use **PyTorch** for flexible and experimental research.

---

### **Q2: Describe two use cases for Jupyter Notebooks in AI development.**
1. **Model Experimentation:** Visualize training metrics interactively.  
2. **Educational Reports:** Blend explanations, code, and visual outputs in one notebook.

---

### **Q3: How does spaCy enhance NLP tasks compared to basic Python string operations?**
- Pre-trained models for tokenization, POS tagging, and NER.  
- Optimized with Cython for speed and scalability.  
- Simplifies complex NLP workflows through pipelines.  

✅ **Benefit:** Automates preprocessing and entity recognition without manual regex or tokenization code.

---

### **Q4: Comparative Analysis — Scikit-learn vs TensorFlow**

| Criteria | Scikit-learn | TensorFlow |
|-----------|----------------|--------------|
| **Target Applications** | Classical ML | Deep Learning |
| **Ease of Use** | Simpler, intuitive APIs | Steeper learning curve |
| **Community Support** | Strong in academia | Strong in production |

---

## ⚙️ PART 2: Practical Implementation (50%)

### **Task 1: Classical ML with Scikit-learn**
**Dataset:** Iris  
**Goal:** Predict species using Decision Tree  
**Evaluation:** Accuracy, Precision, Recall  

📘 File: `notebooks/iris_classifier.ipynb`  
📸 Screenshot: `/screenshots/iris_confusion_matrix.png`

---

### **Task 2: Deep Learning with TensorFlow**
**Dataset:** MNIST  
**Goal:** Build CNN achieving >95% accuracy  
**Outputs:** Accuracy graph, sample predictions  

📘 File: `notebooks/mnist_cnn_tf.ipynb`  
📸 Screenshots:  
- `/screenshots/mnist_accuracy_plot.png`  
- `/screenshots/mnist_sample_predictions.png`

---

### **Task 3: NLP with spaCy**
**Dataset:** Amazon Product Reviews  
**Goal:** Extract Named Entities & analyze sentiment  

📘 File: `notebooks/spacy_nlp_reviews.ipynb`  
📸 Screenshots:  
- `/screenshots/spacy_ner_output.png`  
- `/screenshots/spacy_sentiment_output.png`

---

## 🧭 PART 3: Ethics & Optimization (10%)

### **1. Ethical Considerations**
- **Bias Examples:**
  - MNIST lacks handwriting diversity (age, region, style).
  - Sentiment models can reflect gendered or cultural bias.
- **Mitigation:**
  - TensorFlow **Fairness Indicators** to analyze subgroup metrics.
  - spaCy **rule-based lexicons** to refine sentiment detection.

---

### **2. Troubleshooting**
Common TensorFlow errors & fixes:
- **Dimension mismatch:** Ensure `input_shape=(28,28,1)`.  
- **Nan loss:** Normalize input, reduce learning rate.  
- **Wrong loss function:** Use `sparse_categorical_crossentropy` for integer labels.

---

## 💎 BONUS & ADVANCED FEATURES

### 🧠 **1. Model Comparison Dashboard**
A Streamlit dashboard comparing **Decision Tree (Scikit-learn)** and **CNN (TensorFlow)** performance:

📘 File: `/app/model_dashboard.py`

```python
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("📊 Model Comparison Dashboard")

# Example results
results = {
    "Model": ["Decision Tree", "CNN"],
    "Accuracy": [0.95, 0.985],
    "Precision": [0.94, 0.98],
    "Recall": [0.93, 0.97]
}

df = pd.DataFrame(results)
st.bar_chart(df.set_index("Model"))
st.write("✅ The CNN outperforms the Decision Tree, showing deep learning’s power on image datasets.")


## Iris model trained (example)
- Accuracy: 1.0000
- Precision (macro): 1.0000
- Recall (macro): 1.0000

Screenshots: `screenshots/iris_confusion_matrix.png`, `screenshots/iris_metrics.png`
