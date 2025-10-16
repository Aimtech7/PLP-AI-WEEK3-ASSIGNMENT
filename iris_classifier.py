import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

@st.cache_resource
def load_iris_model():
    model_path = Path("models/iris_decision_tree.pkl")
    if model_path.exists():
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None

def render_iris_classifier():
    st.header("🪴 Iris Species Classifier")
    st.markdown("Predict iris species using a Decision Tree model trained on the classic Iris dataset.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input Features")
        st.markdown("Enter the measurements of the iris flower:")

        sepal_length = st.number_input(
            "Sepal Length (cm)",
            min_value=0.0,
            max_value=10.0,
            value=5.1,
            step=0.1,
            help="Length of the sepal in centimeters"
        )

        sepal_width = st.number_input(
            "Sepal Width (cm)",
            min_value=0.0,
            max_value=10.0,
            value=3.5,
            step=0.1,
            help="Width of the sepal in centimeters"
        )

        petal_length = st.number_input(
            "Petal Length (cm)",
            min_value=0.0,
            max_value=10.0,
            value=1.4,
            step=0.1,
            help="Length of the petal in centimeters"
        )

        petal_width = st.number_input(
            "Petal Width (cm)",
            min_value=0.0,
            max_value=10.0,
            value=0.2,
            step=0.1,
            help="Width of the petal in centimeters"
        )

        predict_button = st.button("🔮 Predict Species", use_container_width=True, type="primary")

    with col2:
        st.subheader("Prediction Results")

        if predict_button:
            model = load_iris_model()

            if model is None:
                st.error("Model not found. Please ensure iris_decision_tree.pkl is in the models/ directory.")
            else:
                with st.spinner("Analyzing iris features..."):
                    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

                    prediction = model.predict(features)[0]
                    probabilities = model.predict_proba(features)[0]

                    species_names = ['Setosa', 'Versicolor', 'Virginica']
                    predicted_species = species_names[prediction]

                    st.success(f"**Predicted Species:** {predicted_species}")
                    st.metric("Confidence", f"{probabilities[prediction] * 100:.1f}%")

                    st.markdown("---")
                    st.markdown("**Confidence Probabilities:**")

                    prob_df = pd.DataFrame({
                        'Species': species_names,
                        'Probability': probabilities
                    })

                    fig, ax = plt.subplots(figsize=(8, 4))
                    colors = ['#22c55e' if i == prediction else '#3b82f6' for i in range(3)]
                    bars = ax.barh(prob_df['Species'], prob_df['Probability'], color=colors)
                    ax.set_xlabel('Probability')
                    ax.set_xlim(0, 1)
                    ax.set_title('Prediction Confidence by Species')

                    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
                        ax.text(prob + 0.02, i, f'{prob*100:.1f}%', va='center')

                    plt.tight_layout()
                    st.pyplot(fig)
        else:
            st.info("Enter flower measurements and click 'Predict Species' to see results.")

    with st.expander("ℹ️ About the Iris Dataset"):
        st.markdown("""
        The **Iris dataset** is a classic dataset in machine learning, containing measurements of 150 iris flowers from three species:

        - **Setosa**: Typically has shorter petals
        - **Versicolor**: Medium-sized petals and sepals
        - **Virginica**: Largest petals and sepals

        **Features:**
        - Sepal Length & Width
        - Petal Length & Width

        **Model:** Decision Tree Classifier trained with scikit-learn
        """)

    with st.expander("📊 Decision Boundary Visualization"):
        st.markdown("**Feature Importance in Classification:**")
        st.markdown("""
        The decision tree uses these features to classify iris species.
        Typically, petal measurements are more discriminative than sepal measurements.
        """)

        try:
            model = load_iris_model()
            if model:
                feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
                importances = model.feature_importances_

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.barh(feature_names, importances, color='#3b82f6')
                ax.set_xlabel('Feature Importance')
                ax.set_title('Feature Importance in Decision Tree')
                plt.tight_layout()
                st.pyplot(fig)
        except:
            st.info("Feature importance visualization requires a trained model.")
