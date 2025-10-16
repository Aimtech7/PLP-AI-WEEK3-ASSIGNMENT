import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

@st.cache_resource
def load_iris_model():
    model_path = Path("models/iris_decision_tree.pkl")
    if model_path.exists():
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None

@st.cache_resource
def load_mnist_model():
    try:
        import tensorflow as tf
        model_path = Path("models/mnist_cnn.h5")
        if model_path.exists():
            return tf.keras.models.load_model(model_path)
        return None
    except Exception as e:
        return None

def render_explainability():
    st.header("🔍 Model Explainability Dashboard")
    st.markdown("Understand how AI models make predictions using LIME and SHAP techniques.")

    model_choice = st.selectbox(
        "Select Model to Explain:",
        ["🪴 Iris Decision Tree", "🔢 MNIST CNN (Coming Soon)"]
    )

    if model_choice == "🪴 Iris Decision Tree":
        render_iris_explainability()
    else:
        st.info("MNIST explainability features are under development. Stay tuned!")

def render_iris_explainability():
    st.subheader("Iris Decision Tree Explainability")

    model = load_iris_model()

    if model is None:
        st.error("Iris model not found. Please ensure iris_decision_tree.pkl is in the models/ directory.")
        return

    tab1, tab2, tab3 = st.tabs(["Feature Importance", "LIME Explanation", "SHAP Analysis"])

    with tab1:
        st.markdown("### Feature Importance Analysis")
        st.markdown("Shows which features have the most impact on the model's predictions.")

        try:
            feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
            importances = model.feature_importances_

            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)

            col1, col2 = st.columns([2, 1])

            with col1:
                fig, ax = plt.subplots(figsize=(8, 5))
                colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(importances)))
                bars = ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors)
                ax.set_xlabel('Importance Score')
                ax.set_title('Feature Importance in Iris Classification')

                for i, (bar, imp) in enumerate(zip(bars, importance_df['Importance'])):
                    ax.text(imp + 0.01, i, f'{imp:.3f}', va='center')

                plt.tight_layout()
                st.pyplot(fig)

            with col2:
                st.dataframe(
                    importance_df.style.format({'Importance': '{:.4f}'}),
                    hide_index=True,
                    use_container_width=True
                )

            st.markdown("""
            **Interpretation:**
            - Higher importance means the feature is more critical for classification
            - Petal measurements typically show higher importance than sepal measurements
            - The model uses these features to split data at decision nodes
            """)

        except Exception as e:
            st.error(f"Error calculating feature importance: {e}")

    with tab2:
        st.markdown("### LIME (Local Interpretable Model-agnostic Explanations)")
        st.markdown("Explains individual predictions by approximating the model locally.")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("**Input Sample for Explanation:**")

            sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
            sepal_width = st.slider("Sepal Width (cm)", 2.0, 5.0, 3.0, 0.1)
            petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.5, 0.1)
            petal_width = st.slider("Petal Width (cm)", 0.1, 3.0, 1.5, 0.1)

            explain_button = st.button("🔎 Generate LIME Explanation", use_container_width=True, type="primary")

        with col2:
            if explain_button:
                try:
                    from lime.lime_tabular import LimeTabularExplainer
                    from sklearn.datasets import load_iris

                    iris_data = load_iris()
                    X_train = iris_data.data
                    feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
                    class_names = ['Setosa', 'Versicolor', 'Virginica']

                    explainer = LimeTabularExplainer(
                        X_train,
                        feature_names=feature_names,
                        class_names=class_names,
                        mode='classification'
                    )

                    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

                    prediction = model.predict(sample)[0]
                    predicted_class = class_names[prediction]

                    st.success(f"**Predicted Class:** {predicted_class}")

                    with st.spinner("Generating LIME explanation..."):
                        exp = explainer.explain_instance(
                            sample[0],
                            model.predict_proba,
                            num_features=4
                        )

                        explanation_list = exp.as_list()

                        st.markdown("**Feature Contributions:**")

                        contrib_data = []
                        for feature_desc, weight in explanation_list:
                            contrib_data.append({
                                'Feature': feature_desc,
                                'Contribution': weight
                            })

                        contrib_df = pd.DataFrame(contrib_data)

                        fig, ax = plt.subplots(figsize=(8, 5))
                        colors = ['#22c55e' if x > 0 else '#ef4444' for x in contrib_df['Contribution']]
                        ax.barh(contrib_df['Feature'], contrib_df['Contribution'], color=colors)
                        ax.set_xlabel('Contribution to Prediction')
                        ax.set_title(f'LIME Explanation for {predicted_class} Prediction')
                        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                        plt.tight_layout()
                        st.pyplot(fig)

                        st.markdown("""
                        **How to read this:**
                        - Green bars push the prediction towards the predicted class
                        - Red bars push the prediction away from the predicted class
                        - Longer bars indicate stronger influence
                        """)

                except ImportError:
                    st.error("LIME not installed. Please install: pip install lime")
                except Exception as e:
                    st.error(f"Error generating LIME explanation: {e}")
            else:
                st.info("Adjust the sliders and click 'Generate LIME Explanation'")

    with tab3:
        st.markdown("### SHAP (SHapley Additive exPlanations)")
        st.markdown("Global model interpretation using game theory-based approach.")

        if st.button("🎯 Generate SHAP Analysis", use_container_width=True, type="primary"):
            try:
                import shap
                from sklearn.datasets import load_iris

                iris_data = load_iris()
                X_train = iris_data.data[:100]
                feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']

                with st.spinner("Computing SHAP values (this may take a moment)..."):
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_train)

                    st.success("SHAP analysis complete!")

                    st.markdown("**SHAP Summary Plot:**")
                    st.markdown("Shows the impact of each feature across all samples.")

                    fig, ax = plt.subplots(figsize=(10, 6))

                    if isinstance(shap_values, list):
                        shap_values_plot = shap_values[1]
                    else:
                        shap_values_plot = shap_values

                    shap.summary_plot(
                        shap_values_plot,
                        X_train,
                        feature_names=feature_names,
                        show=False
                    )

                    plt.tight_layout()
                    st.pyplot(fig)

                    st.markdown("""
                    **Interpretation:**
                    - Each dot represents a sample
                    - Red indicates high feature values, blue indicates low values
                    - Position on x-axis shows impact on model output
                    - Features are ranked by importance (top to bottom)
                    """)

                    st.markdown("**Mean Absolute SHAP Values:**")

                    if isinstance(shap_values, list):
                        mean_shap = np.mean([np.abs(sv).mean(0) for sv in shap_values], axis=0)
                    else:
                        mean_shap = np.abs(shap_values).mean(0)

                    shap_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Mean |SHAP|': mean_shap
                    }).sort_values('Mean |SHAP|', ascending=False)

                    fig2, ax2 = plt.subplots(figsize=(8, 4))
                    ax2.barh(shap_df['Feature'], shap_df['Mean |SHAP|'], color='#3b82f6')
                    ax2.set_xlabel('Mean Absolute SHAP Value')
                    ax2.set_title('Global Feature Importance (SHAP)')
                    plt.tight_layout()
                    st.pyplot(fig2)

            except ImportError:
                st.error("SHAP not installed. Please install: pip install shap")
            except Exception as e:
                st.error(f"Error generating SHAP analysis: {e}")
                st.info("SHAP analysis works best with a trained model and dataset.")
        else:
            st.info("Click the button above to generate SHAP analysis.")

    with st.expander("ℹ️ About Model Explainability"):
        st.markdown("""
        **Why is Explainability Important?**
        - Builds trust in AI systems
        - Helps identify model biases
        - Supports debugging and model improvement
        - Required for regulatory compliance in many industries

        **Techniques:**

        **Feature Importance:**
        - Shows global importance of features across all predictions
        - Built into tree-based models
        - Fast and easy to compute

        **LIME:**
        - Local interpretability for individual predictions
        - Model-agnostic (works with any model)
        - Creates interpretable approximations

        **SHAP:**
        - Based on game theory (Shapley values)
        - Provides both local and global explanations
        - Theoretically sound and consistent
        - More computationally intensive
        """)
