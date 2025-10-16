import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

@st.cache_resource
def load_mnist_model():
    try:
        import tensorflow as tf
        model_path = Path("models/mnist_cnn.h5")
        if model_path.exists():
            return tf.keras.models.load_model(model_path)
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image):
    img = ImageOps.grayscale(image)
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = 255 - img_array
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array, np.array(img)

def render_mnist_classifier():
    st.header("🔢 MNIST Digit Recognition")
    st.markdown("Draw or upload a handwritten digit and let the CNN predict what it is!")

    input_method = st.radio(
        "Choose input method:",
        ["✏️ Draw on Canvas", "📤 Upload Image"],
        horizontal=True
    )

    col1, col2 = st.columns([1, 1])

    image_data = None
    processed_img = None

    with col1:
        if input_method == "✏️ Draw on Canvas":
            st.subheader("Draw a Digit (0-9)")

            try:
                from streamlit_drawable_canvas import st_canvas

                canvas_result = st_canvas(
                    fill_color="rgba(0, 0, 0, 1)",
                    stroke_width=20,
                    stroke_color="#FFFFFF",
                    background_color="#000000",
                    height=280,
                    width=280,
                    drawing_mode="freedraw",
                    key="canvas",
                )

                if canvas_result.image_data is not None:
                    img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
                    img = img.convert('L')
                    image_data = img

            except ImportError:
                st.warning("Canvas drawing not available. Please install streamlit-drawable-canvas.")
                st.code("pip install streamlit-drawable-canvas")

        else:
            st.subheader("Upload Digit Image")
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=['png', 'jpg', 'jpeg'],
                help="Upload a clear image of a handwritten digit"
            )

            if uploaded_file is not None:
                image_data = Image.open(uploaded_file)
                st.image(image_data, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.subheader("Prediction Results")

        if st.button("🎯 Predict Digit", use_container_width=True, type="primary"):
            if image_data is None:
                st.warning("Please draw a digit or upload an image first.")
            else:
                model = load_mnist_model()

                if model is None:
                    st.error("Model not found. Please ensure mnist_cnn.h5 is in the models/ directory.")
                else:
                    with st.spinner("Analyzing digit..."):
                        try:
                            processed_img, display_img = preprocess_image(image_data)

                            prediction = model.predict(processed_img, verbose=0)
                            predicted_digit = np.argmax(prediction[0])
                            confidence = prediction[0][predicted_digit]

                            st.success(f"**Predicted Digit:** {predicted_digit}")
                            st.metric("Confidence", f"{confidence * 100:.2f}%")

                            st.markdown("---")
                            st.markdown("**Preprocessed Image (28x28):**")

                            fig_img, ax_img = plt.subplots(figsize=(3, 3))
                            ax_img.imshow(display_img, cmap='gray')
                            ax_img.axis('off')
                            ax_img.set_title(f'Predicted: {predicted_digit}')
                            st.pyplot(fig_img)

                            st.markdown("**Confidence Distribution:**")

                            fig, ax = plt.subplots(figsize=(8, 4))
                            colors = ['#22c55e' if i == predicted_digit else '#3b82f6' for i in range(10)]
                            bars = ax.bar(range(10), prediction[0], color=colors)
                            ax.set_xlabel('Digit')
                            ax.set_ylabel('Probability')
                            ax.set_title('Prediction Confidence by Digit')
                            ax.set_xticks(range(10))

                            for i, (bar, prob) in enumerate(zip(bars, prediction[0])):
                                if prob > 0.01:
                                    ax.text(i, prob + 0.02, f'{prob*100:.1f}%', ha='center', fontsize=8)

                            plt.tight_layout()
                            st.pyplot(fig)

                        except Exception as e:
                            st.error(f"Error processing image: {e}")
        else:
            st.info("Draw or upload a digit image and click 'Predict Digit' to see results.")

    with st.expander("ℹ️ About MNIST & CNN"):
        st.markdown("""
        **MNIST Dataset:**
        - Contains 70,000 handwritten digit images (0-9)
        - Each image is 28x28 pixels in grayscale
        - Standard benchmark for computer vision models

        **Convolutional Neural Network (CNN):**
        - Deep learning architecture specialized for image processing
        - Uses convolutional layers to detect features
        - Typically achieves >99% accuracy on MNIST

        **Model Architecture:**
        - Input: 28x28 grayscale image
        - Multiple convolutional and pooling layers
        - Dense layers for classification
        - Output: 10 classes (digits 0-9)
        """)

    with st.expander("💡 Tips for Best Results"):
        st.markdown("""
        - Draw digits centered in the canvas
        - Use clear, bold strokes
        - Make digits large enough to fill most of the space
        - Avoid extra marks or decorations
        - For uploads, use clear images with good contrast
        """)
