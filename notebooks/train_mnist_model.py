"""
Training script for MNIST CNN Classifier
Run this to generate the mnist_cnn.h5 model
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from pathlib import Path

def create_cnn_model():
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),

        layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Flatten(),
        layers.Dropout(0.5),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),

        layers.Dense(10, activation='softmax')
    ])

    return model

def train_mnist_model():
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    print(f"Training samples: {x_train.shape[0]}")
    print(f"Test samples: {x_test.shape[0]}")

    print("\nCreating CNN model...")
    model = create_cnn_model()

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\nModel Architecture:")
    model.summary()

    print("\nTraining model...")
    history = model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=10,
        validation_split=0.1,
        verbose=1
    )

    print("\nEvaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    models_dir = Path("../models")
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "mnist_cnn.h5"
    model.save(model_path)

    print(f"\nModel saved to: {model_path}")

    return model, history

if __name__ == "__main__":
    train_mnist_model()
