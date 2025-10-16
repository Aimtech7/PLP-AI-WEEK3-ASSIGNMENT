"""
Quick setup script to train and save both models
Run this before starting the Streamlit app
"""

import sys
from pathlib import Path

print("=" * 60)
print("AI Tools & Frameworks Dashboard - Model Setup")
print("=" * 60)

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

print("\n[1/2] Training Iris Decision Tree Model...")
print("-" * 60)

try:
    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import pickle

    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = DecisionTreeClassifier(max_depth=4, min_samples_split=5, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    model_path = models_dir / "iris_decision_tree.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"✓ Iris model trained successfully!")
    print(f"  Accuracy: {accuracy * 100:.2f}%")
    print(f"  Saved to: {model_path}")

except Exception as e:
    print(f"✗ Error training Iris model: {e}")
    sys.exit(1)

print("\n[2/2] Training MNIST CNN Model...")
print("-" * 60)
print("This may take 5-10 minutes depending on your hardware...")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    import numpy as np

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

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

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\nTraining CNN (10 epochs)...")
    history = model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=10,
        validation_split=0.1,
        verbose=1
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    model_path = models_dir / "mnist_cnn.h5"
    model.save(model_path)

    print(f"\n✓ MNIST model trained successfully!")
    print(f"  Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"  Saved to: {model_path}")

except Exception as e:
    print(f"✗ Error training MNIST model: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("Setup Complete! All models are ready.")
print("=" * 60)
print("\nRun the app with: streamlit run app.py")
