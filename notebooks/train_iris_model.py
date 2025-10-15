"""
Training script for Iris Decision Tree Classifier
Run this to generate the iris_decision_tree.pkl model
"""

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
from pathlib import Path

def train_iris_model():
    print("Loading Iris dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training Decision Tree Classifier...")
    model = DecisionTreeClassifier(
        max_depth=4,
        min_samples_split=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    models_dir = Path("../models")
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "iris_decision_tree.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"\nModel saved to: {model_path}")
    print(f"Feature importances: {model.feature_importances_}")

    return model

if __name__ == "__main__":
    train_iris_model()
