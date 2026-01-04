"""
Logistic Regression Model Training Module
This module trains a Logistic Regression model for rainfall prediction.

Logistic Regression is a simple yet powerful algorithm for binary classification.
It's fast, interpretable, and works well as a baseline model.
"""

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os


def load_processed_data():
    """
    Load preprocessed training and testing data.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print("📂 Loading preprocessed data...")
    
    X_train = np.load('data/processed/X_train.npy')
    X_test = np.load('data/processed/X_test.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_test = np.load('data/processed/y_test.npy')
    
    print(f"✅ Data loaded! Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression model.
    
    Parameters:
    - max_iter: Maximum number of iterations (increased for convergence)
    - random_state: For reproducibility
    - solver: Algorithm to use in optimization
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        LogisticRegression: Trained model
    """
    print("\n🎓 Training Logistic Regression model...")
    
    # Initialize the model
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    print("✅ Model training completed!")
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate the trained model on both training and test data.
    
    Args:
        model: Trained model
        X_train, X_test: Features
        y_train, y_test: True labels
    """
    print("\n📊 Evaluating model performance...")
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate accuracies
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"\n   Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Detailed classification report for test set
    print("\n📈 Classification Report (Test Set):")
    print("-" * 60)
    print(classification_report(y_test, y_test_pred, 
                                target_names=['No Rain', 'Rain'],
                                digits=4))
    
    # Confusion Matrix
    print("🔍 Confusion Matrix (Test Set):")
    print("-" * 60)
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"                 Predicted No Rain | Predicted Rain")
    print(f"Actual No Rain:         {cm[0][0]:>6}      |     {cm[0][1]:>6}")
    print(f"Actual Rain:            {cm[1][0]:>6}      |     {cm[1][1]:>6}")
    print("-" * 60)


def save_model(model, model_name='logistic_regression'):
    """
    Save the trained model to disk.
    
    Args:
        model: Trained model
        model_name: Name for the saved model file
    """
    print(f"\n💾 Saving model...")
    
    # Create directory if it doesn't exist
    os.makedirs('models/saved_models', exist_ok=True)
    
    # Save model
    model_path = f'models/saved_models/{model_name}.pkl'
    joblib.dump(model, model_path)
    
    print(f"✅ Model saved at: {model_path}")


def main():
    """
    Main function to run the complete Logistic Regression training pipeline.
    """
    print("=" * 60)
    print("🚀 Logistic Regression Training Pipeline")
    print("=" * 60)
    
    # Step 1: Load preprocessed data
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # Step 2: Train the model
    model = train_logistic_regression(X_train, y_train)
    
    # Step 3: Evaluate the model
    evaluate_model(model, X_train, X_test, y_train, y_test)
    
    # Step 4: Save the model
    save_model(model, 'logistic_regression')
    
    print("\n" + "=" * 60)
    print("✅ Logistic Regression Pipeline Completed!")
    print("=" * 60)
    
    return model


# Run training if this file is executed directly
if __name__ == "__main__":
    main()