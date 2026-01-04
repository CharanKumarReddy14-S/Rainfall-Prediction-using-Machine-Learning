"""
Random Forest Classifier Training Module
This module trains a Random Forest model for rainfall prediction.

Random Forest is an ensemble learning method that creates multiple decision trees
and combines their predictions for better accuracy and robustness.
"""

import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
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


def train_random_forest(X_train, y_train):
    """
    Train a Random Forest Classifier.
    
    Parameters:
    - n_estimators: Number of trees in the forest
    - max_depth: Maximum depth of each tree
    - min_samples_split: Minimum samples required to split a node
    - random_state: For reproducibility
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        RandomForestClassifier: Trained model
    """
    print("\n🌲 Training Random Forest Classifier...")
    print("   Building 100 decision trees...")
    
    # Initialize the model with optimal parameters
    model = RandomForestClassifier(
        n_estimators=100,        # Number of trees
        max_depth=10,            # Maximum depth of trees
        min_samples_split=5,     # Minimum samples to split a node
        random_state=42,
        n_jobs=-1                # Use all available CPU cores
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
    
    # Check for overfitting
    if train_accuracy - test_accuracy > 0.1:
        print("   ⚠️  Warning: Possible overfitting detected!")
    else:
        print("   ✅ Good generalization!")
    
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


def plot_feature_importance(model):
    """
    Visualize feature importance from the Random Forest model.
    This shows which weather features are most important for prediction.
    
    Args:
        model: Trained Random Forest model
    """
    print("\n📊 Plotting feature importance...")
    
    # Feature names
    feature_names = ['MinTemp', 'MaxTemp', 'Humidity', 'WindSpeed', 'Pressure', 'RainToday']
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance in Random Forest Model', fontsize=16, fontweight='bold')
    plt.bar(range(len(importances)), importances[indices], color='skyblue', edgecolor='navy')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Importance Score', fontsize=12)
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('models/visualizations', exist_ok=True)
    plt.savefig('models/visualizations/feature_importance_rf.png', dpi=300)
    print("✅ Feature importance plot saved at: models/visualizations/feature_importance_rf.png")
    plt.close()
    
    # Print importance values
    print("\n🎯 Feature Importance Rankings:")
    print("-" * 40)
    for i, idx in enumerate(indices, 1):
        print(f"   {i}. {feature_names[idx]:<15} : {importances[idx]:.4f}")


def save_model(model, model_name='random_forest'):
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
    Main function to run the complete Random Forest training pipeline.
    """
    print("=" * 60)
    print("🚀 Random Forest Training Pipeline")
    print("=" * 60)
    
    # Step 1: Load preprocessed data
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # Step 2: Train the model
    model = train_random_forest(X_train, y_train)
    
    # Step 3: Evaluate the model
    evaluate_model(model, X_train, X_test, y_train, y_test)
    
    # Step 4: Plot feature importance
    plot_feature_importance(model)
    
    # Step 5: Save the model
    save_model(model, 'random_forest')
    
    print("\n" + "=" * 60)
    print("✅ Random Forest Pipeline Completed!")
    print("=" * 60)
    
    return model


# Run training if this file is executed directly
if __name__ == "__main__":
    main()