"""
Model Evaluation and Comparison Module
This module loads all trained models and compares their performance
on the same test dataset to determine which model performs best.
"""

import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow import keras
import os


def load_test_data():
    """
    Load preprocessed test data.
    
    Returns:
        tuple: (X_test, y_test)
    """
    print("📂 Loading test data...")
    
    X_test = np.load('data/processed/X_test.npy')
    y_test = np.load('data/processed/y_test.npy')
    
    print(f"✅ Test data loaded! Samples: {len(X_test)}")
    return X_test, y_test


def load_models():
    """
    Load all trained models.
    
    Returns:
        dict: Dictionary of loaded models
    """
    print("\n🔧 Loading trained models...")
    
    models = {}
    
    # Load Logistic Regression
    try:
        models['Logistic Regression'] = joblib.load('models/saved_models/logistic_regression.pkl')
        print("   ✅ Logistic Regression loaded")
    except FileNotFoundError:
        print("   ⚠️  Logistic Regression model not found")
    
    # Load Random Forest
    try:
        models['Random Forest'] = joblib.load('models/saved_models/random_forest.pkl')
        print("   ✅ Random Forest loaded")
    except FileNotFoundError:
        print("   ⚠️  Random Forest model not found")
    
    # Load LSTM
    try:
        models['LSTM'] = keras.models.load_model('models/saved_models/lstm.keras')
        print("   ✅ LSTM loaded")
    except (FileNotFoundError, OSError):
        print("   ⚠️  LSTM model not found")
    
    return models


def evaluate_single_model(model, model_name, X_test, y_test):
    """
    Evaluate a single model and return metrics.
    
    Args:
        model: Trained model
        model_name: Name of the model
        X_test: Test features
        y_test: True labels
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Handle LSTM separately (needs 3D input)
    if model_name == 'LSTM':
        X_test_input = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        y_pred_prob = model.predict(X_test_input, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    else:
        y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0)
    }
    
    return metrics, y_pred


def compare_models(models, X_test, y_test):
    """
    Compare all models and display results.
    
    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: True labels
        
    Returns:
        dict: Comparison results
    """
    print("\n📊 Comparing Model Performance...")
    print("=" * 80)
    
    results = {}
    predictions = {}
    
    # Evaluate each model
    for model_name, model in models.items():
        print(f"\n🔍 Evaluating {model_name}...")
        metrics, y_pred = evaluate_single_model(model, model_name, X_test, y_test)
        results[model_name] = metrics
        predictions[model_name] = y_pred
        
        # Print metrics
        print(f"   Accuracy:  {metrics['Accuracy']:.4f} ({metrics['Accuracy']*100:.2f}%)")
        print(f"   Precision: {metrics['Precision']:.4f}")
        print(f"   Recall:    {metrics['Recall']:.4f}")
        print(f"   F1-Score:  {metrics['F1-Score']:.4f}")
    
    return results, predictions


def create_comparison_table(results):
    """
    Create and display a comparison table of all models.
    
    Args:
        results: Dictionary of model results
    """
    print("\n" + "=" * 80)
    print("📋 MODEL COMPARISON TABLE")
    print("=" * 80)
    
    # Header
    print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 80)
    
    # Print results for each model
    for model_name, metrics in results.items():
        print(f"{model_name:<25} "
              f"{metrics['Accuracy']:.4f}      "
              f"{metrics['Precision']:.4f}      "
              f"{metrics['Recall']:.4f}      "
              f"{metrics['F1-Score']:.4f}")
    
    print("=" * 80)
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['Accuracy'])
    print(f"\n🏆 Best Model: {best_model[0]} (Accuracy: {best_model[1]['Accuracy']:.4f})")


def plot_comparison_chart(results):
    """
    Create a bar chart comparing all models across metrics.
    
    Args:
        results: Dictionary of model results
    """
    print("\n📊 Creating comparison chart...")
    
    # Prepare data for plotting
    models = list(results.keys())
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(models))
    width = 0.2
    
    # Plot bars for each metric
    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in models]
        ax.bar(x + i * width, values, width, label=metric)
    
    # Customize plot
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('models/visualizations', exist_ok=True)
    plt.savefig('models/visualizations/model_comparison.png', dpi=300)
    print("✅ Comparison chart saved at: models/visualizations/model_comparison.png")
    plt.close()


def plot_confusion_matrices(predictions, y_test):
    """
    Plot confusion matrices for all models side by side.
    
    Args:
        predictions: Dictionary of model predictions
        y_test: True labels
    """
    print("\n📊 Creating confusion matrices...")
    
    n_models = len(predictions)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    
    # Handle single model case
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, y_pred) in enumerate(predictions.items()):
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Rain', 'Rain'],
                   yticklabels=['No Rain', 'Rain'],
                   ax=axes[idx], cbar=False)
        
        axes[idx].set_title(f'{model_name}\nConfusion Matrix', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predicted', fontsize=10)
        axes[idx].set_ylabel('Actual', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig('models/visualizations/confusion_matrices.png', dpi=300)
    print("✅ Confusion matrices saved at: models/visualizations/confusion_matrices.png")
    plt.close()


def recommend_best_model(results):
    """
    Recommend the best model based on overall performance.
    
    Args:
        results: Dictionary of model results
    """
    print("\n" + "=" * 80)
    print("💡 MODEL RECOMMENDATION")
    print("=" * 80)
    
    # Calculate average score for each model
    avg_scores = {}
    for model_name, metrics in results.items():
        avg_score = np.mean([metrics['Accuracy'], metrics['Precision'], 
                            metrics['Recall'], metrics['F1-Score']])
        avg_scores[model_name] = avg_score
    
    # Sort models by average score
    ranked_models = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
    
    print("\n🏅 Model Rankings (by average performance):")
    print("-" * 80)
    for rank, (model_name, score) in enumerate(ranked_models, 1):
        print(f"   {rank}. {model_name:<25} Average Score: {score:.4f}")
    
    # Recommendation
    best_model = ranked_models[0][0]
    print("\n" + "-" * 80)
    print(f"✅ RECOMMENDED MODEL: {best_model}")
    print("-" * 80)
    
    # Provide context
    if best_model == 'Logistic Regression':
        print("   💡 Logistic Regression is fast and interpretable.")
        print("   💡 Best for: Quick predictions and understanding feature importance")
    elif best_model == 'Random Forest':
        print("   💡 Random Forest handles complex patterns well.")
        print("   💡 Best for: High accuracy with moderate computational cost")
    elif best_model == 'LSTM':
        print("   💡 LSTM captures time-series patterns.")
        print("   💡 Best for: Sequential weather data with temporal dependencies")


def main():
    """
    Main function to run complete model evaluation and comparison.
    """
    print("=" * 80)
    print("🚀 Model Evaluation and Comparison Pipeline")
    print("=" * 80)
    
    # Step 1: Load test data
    X_test, y_test = load_test_data()
    
    # Step 2: Load all trained models
    models = load_models()
    
    if not models:
        print("\n❌ No trained models found!")
        print("   Please train models first using:")
        print("   - python src/train_logistic_regression.py")
        print("   - python src/train_random_forest.py")
        print("   - python src/train_lstm.py")
        return
    
    # Step 3: Compare models
    results, predictions = compare_models(models, X_test, y_test)
    
    # Step 4: Create comparison table
    create_comparison_table(results)
    
    # Step 5: Plot comparison chart
    plot_comparison_chart(results)
    
    # Step 6: Plot confusion matrices
    plot_confusion_matrices(predictions, y_test)
    
    # Step 7: Recommend best model
    recommend_best_model(results)
    
    print("\n" + "=" * 80)
    print("✅ Evaluation Pipeline Completed!")
    print("=" * 80)


# Run evaluation if this file is executed directly
if __name__ == "__main__":
    main()