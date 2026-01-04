"""
LSTM Neural Network Training Module
This module trains an LSTM (Long Short-Term Memory) model for rainfall prediction.

LSTM is a type of Recurrent Neural Network (RNN) that can learn patterns
in sequential data, making it ideal for time-series weather prediction.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


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


def reshape_data_for_lstm(X_train, X_test):
    """
    Reshape data for LSTM input.
    LSTM expects 3D input: (samples, timesteps, features)
    
    Args:
        X_train: Training features (2D)
        X_test: Test features (2D)
        
    Returns:
        tuple: Reshaped (X_train, X_test) in 3D format
    """
    print("\n🔄 Reshaping data for LSTM...")
    
    # Reshape to (samples, timesteps=1, features)
    X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
    print(f"   Training shape: {X_train_reshaped.shape}")
    print(f"   Test shape: {X_test_reshaped.shape}")
    
    return X_train_reshaped, X_test_reshaped


def build_lstm_model(input_shape):
    """
    Build LSTM neural network architecture.
    
    Architecture:
    - LSTM Layer 1: 50 units with return sequences
    - Dropout: 20% to prevent overfitting
    - LSTM Layer 2: 50 units
    - Dropout: 20%
    - Dense Layer: 1 unit with sigmoid activation (binary classification)
    
    Args:
        input_shape: Shape of input data (timesteps, features)
        
    Returns:
        Sequential: Compiled LSTM model
    """
    print("\n🏗️  Building LSTM model architecture...")
    
    model = Sequential([
        # First LSTM layer with 50 units
        LSTM(50, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(0.2),  # Dropout to prevent overfitting
        
        # Second LSTM layer with 50 units
        LSTM(50, activation='relu'),
        Dropout(0.2),
        
        # Output layer (binary classification)
        Dense(1, activation='sigmoid')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ Model architecture built!")
    print("\n📋 Model Summary:")
    print("-" * 60)
    model.summary()
    print("-" * 60)
    
    return model


def train_lstm_model(model, X_train, y_train, X_test, y_test):
    """
    Train the LSTM model with early stopping and model checkpointing.
    
    Args:
        model: Compiled LSTM model
        X_train, y_train: Training data
        X_test, y_test: Validation data
        
    Returns:
        tuple: (trained model, training history)
    """
    print("\n🎓 Training LSTM model...")
    print("   This may take a few minutes...")
    
    # Create callbacks
    # Early stopping: Stop training if validation loss doesn't improve
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Model checkpoint: Save the best model during training
    os.makedirs('models/saved_models', exist_ok=True)
    checkpoint = ModelCheckpoint(
        'models/saved_models/lstm_best.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )
    
    print("✅ Model training completed!")
    return model, history


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate the trained LSTM model.
    
    Args:
        model: Trained LSTM model
        X_train, X_test: Features
        y_train, y_test: True labels
    """
    print("\n📊 Evaluating model performance...")
    
    # Make predictions
    y_train_pred_prob = model.predict(X_train, verbose=0)
    y_test_pred_prob = model.predict(X_test, verbose=0)
    
    # Convert probabilities to binary predictions (threshold = 0.5)
    y_train_pred = (y_train_pred_prob > 0.5).astype(int).flatten()
    y_test_pred = (y_test_pred_prob > 0.5).astype(int).flatten()
    
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


def plot_training_history(history):
    """
    Plot training and validation accuracy/loss over epochs.
    
    Args:
        history: Training history object
    """
    print("\n📊 Plotting training history...")
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[1].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('models/visualizations', exist_ok=True)
    plt.savefig('models/visualizations/lstm_training_history.png', dpi=300)
    print("✅ Training history plot saved at: models/visualizations/lstm_training_history.png")
    plt.close()


def save_model(model, model_name='lstm'):
    """
    Save the final trained model.
    
    Args:
        model: Trained LSTM model
        model_name: Name for the saved model file
    """
    print(f"\n💾 Saving final model...")
    
    # Save model
    model_path = f'models/saved_models/{model_name}.keras'
    model.save(model_path)
    
    print(f"✅ Model saved at: {model_path}")


def main():
    """
    Main function to run the complete LSTM training pipeline.
    """
    print("=" * 60)
    print("🚀 LSTM Neural Network Training Pipeline")
    print("=" * 60)
    
    # Step 1: Load preprocessed data
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # Step 2: Reshape data for LSTM
    X_train_lstm, X_test_lstm = reshape_data_for_lstm(X_train, X_test)
    
    # Step 3: Build LSTM model
    input_shape = (X_train_lstm.shape[1], X_train_lstm.shape[2])
    model = build_lstm_model(input_shape)
    
    # Step 4: Train the model
    model, history = train_lstm_model(model, X_train_lstm, y_train, X_test_lstm, y_test)
    
    # Step 5: Evaluate the model
    evaluate_model(model, X_train_lstm, X_test_lstm, y_train, y_test)
    
    # Step 6: Plot training history
    plot_training_history(history)
    
    # Step 7: Save the final model
    save_model(model, 'lstm')
    
    print("\n" + "=" * 60)
    print("✅ LSTM Training Pipeline Completed!")
    print("=" * 60)
    
    return model


# Run training if this file is executed directly
if __name__ == "__main__":
    main()