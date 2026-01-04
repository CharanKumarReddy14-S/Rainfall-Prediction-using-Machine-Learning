"""
Data Preprocessing Module for Rainfall Prediction
This module handles:
- Loading raw weather data
- Handling missing values
- Encoding categorical variables
- Feature scaling
- Saving processed data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import joblib


def create_sample_data():
    """
    Create sample weather dataset if no CSV file exists.
    This generates realistic weather data for demonstration.
    """
    print("📊 Generating sample weather data...")
    
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic weather data
    data = {
        'Date': pd.date_range(start='2021-01-01', periods=n_samples, freq='D'),
        'MinTemp': np.random.uniform(10, 20, n_samples),
        'MaxTemp': np.random.uniform(20, 35, n_samples),
        'Humidity': np.random.uniform(40, 90, n_samples),
        'WindSpeed': np.random.uniform(5, 25, n_samples),
        'Pressure': np.random.uniform(1000, 1020, n_samples)
    }
    
    # Add some correlation between humidity and rain
    rain_today = ['Yes' if h > 70 else 'No' for h in data['Humidity']]
    rain_tomorrow = ['Yes' if h > 68 else 'No' for h in data['Humidity']]
    
    data['RainToday'] = rain_today
    data['RainTomorrow'] = rain_tomorrow
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/weather.csv', index=False)
    print("✅ Sample data created at: data/weather.csv")
    
    return df


def load_data(filepath='data/weather.csv'):
    """
    Load weather data from CSV file.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded weather data
    """
    print(f"📂 Loading data from: {filepath}")
    
    # Create sample data if file doesn't exist
    if not os.path.exists(filepath):
        print("⚠️  Data file not found. Creating sample data...")
        return create_sample_data()
    
    df = pd.read_csv(filepath)
    print(f"✅ Data loaded successfully! Shape: {df.shape}")
    return df


def handle_missing_values(df):
    """
    Handle missing values in the dataset.
    Strategy:
    - Numerical columns: Fill with median
    - Categorical columns: Fill with mode
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with missing values handled
    """
    print("\n🔧 Handling missing values...")
    
    # Check for missing values
    missing_before = df.isnull().sum().sum()
    print(f"   Missing values before: {missing_before}")
    
    # Handle numerical columns
    numerical_cols = ['MinTemp', 'MaxTemp', 'Humidity', 'WindSpeed', 'Pressure']
    for col in numerical_cols:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)
    
    # Handle categorical columns
    categorical_cols = ['RainToday', 'RainTomorrow']
    for col in categorical_cols:
        if col in df.columns:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    missing_after = df.isnull().sum().sum()
    print(f"   Missing values after: {missing_after}")
    print("✅ Missing values handled!")
    
    return df


def encode_categorical_variables(df):
    """
    Encode categorical variables (Yes/No) to numerical (1/0).
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with encoded variables
        dict: Label encoders for each categorical column
    """
    print("\n🔤 Encoding categorical variables...")
    
    label_encoders = {}
    
    # Encode RainToday and RainTomorrow
    for col in ['RainToday', 'RainTomorrow']:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
            print(f"   {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    print("✅ Categorical variables encoded!")
    return df, label_encoders


def feature_scaling(X_train, X_test):
    """
    Scale numerical features using StandardScaler.
    This normalizes features to have mean=0 and std=1.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features
        
    Returns:
        tuple: (scaled_X_train, scaled_X_test, scaler)
    """
    print("\n📏 Scaling features...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("✅ Features scaled!")
    return X_train_scaled, X_test_scaled, scaler


def prepare_features_and_target(df):
    """
    Separate features (X) and target variable (y).
    
    Args:
        df (pd.DataFrame): Preprocessed dataframe
        
    Returns:
        tuple: (X, y) - Features and target
    """
    print("\n🎯 Preparing features and target...")
    
    # Drop Date column if exists
    if 'Date' in df.columns:
        df = df.drop('Date', axis=1)
    
    # Define features and target
    X = df.drop('RainTomorrow', axis=1)
    y = df['RainTomorrow']
    
    print(f"   Features shape: {X.shape}")
    print(f"   Target shape: {y.shape}")
    print(f"   Feature columns: {list(X.columns)}")
    
    return X, y


def save_processed_data(X_train, X_test, y_train, y_test, scaler, label_encoders):
    """
    Save processed data and preprocessing objects for later use.
    
    Args:
        X_train, X_test, y_train, y_test: Train/test splits
        scaler: Fitted StandardScaler
        label_encoders: Dictionary of label encoders
    """
    print("\n💾 Saving processed data...")
    
    # Create directories
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Save train/test splits
    np.save('data/processed/X_train.npy', X_train)
    np.save('data/processed/X_test.npy', X_test)
    np.save('data/processed/y_train.npy', y_train)
    np.save('data/processed/y_test.npy', y_test)
    
    # Save preprocessing objects
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(label_encoders, 'models/label_encoders.pkl')
    
    print("✅ Processed data saved!")
    print("   Location: data/processed/")


def preprocess_pipeline():
    """
    Complete preprocessing pipeline.
    This is the main function that runs all preprocessing steps.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler, label_encoders)
    """
    print("=" * 60)
    print("🚀 Starting Data Preprocessing Pipeline")
    print("=" * 60)
    
    # Step 1: Load data
    df = load_data()
    
    # Step 2: Handle missing values
    df = handle_missing_values(df)
    
    # Step 3: Encode categorical variables
    df, label_encoders = encode_categorical_variables(df)
    
    # Step 4: Prepare features and target
    X, y = prepare_features_and_target(df)
    
    # Step 5: Split data into train and test sets
    print("\n✂️  Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training set size: {len(X_train)}")
    print(f"   Test set size: {len(X_test)}")
    
    # Step 6: Feature scaling
    X_train_scaled, X_test_scaled, scaler = feature_scaling(X_train, X_test)
    
    # Step 7: Save processed data
    save_processed_data(X_train_scaled, X_test_scaled, y_train, y_test, 
                       scaler, label_encoders)
    
    print("\n" + "=" * 60)
    print("✅ Preprocessing Pipeline Completed Successfully!")
    print("=" * 60)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoders


# Run preprocessing if this file is executed directly
if __name__ == "__main__":
    preprocess_pipeline()