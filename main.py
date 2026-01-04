"""
Main Pipeline for Rainfall Prediction Project
This script orchestrates the entire machine learning pipeline:
1. Data preprocessing
2. Model training (all three models)
3. Model evaluation and comparison
4. Sample predictions

Run this file to execute the complete project pipeline.
"""

import sys
import os
import time

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import modules
from src import data_preprocessing
from src import train_logistic_regression
from src import train_random_forest
from src import train_lstm
from src import evaluate_model
from src import predict


def print_header(text):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def print_step(step_num, total_steps, description):
    """Print formatted step information."""
    print(f"\n{'─' * 80}")
    print(f"📍 STEP {step_num}/{total_steps}: {description}")
    print(f"{'─' * 80}")


def run_preprocessing():
    """Execute data preprocessing pipeline."""
    print_step(1, 5, "DATA PREPROCESSING")
    data_preprocessing.preprocess_pipeline()
    print("\n✅ Preprocessing completed successfully!")
    time.sleep(1)


def run_model_training():
    """Execute training for all three models."""
    print_step(2, 5, "MODEL TRAINING")
    
    # Train Logistic Regression
    print("\n" + "─" * 80)
    print("🔸 Training Model 1/3: Logistic Regression")
    print("─" * 80)
    train_logistic_regression.main()
    time.sleep(1)
    
    # Train Random Forest
    print("\n" + "─" * 80)
    print("🔸 Training Model 2/3: Random Forest")
    print("─" * 80)
    train_random_forest.main()
    time.sleep(1)
    
    # Train LSTM
    print("\n" + "─" * 80)
    print("🔸 Training Model 3/3: LSTM Neural Network")
    print("─" * 80)
    train_lstm.main()
    
    print("\n✅ All models trained successfully!")
    time.sleep(1)


def run_evaluation():
    """Execute model evaluation and comparison."""
    print_step(3, 5, "MODEL EVALUATION & COMPARISON")
    evaluate_model.main()
    print("\n✅ Evaluation completed successfully!")
    time.sleep(1)


def run_sample_predictions():
    """Run sample predictions using the trained models."""
    print_step(4, 5, "SAMPLE PREDICTIONS")
    
    # Sample weather data scenarios
    scenarios = [
        {
            'name': 'High Rain Probability',
            'data': {
                'MinTemp': 15.0,
                'MaxTemp': 22.0,
                'Humidity': 88.0,
                'WindSpeed': 12.0,
                'Pressure': 1008.0,
                'RainToday': 'Yes'
            }
        },
        {
            'name': 'Low Rain Probability',
            'data': {
                'MinTemp': 22.0,
                'MaxTemp': 33.0,
                'Humidity': 42.0,
                'WindSpeed': 8.0,
                'Pressure': 1016.0,
                'RainToday': 'No'
            }
        }
    ]
    
    # Run predictions
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'─' * 80}")
        print(f"🌤️  Sample Prediction {i}: {scenario['name']}")
        print(f"{'─' * 80}")
        predict.predict_rainfall(scenario['data'], model_name='random_forest')
        time.sleep(0.5)
    
    print("\n✅ Sample predictions completed!")
    time.sleep(1)


def print_project_summary():
    """Print final project summary."""
    print_step(5, 5, "PROJECT SUMMARY")
    
    print("\n📊 PROJECT OVERVIEW")
    print("─" * 80)
    print("✅ Data preprocessing completed")
    print("✅ Three models trained:")
    print("   • Logistic Regression")
    print("   • Random Forest Classifier")
    print("   • LSTM Neural Network")
    print("✅ Model evaluation and comparison completed")
    print("✅ Predictions tested successfully")
    
    print("\n📁 OUTPUT FILES GENERATED")
    print("─" * 80)
    print("📂 data/processed/")
    print("   • X_train.npy, X_test.npy, y_train.npy, y_test.npy")
    print("\n📂 models/saved_models/")
    print("   • logistic_regression.pkl")
    print("   • random_forest.pkl")
    print("   • lstm.keras")
    print("\n📂 models/visualizations/")
    print("   • feature_importance_rf.png")
    print("   • lstm_training_history.png")
    print("   • model_comparison.png")
    print("   • confusion_matrices.png")
    print("\n📂 models/")
    print("   • scaler.pkl")
    print("   • label_encoders.pkl")
    
    print("\n🚀 NEXT STEPS")
    print("─" * 80)
    print("1. Review model comparisons in: models/visualizations/")
    print("2. Make predictions with: python src/predict.py")
    print("3. Explore data with: jupyter notebook notebooks/eda.ipynb")
    print("4. Read the README.md for detailed documentation")
    
    print("\n💡 TIPS")
    print("─" * 80)
    print("• Use the best performing model for production")
    print("• Retrain models with more data for better accuracy")
    print("• Adjust hyperparameters for optimization")
    print("• Monitor model performance over time")


def main():
    """
    Main function to run the complete pipeline.
    """
    try:
        # Print welcome banner
        print("\n" + "=" * 80)
        print("  🌧️  RAINFALL PREDICTION MACHINE LEARNING PROJECT")
        print("  Complete End-to-End Pipeline")
        print("=" * 80)
        print("\n📋 This pipeline will:")
        print("   1. Preprocess the weather data")
        print("   2. Train three different ML models")
        print("   3. Evaluate and compare model performance")
        print("   4. Run sample predictions")
        print("   5. Generate comprehensive reports")
        print("\n⏱️  Estimated time: 5-10 minutes")
        
        input("\n➡️  Press ENTER to start the pipeline...")
        
        start_time = time.time()
        
        # Step 1: Preprocessing
        run_preprocessing()
        
        # Step 2: Model Training
        run_model_training()
        
        # Step 3: Evaluation
        run_evaluation()
        
        # Step 4: Sample Predictions
        run_sample_predictions()
        
        # Step 5: Project Summary
        print_project_summary()
        
        # Calculate total time
        end_time = time.time()
        total_time = end_time - start_time
        minutes = int(total_time // 60)
        seconds = int(total_time % 60)
        
        print("\n" + "=" * 80)
        print(f"✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"⏱️  Total Time: {minutes} minutes {seconds} seconds")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\n❌ Pipeline interrupted by user.")
        print("   You can resume by running specific modules individually.")
        sys.exit(1)
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ PIPELINE ERROR")
        print("=" * 80)
        print(f"Error: {str(e)}")
        print("\n🔍 Troubleshooting:")
        print("   1. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("   2. Check that you're in the project root directory")
        print("   3. Verify Python version is 3.7 or higher")
        print("   4. Run individual modules to identify the issue")
        sys.exit(1)


if __name__ == "__main__":
    main()