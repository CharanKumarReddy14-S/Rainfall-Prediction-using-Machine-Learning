# 🌧️ Rainfall Prediction Machine Learning Project

## 📋 Project Overview

This project predicts whether it will rain tomorrow based on historical weather data using three different machine learning approaches:

1. **Logistic Regression** - Fast baseline model
2. **Random Forest Classifier** - Ensemble learning for better accuracy
3. **LSTM Neural Network** - Deep learning for time-series patterns

## 🎯 Real-World Applications

### 1. **Agriculture** 🌾
- Help farmers plan irrigation schedules
- Optimize planting and harvesting times
- Prevent crop damage from unexpected rainfall

### 2. **Water Management** 💧
- Predict reservoir levels
- Plan water resource allocation
- Manage drought and flood risks

### 3. **Disaster Preparedness** 🚨
- Early warning systems for floods
- Emergency response planning
- Infrastructure maintenance scheduling

## 📊 Dataset

The project uses historical weather data with the following features:
- **Date**: Date of observation
- **MinTemp**: Minimum temperature (°C)
- **MaxTemp**: Maximum temperature (°C)
- **Humidity**: Relative humidity (%)
- **WindSpeed**: Wind speed (km/h)
- **Pressure**: Atmospheric pressure (hPa)
- **RainToday**: Did it rain today? (Yes/No)
- **RainTomorrow**: Target variable - Will it rain tomorrow? (Yes/No)

## 🚀 Installation

### Step 1: Clone or Download the Project
```bash
cd rainfall_prediction
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Mac/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
rainfall_prediction/
│
├── data/
│   └── weather.csv              # Historical weather dataset
│
├── notebooks/
│   └── eda.ipynb                # Exploratory Data Analysis
│
├── src/
│   ├── data_preprocessing.py    # Data cleaning and preparation
│   ├── train_logistic_regression.py  # Logistic Regression model
│   ├── train_random_forest.py   # Random Forest model
│   ├── train_lstm.py            # LSTM Neural Network
│   ├── evaluate_model.py        # Model evaluation and comparison
│   └── predict.py               # Make predictions on new data
│
├── models/
│   └── saved_models/            # Trained models saved here
│
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── main.py                      # Main pipeline to run everything
```

## 🎮 How to Run

### Option 1: Run Complete Pipeline
```bash
python main.py
```

This will:
1. Preprocess the data
2. Train all three models
3. Evaluate and compare models
4. Save the best model

### Option 2: Run Individual Components

#### Preprocess Data
```bash
python src/data_preprocessing.py
```

#### Train Individual Models
```bash
python src/train_logistic_regression.py
python src/train_random_forest.py
python src/train_lstm.py
```

#### Evaluate Models
```bash
python src/evaluate_model.py
```

#### Make Predictions
```bash
python src/predict.py
```

### Option 3: Exploratory Data Analysis
```bash
jupyter notebook notebooks/eda.ipynb
```

## 📈 Model Performance

After training, you'll see comparison metrics:
- **Accuracy**: Overall correctness
- **Precision**: How many predicted rain days were correct
- **Recall**: How many actual rain days were caught
- **F1-Score**: Balanced measure of precision and recall

## 🔮 Making Predictions

To predict rainfall for new data:

```python
from src.predict import predict_rainfall

# Input today's weather data
weather_data = {
    'MinTemp': 15.2,
    'MaxTemp': 28.5,
    'Humidity': 75.0,
    'WindSpeed': 12.3,
    'Pressure': 1013.5,
    'RainToday': 'Yes'
}

prediction = predict_rainfall(weather_data)
print(f"Rain Tomorrow: {prediction}")
```

## 🧪 Creating Sample Data

If you don't have `weather.csv`, the preprocessing script will generate sample data automatically.

## 📊 Visualizations

The project generates:
- Correlation heatmap showing feature relationships
- Feature importance chart (Random Forest)
- Training history plots (LSTM)
- Confusion matrices for all models

## 🛠️ Troubleshooting

### Issue: Module not found
**Solution**: Make sure you're in the project directory and all dependencies are installed
```bash
pip install -r requirements.txt
```

### Issue: No data file found
**Solution**: The script will generate sample data automatically, or place your `weather.csv` in the `data/` folder

### Issue: TensorFlow warnings
**Solution**: These are usually harmless. You can suppress them with:
```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```

## 📚 Learn More

- **Logistic Regression**: Simple binary classification
- **Random Forest**: Ensemble of decision trees
- **LSTM**: Recurrent neural network for sequences

## 👥 Contributing

Feel free to fork this project and add:
- More weather features
- Additional ML algorithms
- Real-time weather API integration
- Web dashboard for predictions

## 📝 License

This project is open source and available for educational purposes.

## 🤝 Support

For issues or questions, please check the troubleshooting section or review the code comments for detailed explanations.

---

**Happy Predicting! 🌦️**