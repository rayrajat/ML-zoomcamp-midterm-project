# Red Wine Quality Prediction 🍷  
**ML Zoomcamp 2025 – Mid-term Project**

Predicting the quality score (3–8) of red wines using physicochemical features such as acidity, sugar, alcohol content, and sulfates.

This is a **regression** task that automates wine quality assessment — traditionally done by expert tasters — making the process faster, cheaper, and more consistent for winemakers.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-green)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue)](https://www.docker.com/)

## 📊 Dataset
- **Source**: UCI Machine Learning Repository  
- **Direct link**: [winequality-red.csv](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv)  
- 1,599 red wine samples  
- 11 input features + 1 target (`quality`: integer from 3 to 8)  
- No missing values, all numeric

## 🧠 Problem Statement
Human wine tasting is subjective, slow, and expensive. Can we predict the perceived quality of a red wine using only its chemical properties?

**Goal**: Build a regression model that accurately predicts the quality score from 11 physicochemical features.

**Key Insights from EDA**:
- Strongest positive correlation: **alcohol** (+0.48)
- Strongest negative correlation: **volatile acidity** (-0.39)
- Quality scores are imbalanced (mostly 5 and 6)
- Non-linear relationships → tree-based models perform best

## 🚀 Best Model
After comparing Linear Regression, Ridge, and Random Forest with hyperparameter tuning (GridSearchCV):

| Model              | RMSE (test) | R²    |
|--------------------|-------------|-------|
| Linear Regression  | ~0.65       | 0.36  |
| Ridge              | ~0.65       | 0.36  |
| **Random Forest**  | **~0.54**   | **0.52** |

**RandomForestRegressor (n_estimators=100, max_depth=10)** is used in production.

## 📁 Project Structure
WineQualityRegression/
├── wine_quality_regression.ipynb   # Full EDA, experiments & documentation
├── train.py                      # Trains and saves the model
├── predict.py                    # Quick local prediction test
├── serve.py                      # Flask API (web service)
├── model.pkl                     # Trained model
├── scaler.pkl                    # Fitted StandardScaler
├── requirements.txt
├── Dockerfile
└── README.md

## ▶️ How to Run the Project

### 1. Clone the repository

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name/wine_regression

2. Install dependencies
pip install -r requirements.txt

3. Train the model (or use pre-trained)

python train.py

→ Creates/updates model.pkl and scaler.pkl

4A. Run Flask API locally

python serve.py

Server starts at: http://localhost:9696

4B. Or run with Docker

docker build -t wine-quality-api .
docker run -p 9696:9696 wine-quality-api

5. Test the API

curl -X POST http://localhost:9696/predict \
     -H "Content-Type: application/json" \
     -d '[{"fixed acidity":7.4,"volatile acidity":0.7,"citric acid":0.0,"residual sugar":1.9,"chlorides":0.076,"free sulfur dioxide":11,"total sulfur dioxide":34,"density":0.9978,"pH":3.51,"sulphates":0.56,"alcohol":9.4}]'

Expected response:

{"predictions":[5.12]}

6. Quick local prediction (no server)
python predict.py