import pandas as pd
import joblib

# Load saved stuff
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Example new wine data (11 features)
new_data = pd.DataFrame({
    'fixed acidity': [7.4], 'volatile acidity': [0.7], 'citric acid': [0],
    'residual sugar': [1.9], 'chlorides': [0.076], 'free sulfur dioxide': [11],
    'total sulfur dioxide': [34], 'density': [0.9978], 'pH': [3.51],
    'sulphates': [0.56], 'alcohol': [9.4]
})

# Scale and predict
new_scaled = scaler.transform(new_data)
pred = model.predict(new_scaled)
print(f'Predicted quality: {pred[0]:.2f}')