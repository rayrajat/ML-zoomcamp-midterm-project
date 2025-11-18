from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Expect JSON list of dicts
    df = pd.DataFrame(data)
    scaled = scaler.transform(df)
    preds = model.predict(scaled)
    return jsonify({'predictions': preds.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9696, debug=True)