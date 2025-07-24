from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import joblib
import pandas as pd
import numpy as np

# Initialize the Flask application
app = Flask(__name__)
origins = [
    "http://localhost:3000", # Your local React app
    "https://exam-app-cnp7.vercel.app" # IMPORTANT: Add your deployed frontend URL here later
]

CORS(app, origins=origins)
# --- Load the saved model, scaler, AND columns ---
model = joblib.load('diamond_price_predictor_model.joblib')
scaler = joblib.load('scaler.joblib')
MODEL_COLUMNS = joblib.load('model_columns.joblib') # Load the columns

# --- Define the API endpoint for prediction ---
@app.route('/predict', methods=['POST'])
def predict():
    """Receives diamond features in a POST request and returns a price prediction."""
    try:
        # Get data from the POST request
        data = request.get_json()

        # --- Create a DataFrame from the input data ---
        # Use the loaded MODEL_COLUMNS to ensure order and names are correct
        input_df = pd.DataFrame(columns=MODEL_COLUMNS)
        input_df.loc[0] = np.zeros(len(MODEL_COLUMNS))

        # --- Populate the DataFrame with user input ---
        # Set the numerical features
        input_df['carat'] = data['carat']
        input_df['depth'] = data['depth']
        input_df['table'] = data['table']
        input_df['x'] = data['x']
        input_df['y'] = data['y']
        input_df['z'] = data['z']

        # Set the one-hot encoded categorical features
        if f"cut_{data['cut']}" in MODEL_COLUMNS:
            input_df[f"cut_{data['cut']}"] = 1
        
        if f"color_{data['color']}" in MODEL_COLUMNS:
            input_df[f"color_{data['color']}"] = 1

        if f"clarity_{data['clarity']}" in MODEL_COLUMNS:
            input_df[f"clarity_{data['clarity']}"] = 1

        # --- Scale the features ---
        scaled_features = scaler.transform(input_df)

        # --- Make a prediction ---
        prediction = model.predict(scaled_features)

        # --- Return the prediction as JSON ---
        return jsonify({'predicted_price': round(prediction[0], 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# --- Run the app ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)