from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import re

app = Flask(__name__)
CORS(app)

# Load the model bundle
bundle = joblib.load('recipe_model_bundle.pkl')
model = bundle['model']
label_encoder = bundle['label_encoder']
scaler = bundle['scaler']
allergen_encoder = bundle['allergen_encoder']
difficulty_encoder = bundle['difficulty_encoder']

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.route('/predict_r', methods=['POST'])
def predict():
    try:
        data = request.json

        cooking_time = float(data.get('Cooking_Time_Minutes', 0))
        difficulty_level = clean_text(data.get('Difficulty_Level', 'Unknown'))
        calories = float(data.get('Calories_Per_Serving', 0))
        allergen_info = data.get('Allergen_Information', 'Unknown')

        # Encode difficulty and allergen
        difficulty_encoded = difficulty_encoder.transform([difficulty_level])[0]
        allergen_encoded = allergen_encoder.transform([[allergen_info]]).toarray()

        # Normalize numerical features
        numeric_features = scaler.transform([[cooking_time, calories]])[0]

        # Combine all features
        features = np.hstack([numeric_features, [difficulty_encoded], allergen_encoded[0]])
        features = features.reshape(1, -1)

        # Predict
        prediction = model.predict(features)[0]
        recipe_name = label_encoder.inverse_transform([prediction])[0]

        return jsonify({'recipe_name': recipe_name})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
