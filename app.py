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

@app.route('/', methods=['GET'])
def check():
    return 'App is Working'



if __name__ == '__main__':
    app.run(debug=True)
