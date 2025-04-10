from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import re

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def check():
    return 'App is Working'



if __name__ == '__main__':
    app.run(debug=True)
