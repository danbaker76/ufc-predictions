from flask import Flask, request, jsonify
import sys
import os

# Add project root to path (ensures 'src' is importable)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.predict import predict_fight

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    ...