from flask import Flask, request, jsonify
import sys, os, traceback

# Ensure the project root is on sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.predict import predict_fight

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        fighter_a = data.get('fighter_a')
        fighter_b = data.get('fighter_b')
        if not fighter_a or not fighter_b:
            return jsonify({'error': 'Provide both fighter_a and fighter_b'}), 400

        prob = predict_fight(fighter_a, fighter_b)
        return jsonify({
            'fighter_a': fighter_a,
            'fighter_b': fighter_b,
            'probability_red_wins': prob
        })
    except Exception as e:
        # This will capture ANY exception and return it
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

# Simple health-check endpoint
@app.route('/')
def home():
    return jsonify({'status': 'ok', 'message': 'UFC Prediction API is running'})