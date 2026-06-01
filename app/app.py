from flask import Flask, request, jsonify
import sys
import os

# Add project root to path so we can import from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.predict import predict_fight

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    fighter_a = data.get('fighter_a')
    fighter_b = data.get('fighter_b')
    if not fighter_a or not fighter_b:
        return jsonify({'error': 'Provide both fighter_a and fighter_b'}), 400
    try:
        prob = predict_fight(fighter_a, fighter_b)
        return jsonify({
            'fighter_a': fighter_a,
            'fighter_b': fighter_b,
            'probability_red_wins': prob
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)