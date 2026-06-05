from flask import Flask, request, jsonify
from flask_cors import CORS          # <-- add this import
import sys, os, traceback

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.predict import predict_fight

app = Flask(__name__)
CORS(app)                            # <-- add this line

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
            'probability_red_wins': float(prob)
        })
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run(debug=True)