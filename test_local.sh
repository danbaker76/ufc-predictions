#!/bin/bash
# test_local.sh – validate the UFC predictor before pushing to Render

set -e

echo "=============================================="
echo " 1. Testing prediction logic serialization"
echo "=============================================="
python -c "
from src.predict import predict_fight
import json

prob = predict_fight('Jon Jones', 'Stipe Miocic')
print(f'Prediction type: {type(prob).__name__}')

try:
    json.dumps({'probability': prob})
    print('✓ JSON serialization works directly')
except (TypeError, OverflowError) as e:
    print(f'✗ JSON serialization failed: {e}')
    print('  → cast to float in predict_fight or api.py')
"
echo ""

echo "=============================================="
echo " 2. Testing local Flask API (if running)"
echo "=============================================="
# Check if anything is listening on port 5000
if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    RESPONSE=$(curl -s -X POST http://localhost:5000/predict \
        -H "Content-Type: application/json" \
        -d '{"fighter_a":"Jon Jones","fighter_b":"Stipe Miocic"}')
    echo "$RESPONSE" | python -m json.tool
    # Additional check for the float32 bug
    if echo "$RESPONSE" | python -c "import sys,json; json.loads(sys.stdin.read())" >/dev/null 2>&1; then
        echo "✓ Local API returns valid JSON"
    else
        echo "✗ Local API response is not valid JSON"
    fi
else
    echo "Local API not running. Start it with:   python api.py"
fi