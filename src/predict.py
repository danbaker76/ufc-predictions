import pandas as pd
import numpy as np
import re
import joblib
import sys

def height_to_inches(height_str):
    if pd.isna(height_str):
        return np.nan
    match = re.match(r"(\d+)'\s*(\d+)\"?", str(height_str))
    if match:
        feet, inches = int(match.group(1)), int(match.group(2))
        return feet * 12 + inches
    return np.nan

def weight_to_float(weight_str):
    if pd.isna(weight_str):
        return np.nan
    num = re.findall(r"[\d.]+", str(weight_str))
    return float(num[0]) if num else np.nan

def reach_to_float(reach_str):
    if pd.isna(reach_str):
        return np.nan
    num = re.findall(r"[\d.]+", str(reach_str))
    return float(num[0]) if num else np.nan

def pct_str_to_float(s):
    if pd.isna(s):
        return np.nan
    try:
        return float(s.strip('%')) / 100
    except:
        return np.nan

def load_fighter_stats(name, fighter_df):
    """Return a dict with keys matching the feature suffixes (e.g. 'Height', 'SLpM', 'age')."""
    row = fighter_df[fighter_df['fighter_name'] == name]
    if row.empty:
        raise ValueError(f"Fighter '{name}' not found in database.")
    row = row.iloc[0]
    stats = {
        'Height': height_to_inches(row.get('Height')),
        'Weight': weight_to_float(row.get('Weight')),
        'Reach': reach_to_float(row.get('Reach')),
        'Stance': row.get('Stance') if pd.notna(row.get('Stance')) else 'Unknown',
        'age': np.nan,   # age unknown, will be filled later
        'SLpM': float(row['SLpM']) if pd.notna(row.get('SLpM')) else np.nan,
        'Str_Acc': pct_str_to_float(row.get('Str_Acc')),
        'SApM': float(row['SApM']) if pd.notna(row.get('SApM')) else np.nan,
        'Str_Def': pct_str_to_float(row.get('Str_Def')),
        'TD_Avg': float(row['TD_Avg']) if pd.notna(row.get('TD_Avg')) else np.nan,
        'TD_Acc': pct_str_to_float(row.get('TD_Acc')),
        'TD_Def': pct_str_to_float(row.get('TD_Def')),
        'Sub_Avg': float(row['Sub_Avg']) if pd.notna(row.get('Sub_Avg')) else np.nan,
    }
    return stats

def predict_fight(fighter_a, fighter_b):
    fighter_df = pd.read_csv('data/raw/raw_fighter_details.csv')
    model = joblib.load('models/ufc_model.pkl')

    red = load_fighter_stats(fighter_a, fighter_df)
    blue = load_fighter_stats(fighter_b, fighter_df)

    # Feature order exactly as used during training
    feature_names = [
        'R_Height', 'R_Weight', 'R_Reach', 'R_Stance', 'R_age',
        'R_SLpM', 'R_Str_Acc', 'R_SApM', 'R_Str_Def',
        'R_TD_Avg', 'R_TD_Acc', 'R_TD_Def', 'R_Sub_Avg',
        'B_Height', 'B_Weight', 'B_Reach', 'B_Stance', 'B_age',
        'B_SLpM', 'B_Str_Acc', 'B_SApM', 'B_Str_Def',
        'B_TD_Avg', 'B_TD_Acc', 'B_TD_Def', 'B_Sub_Avg'
    ]

    X_dict = {}
    for feat in feature_names:
        # suffix is the key inside red/blue dict (e.g. 'Height', 'SLpM', 'age')
        suffix = feat[2:]
        stats = red if feat.startswith('R_') else blue
        X_dict[feat] = stats[suffix]

    X = pd.DataFrame([X_dict])

    # Encode stance categories (same mapping as used in training)
    stance_map = {'Orthodox': 0, 'Southpaw': 1, 'Switch': 2, 'Open Stance': 3, 'Unknown': 4}
    X['R_Stance'] = X['R_Stance'].fillna('Unknown').map(stance_map).fillna(4)
    X['B_Stance'] = X['B_Stance'].fillna('Unknown').map(stance_map).fillna(4)

    # Fill remaining NaN with 0 (or better with training medians, but 0 is okay for now)
    X = X.fillna(0)

    prob = model.predict_proba(X)[0, 1]
    return prob

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python src/predict.py 'Fighter A' 'Fighter B'")
        sys.exit(1)
    fighter_a = sys.argv[1]
    fighter_b = sys.argv[2]
    prob = predict_fight(fighter_a, fighter_b)
    print(f"Probability that {fighter_a} (Red corner) wins: {prob:.2%}")