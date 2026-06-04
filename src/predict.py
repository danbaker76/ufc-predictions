import pandas as pd
import numpy as np
import joblib
import sys
from sqlalchemy import create_engine

MODEL_PATH = 'models/ufc_model.pkl'
DB_PATH = 'sqlite:///ufc.db'

# Feature order must match training exactly
FEATURE_COLS = [
    'R_Height', 'R_Weight', 'R_Reach', 'R_Stance', 'R_age',
    'R_SLpM', 'R_Str_Acc', 'R_SApM', 'R_Str_Def',
    'R_TD_Avg', 'R_TD_Acc', 'R_TD_Def', 'R_Sub_Avg',
    'B_Height', 'B_Weight', 'B_Reach', 'B_Stance', 'B_age',
    'B_SLpM', 'B_Str_Acc', 'B_SApM', 'B_Str_Def',
    'B_TD_Avg', 'B_TD_Acc', 'B_TD_Def', 'B_Sub_Avg',
    'R_hist_sig_str_avg', 'R_hist_td_avg', 'R_hist_ctrl_avg',
    'R_hist_sig_str_pct_avg', 'R_hist_td_pct_avg', 'R_hist_won_avg', 'R_hist_fights_before',
    'B_hist_sig_str_avg', 'B_hist_td_avg', 'B_hist_ctrl_avg',
    'B_hist_sig_str_pct_avg', 'B_hist_td_pct_avg', 'B_hist_won_avg', 'B_hist_fights_before'
]

def predict_fight(fighter_a, fighter_b):
    engine = create_engine(DB_PATH)
    model = joblib.load(MODEL_PATH)

    # Fetch latest features for both fighters
    fighters_df = pd.read_sql("SELECT * FROM fighter_current_features", engine)
    fighter_a_row = fighters_df[fighters_df['fighter'] == fighter_a]
    fighter_b_row = fighters_df[fighters_df['fighter'] == fighter_b]

    if fighter_a_row.empty or fighter_b_row.empty:
        raise ValueError("One or both fighters not found in database.")

    # Get first (latest) row for each
    a_feats = fighter_a_row.iloc[0]
    b_feats = fighter_b_row.iloc[0]

    # Build the input row
    X_dict = {}
    for feat in FEATURE_COLS:
        if feat.startswith('R_'):
            key = feat  # the column in the table is exactly like 'R_Height', etc.
            X_dict[feat] = a_feats.get(key, np.nan)
        else:  # B_
            key = feat
            X_dict[feat] = b_feats.get(key, np.nan)

    X = pd.DataFrame([X_dict])

    # Encode stance (same mapping as training)
    stance_map = {'Orthodox': 0, 'Southpaw': 1, 'Switch': 2, 'Open Stance': 3, 'Unknown': 4}
    X['R_Stance'] = X['R_Stance'].fillna('Unknown').map(stance_map).fillna(4)
    X['B_Stance'] = X['B_Stance'].fillna('Unknown').map(stance_map).fillna(4)

    # Fill any remaining NaN with 0 (or median from training, but 0 is okay)
    X = X.fillna(0)

    prob = model.predict_proba(X)[0, 1]
    return prob

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python src/predict.py 'Fighter A' 'Fighter B'")
        sys.exit(1)
    fighter_a = sys.argv[1]
    fighter_b = sys.argv[2]
    try:
        prob = predict_fight(fighter_a, fighter_b)
        print(f"Probability that {fighter_a} (Red corner) wins: {prob:.2%}")
    except Exception as e:
        print(f"Error: {e}")