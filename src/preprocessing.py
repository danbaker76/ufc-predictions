import pandas as pd
import numpy as np
import os
import re
from src.db_utils import save_to_db

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

def pct_to_float(series):
    return series.str.rstrip('%').astype(float) / 100

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

def build_features():
    # ---- Load raw CSVs ----
    fighter_df = pd.read_csv(os.path.join(RAW_DIR, 'raw_fighter_details.csv'))
    fight_df   = pd.read_csv(os.path.join(RAW_DIR, 'raw_event_details.csv'), sep=';')

    # ---- Clean fighter percentages (note TD_Acc, TD_Def, not Td) ----
    for col in ['Str_Acc', 'Str_Def', 'TD_Acc', 'TD_Def']:
        if col in fighter_df.columns:
            fighter_df[col] = pct_to_float(fighter_df[col])

    # ---- Parse numeric from Height, Weight, Reach ----
    fighter_df['Height'] = fighter_df['Height'].apply(height_to_inches)
    fighter_df['Weight'] = fighter_df['Weight'].apply(weight_to_float)
    fighter_df['Reach']  = fighter_df['Reach'].apply(reach_to_float)

    # ---- Parse fight date ----
    fight_df['date'] = pd.to_datetime(fight_df['date'])

    # ---- Fighter attributes to attach ----
    attr_cols = ['fighter_name', 'Height', 'Weight', 'Reach', 'Stance', 'DOB',
                 'SLpM', 'Str_Acc', 'SApM', 'Str_Def',
                 'TD_Avg', 'TD_Acc', 'TD_Def', 'Sub_Avg']
    fighter_attrs = fighter_df[attr_cols].copy()

    # ---- Merge red attributes ----
    red_attrs = fighter_attrs.add_prefix('R_').rename(columns={'R_fighter_name': 'R_fighter'})
    base = fight_df.merge(red_attrs, left_on='R_fighter', right_on='R_fighter', how='left')

    # ---- Merge blue attributes ----
    blue_attrs = fighter_attrs.add_prefix('B_').rename(columns={'B_fighter_name': 'B_fighter'})
    master = base.merge(blue_attrs, left_on='B_fighter', right_on='B_fighter', how='left')

    # ---- Create target: 1 if Red corner won, 0 if Blue corner won, drop others ----
    master['target'] = np.where(master['Winner'] == master['R_fighter'], 1,
                         np.where(master['Winner'] == master['B_fighter'], 0, np.nan))
    # Drop rows where target is NaN (draws, no contests, mismatches)
    master = master.dropna(subset=['target']).copy()
    master['target'] = master['target'].astype(int)

    # ---- Compute age at fight time ----
    master['R_DOB'] = pd.to_datetime(master['R_DOB'], errors='coerce')
    master['B_DOB'] = pd.to_datetime(master['B_DOB'], errors='coerce')
    master['R_age'] = (master['date'] - master['R_DOB']).dt.days / 365.25
    master['B_age'] = (master['date'] - master['B_DOB']).dt.days / 365.25

    # ---- Final feature set ----
    feature_cols = [
        'R_Height', 'R_Weight', 'R_Reach', 'R_Stance', 'R_age',
        'R_SLpM', 'R_Str_Acc', 'R_SApM', 'R_Str_Def',
        'R_TD_Avg', 'R_TD_Acc', 'R_TD_Def', 'R_Sub_Avg',
        'B_Height', 'B_Weight', 'B_Reach', 'B_Stance', 'B_age',
        'B_SLpM', 'B_Str_Acc', 'B_SApM', 'B_Str_Def',
        'B_TD_Avg', 'B_TD_Acc', 'B_TD_Def', 'B_Sub_Avg',
        'date',
        'target'
    ]
    master = master[feature_cols]

    # ---- Save ----
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    master.to_csv(os.path.join(PROCESSED_DIR, 'fight_features.csv'), index=False)
    print(f"Saved {len(master)} rows to fight_features.csv")

    save_to_db(master)

    return master

if __name__ == '__main__':
    build_features()