import pandas as pd
import numpy as np
import os
import re
from src.db_utils import save_to_db

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

# --- Utility parsers (unchanged) ---
def pct_to_float(series):
    return series.str.rstrip('%').astype(float) / 100

def height_to_inches(height_str):
    if pd.isna(height_str): return np.nan
    match = re.match(r"(\d+)'\s*(\d+)\"?", str(height_str))
    if match:
        feet, inches = int(match.group(1)), int(match.group(2))
        return feet * 12 + inches
    return np.nan

def weight_to_float(weight_str):
    if pd.isna(weight_str): return np.nan
    num = re.findall(r"[\d.]+", str(weight_str))
    return float(num[0]) if num else np.nan

def reach_to_float(reach_str):
    if pd.isna(reach_str): return np.nan
    num = re.findall(r"[\d.]+", str(reach_str))
    return float(num[0]) if num else np.nan

# --- Historical feature builder ---
def build_historical_features(fight_df):
    """Create expanding-window stats for each fighter up to (but not including) the current fight."""
    stat_cols = [
        'SIG_STR', 'SIG_STR_pct', 'TOTAL_STR', 'TD', 'TD_pct',
        'SUB_ATT', 'REV', 'CTRL', 'HEAD', 'BODY', 'LEG',
        'DISTANCE', 'CLINCH', 'GROUND'
    ]
    # Red corner data
    red = fight_df[['R_fighter', 'B_fighter', 'Winner', 'date'] + [f'R_{c}' for c in stat_cols]].copy()
    red.columns = ['fighter', 'opponent', 'winner', 'date'] + [c.lower() for c in stat_cols]
    red['corner'] = 'red'
    # Blue corner data
    blue = fight_df[['B_fighter', 'R_fighter', 'Winner', 'date'] + [f'B_{c}' for c in stat_cols]].copy()
    blue.columns = ['fighter', 'opponent', 'winner', 'date'] + [c.lower() for c in stat_cols]
    blue['corner'] = 'blue'
    
    tall = pd.concat([red, blue], ignore_index=True)
    tall['date'] = pd.to_datetime(tall['date'])
    tall['won'] = (tall['winner'] == tall['fighter']).astype(int)
    tall.sort_values(['fighter', 'date'], inplace=True)
    
    # Columns we want to turn into averages
    metrics = ['sig_str', 'td', 'ctrl', 'sig_str_pct', 'td_pct']  # add more as needed
    # Shift to exclude current fight
    for col in metrics:
        tall[col] = tall.groupby('fighter')[col].shift(1)
    tall['won'] = tall.groupby('fighter')['won'].shift(1)
    
    # Expanding means (all history)
    for col in metrics + ['won']:
        tall[f'{col}_avg'] = tall.groupby('fighter')[col].expanding().mean().reset_index(level=0, drop=True)
    
    # Fight count (number of previous fights)
    tall['fights_before'] = tall.groupby('fighter').cumcount()
    
    # Now we have a row for each fighter in each fight with historical stats.
    # We'll merge back onto the original fight_df.
    # Create Red historical features
    red_hist = tall[tall['corner'] == 'red'].add_prefix('R_hist_').rename(columns={
        'R_hist_fighter': 'R_fighter',
        'R_hist_date': 'date'
    })
    keep_cols_red = ['R_fighter', 'date'] + [f'R_hist_{c}_avg' for c in metrics] + ['R_hist_won_avg', 'R_hist_fights_before']
    red_hist = red_hist[keep_cols_red]
    
    # Create Blue historical features
    blue_hist = tall[tall['corner'] == 'blue'].add_prefix('B_hist_').rename(columns={
        'B_hist_fighter': 'B_fighter',
        'B_hist_date': 'date'
    })
    keep_cols_blue = ['B_fighter', 'date'] + [f'B_hist_{c}_avg' for c in metrics] + ['B_hist_won_avg', 'B_hist_fights_before']
    blue_hist = blue_hist[keep_cols_blue]
    
    # Merge historical features with the original fight data
    fight_df = fight_df.merge(red_hist, on=['R_fighter', 'date'], how='left')
    fight_df = fight_df.merge(blue_hist, on=['B_fighter', 'date'], how='left')
    return fight_df

# --- Main feature engineering ---
def build_features():
    # Load raw data
    fighter_df = pd.read_csv(os.path.join(RAW_DIR, 'raw_fighter_details.csv'))
    fight_df   = pd.read_csv(os.path.join(RAW_DIR, 'raw_event_details.csv'), sep=';')
    
    # Clean fighter attributes
    for col in ['Str_Acc', 'Str_Def', 'TD_Acc', 'TD_Def']:
        if col in fighter_df.columns:
            fighter_df[col] = pct_to_float(fighter_df[col])
    fighter_df['Height'] = fighter_df['Height'].apply(height_to_inches)
    fighter_df['Weight'] = fighter_df['Weight'].apply(weight_to_float)
    fighter_df['Reach']  = fighter_df['Reach'].apply(reach_to_float)
    
    # Parse fight date
    fight_df['date'] = pd.to_datetime(fight_df['date'])
    
    # Add historical features
    fight_df = build_historical_features(fight_df)
    
    # Merge fighter attributes for red and blue
    attr_cols = ['fighter_name', 'Height', 'Weight', 'Reach', 'Stance', 'DOB',
                 'SLpM', 'Str_Acc', 'SApM', 'Str_Def',
                 'TD_Avg', 'TD_Acc', 'TD_Def', 'Sub_Avg']
    fighter_attrs = fighter_df[attr_cols].copy()
    
    red_attrs = fighter_attrs.add_prefix('R_').rename(columns={'R_fighter_name': 'R_fighter'})
    base = fight_df.merge(red_attrs, left_on='R_fighter', right_on='R_fighter', how='left')
    
    blue_attrs = fighter_attrs.add_prefix('B_').rename(columns={'B_fighter_name': 'B_fighter'})
    master = base.merge(blue_attrs, left_on='B_fighter', right_on='B_fighter', how='left')
    
    # Create target (1 if Red corner won, 0 if Blue won)
    master['target'] = (master['Winner'] == master['R_fighter']).astype(int)
    # Drop draws / mismatches
    master = master[master['Winner'].isin(master['R_fighter']) | master['Winner'].isin(master['B_fighter'])]
    
    # Compute age at fight time
    master['R_DOB'] = pd.to_datetime(master['R_DOB'], errors='coerce')
    master['B_DOB'] = pd.to_datetime(master['B_DOB'], errors='coerce')
    master['R_age'] = (master['date'] - master['R_DOB']).dt.days / 365.25
    master['B_age'] = (master['date'] - master['B_DOB']).dt.days / 365.25
    
    # Select final feature columns (static + historical)
    static_features = [
        'R_Height', 'R_Weight', 'R_Reach', 'R_Stance', 'R_age',
        'R_SLpM', 'R_Str_Acc', 'R_SApM', 'R_Str_Def',
        'R_TD_Avg', 'R_TD_Acc', 'R_TD_Def', 'R_Sub_Avg',
        'B_Height', 'B_Weight', 'B_Reach', 'B_Stance', 'B_age',
        'B_SLpM', 'B_Str_Acc', 'B_SApM', 'B_Str_Def',
        'B_TD_Avg', 'B_TD_Acc', 'B_TD_Def', 'B_Sub_Avg'
    ]
    hist_features = [
        'R_hist_sig_str_avg', 'R_hist_td_avg', 'R_hist_ctrl_avg',
        'R_hist_sig_str_pct_avg', 'R_hist_td_pct_avg', 'R_hist_won_avg', 'R_hist_fights_before',
        'B_hist_sig_str_avg', 'B_hist_td_avg', 'B_hist_ctrl_avg',
        'B_hist_sig_str_pct_avg', 'B_hist_td_pct_avg', 'B_hist_won_avg', 'B_hist_fights_before'
    ]
    feature_cols = static_features + hist_features + ['date', 'target']
    master = master[feature_cols].dropna(subset=['target'])
    
    # Save processed data
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    master.to_csv(os.path.join(PROCESSED_DIR, 'fight_features.csv'), index=False)
    print(f"Saved {len(master)} rows with {len(feature_cols)-2} features")
    
    # Store in database
    save_to_db(master)
    return master

if __name__ == '__main__':
    build_features()