import pandas as pd
import numpy as np
import os
import re
from sqlalchemy import create_engine

RAW_DIR = "data/raw"

# ---- Parsers (same as preprocessing) ----
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

def parse_fight_stat(value):
    if pd.isna(value): return np.nan
    if isinstance(value, (int, float)): return float(value)
    s = str(value).strip()
    if ' of ' in s: return float(s.split(' of ')[0])
    if '%' in s: return float(s.rstrip('%'))
    try: return float(s)
    except: return np.nan

def build_current_features():
    # Load raw data
    fighter_df = pd.read_csv(os.path.join(RAW_DIR, 'raw_fighter_details.csv'))
    fight_df   = pd.read_csv(os.path.join(RAW_DIR, 'raw_event_details.csv'), sep=';')

    # Clean fighter details
    for col in ['Str_Acc', 'Str_Def', 'TD_Acc', 'TD_Def']:
        if col in fighter_df.columns:
            fighter_df[col] = pct_to_float(fighter_df[col])
    fighter_df['Height'] = fighter_df['Height'].apply(height_to_inches)
    fighter_df['Weight'] = fighter_df['Weight'].apply(weight_to_float)
    fighter_df['Reach']  = fighter_df['Reach'].apply(reach_to_float)

    # Parse fight date
    fight_df['date'] = pd.to_datetime(fight_df['date'])

    # ---- Build tall history table (same logic as preprocessing) ----
    stat_cols = [
        'SIG_STR.', 'SIG_STR_pct', 'TOTAL_STR.', 'TD', 'TD_pct',
        'SUB_ATT', 'REV', 'CTRL', 'HEAD', 'BODY', 'LEG',
        'DISTANCE', 'CLINCH', 'GROUND'
    ]
    # Red corner
    red = fight_df[['R_fighter', 'B_fighter', 'Winner', 'date'] + [f'R_{c}' for c in stat_cols]].copy()
    red.columns = ['fighter', 'opponent', 'winner', 'date'] + [c.lower().rstrip('.') for c in stat_cols]
    red['corner'] = 'red'

    # Blue corner
    blue = fight_df[['B_fighter', 'R_fighter', 'Winner', 'date'] + [f'B_{c}' for c in stat_cols]].copy()
    blue.columns = ['fighter', 'opponent', 'winner', 'date'] + [c.lower().rstrip('.') for c in stat_cols]
    blue['corner'] = 'blue'

    tall = pd.concat([red, blue], ignore_index=True)
    tall['date'] = pd.to_datetime(tall['date'])
    tall['won'] = (tall['winner'] == tall['fighter']).astype(int)

    # Convert stat strings to floats
    for col in tall.columns:
        if col in ['fighter','opponent','winner','date','corner','won']: continue
        tall[col] = tall[col].apply(parse_fight_stat)

    # Sort and shift to exclude current fight
    tall.sort_values(['fighter','date'], inplace=True)
    metric_cols = [c for c in tall.columns if c not in ['fighter','opponent','winner','date','corner','won']]
    for col in metric_cols:
        tall[col] = tall.groupby('fighter')[col].shift(1)
    tall['won'] = tall.groupby('fighter')['won'].shift(1)

    # Compute expanding averages
    for col in metric_cols + ['won']:
        if tall[col].notna().sum() == 0:
            tall[f'{col}_avg'] = np.nan
        else:
            tall[f'{col}_avg'] = tall.groupby('fighter')[col].expanding().mean().reset_index(level=0, drop=True)

    tall['fights_before'] = tall.groupby('fighter').cumcount()

    # ---- Get the most recent row for each fighter ----
    latest_tall = tall.sort_values('date').groupby('fighter').tail(1).copy()
    # Keep only the columns we need: fighter + all the avg columns + fights_before
    keep_cols = ['fighter'] + [f'{c}_avg' for c in metric_cols] + ['won_avg', 'fights_before']
    latest_tall = latest_tall[keep_cols].rename(columns={'won_avg': 'hist_won_avg'})

    # ---- Add static fighter attributes as R_ prefixed columns ----
    fighter_attrs = fighter_df[['fighter_name', 'Height', 'Weight', 'Reach', 'Stance', 'DOB',
                                 'SLpM', 'Str_Acc', 'SApM', 'Str_Def',
                                 'TD_Avg', 'TD_Acc', 'TD_Def', 'Sub_Avg']].copy()
    # Rename to R_ prefix
    rename_map = {
        'Height': 'R_Height', 'Weight': 'R_Weight', 'Reach': 'R_Reach', 'Stance': 'R_Stance',
        'DOB': 'R_DOB', 'SLpM': 'R_SLpM', 'Str_Acc': 'R_Str_Acc', 'SApM': 'R_SApM',
        'Str_Def': 'R_Str_Def', 'TD_Avg': 'R_TD_Avg', 'TD_Acc': 'R_TD_Acc',
        'TD_Def': 'R_TD_Def', 'Sub_Avg': 'R_Sub_Avg'
    }
    fighter_attrs.rename(columns=rename_map, inplace=True)
    fighter_attrs.rename(columns={'fighter_name': 'fighter'}, inplace=True)

    # Compute approximate age (current age from DOB) – we don't have a fight date, so we'll use today
    today = pd.Timestamp.today()
    fighter_attrs['R_DOB'] = pd.to_datetime(fighter_attrs['R_DOB'], errors='coerce')
    fighter_attrs['R_age'] = (today - fighter_attrs['R_DOB']).dt.days / 365.25

    # Merge with history table
    current = pd.merge(latest_tall, fighter_attrs, on='fighter', how='left')

    # Rename history columns to have R_hist_ prefix (matching feature names)
    # The tall history columns currently are like 'sig_str_avg', 'td_avg', etc.
    # We need them to be 'R_hist_sig_str_avg', etc.
    hist_rename = {}
    for col in latest_tall.columns:
        if col == 'fighter' or col.startswith('R_'):
            continue
        hist_rename[col] = 'R_hist_' + col
    current.rename(columns=hist_rename, inplace=True)

    # Also add B_ columns? The prediction script expects both R_ and B_ for a matchup.
    # But our fighter_current_features table should have only R_ prefixed columns (representing the fighter as Red corner).
    # So we are good.

    # Save to database
    engine = create_engine('sqlite:///ufc.db')
    current.to_sql('fighter_current_features', engine, if_exists='replace', index=False)
    print(f"Saved current features for {len(current)} fighters.")

if __name__ == '__main__':
    build_current_features()