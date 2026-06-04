# src/features.py
import pandas as pd
import numpy as np

def create_historical_features(fight_df, fighter_df):
    """
    fight_df: raw fight data (with all stats columns)
    fighter_df: raw fighter details
    Returns a DataFrame with historical features for each fight (red & blue).
    """
    # Select and rename fight stat columns for both corners
    stat_cols = [
        'SIG_STR', 'SIG_STR_pct', 'TOTAL_STR', 'TD', 'TD_pct',
        'SUB_ATT', 'REV', 'CTRL', 'HEAD', 'BODY', 'LEG',
        'DISTANCE', 'CLINCH', 'GROUND'
    ]
    # Build a tall table: one row per fighter per fight
    red = fight_df[['R_fighter', 'B_fighter', 'Winner', 'date'] + [f'R_{c}' for c in stat_cols]].copy()
    red.columns = ['fighter', 'opponent', 'winner', 'date'] + [c.lower() for c in stat_cols]
    red['won'] = (red['winner'] == red['fighter']).astype(int)
    
    blue = fight_df[['B_fighter', 'R_fighter', 'Winner', 'date'] + [f'B_{c}' for c in stat_cols]].copy()
    blue.columns = ['fighter', 'opponent', 'winner', 'date'] + [c.lower() for c in stat_cols]
    blue['won'] = (blue['winner'] == blue['fighter']).astype(int)
    
    tall = pd.concat([red, blue], ignore_index=True)
    tall['date'] = pd.to_datetime(tall['date'])
    tall.sort_values(['fighter', 'date'], inplace=True)
    
    # Compute expanding window stats for each fighter (shift to exclude current)
    # We'll aggregate a few key stats: sig_str_landed, takedown_landed, control time, win rate
    tall['sig_str_landed'] = tall['sig_str']
    tall['td_landed'] = tall['td']
    tall['ctrl'] = tall['ctrl']
    
    # Shift so current fight is not included
    for col in ['sig_str_landed', 'td_landed', 'ctrl', 'won']:
        tall[col] = tall.groupby('fighter')[col].shift(1)
    
    # Expanding mean of last 3 fights (rolling window)
    # For simplicity, we'll use expanding().mean() and then take last 3 if we want, but we can just use expanding mean
    # Better: use a custom rolling(3, min_periods=1) but that's not expanding. We'll create both.
    
    # Expanding mean (all prior fights)
    tall['avg_sig_str_landed_exp'] = tall.groupby('fighter')['sig_str_landed'].expanding().mean().reset_index(level=0, drop=True)
    tall['avg_td_landed_exp'] = tall.groupby('fighter')['td_landed'].expanding().mean().reset_index(level=0, drop=True)
    tall['win_rate_exp'] = tall.groupby('fighter')['won'].expanding().mean().reset_index(level=0, drop=True)
    
    # Rolling last 3 fights (if we want recency)
    tall['avg_sig_str_landed_3'] = tall.groupby('fighter')['sig_str_landed'].transform(
        lambda x: x.rolling(3, min_periods=1).mean())
    tall['avg_td_landed_3'] = tall.groupby('fighter')['td_landed'].transform(
        lambda x: x.rolling(3, min_periods=1).mean())
    tall['win_rate_3'] = tall.groupby('fighter')['won'].transform(
        lambda x: x.rolling(3, min_periods=1).mean())
    
    # Create a "fight count" feature
    tall['fights_before'] = tall.groupby('fighter').cumcount()  # 0-based: number of fights prior
    
    # Now merge back to the original fight rows (we need to map by fighter & date)
    # For each fight (identified by R_fighter, B_fighter, date), we need the historical stats of both fighters
    # We'll create two lookup DataFrames (for red and blue) and join.
    red_features = tall.add_prefix('R_hist_').rename(columns={'R_hist_fighter': 'R_fighter', 'R_hist_date': 'date'})
    blue_features = tall.add_prefix('B_hist_').rename(columns={'B_hist_fighter': 'B_fighter', 'B_hist_date': 'date'})
    
    # The tall table contains duplicate dates if a fighter fought multiple times on the same day? Unlikely but possible.
    # We'll merge on fighter and exact date (the current fight's date matches the tall row's date because we didn't remove it).
    # We need to merge on the current fight row, but our tall table already contains the current fight row with shifted stats.
    # So for a given fight, the row in tall where fighter==R_fighter and date==fight.date gives the right historical stats.
    # We'll just merge directly with fight_df on R_fighter and date, then similarly for blue.
    # For simplicity, we can merge the tall table with fight_df twice.
    
    return tall  # for now, just return the tall table; we'll integrate it in preprocessing.