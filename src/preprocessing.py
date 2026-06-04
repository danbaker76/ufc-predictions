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
    
    # ── Build current (latest) fighter features for future predictions ──
    # Do this while master still has fighter names
    master_sorted = master.sort_values('date')
    
    # Red corner latest
    latest_red = (
        master_sorted.groupby('R_fighter')
        .tail(1)[['R_fighter'] + [c for c in master.columns if c.startswith('R_')]]
        .copy()
        .rename(columns={'R_fighter': 'fighter'})
    )
    # Blue corner latest
    latest_blue = (
        master_sorted.groupby('B_fighter')
        .tail(1)[['B_fighter'] + [c for c in master.columns if c.startswith('B_')]]
        .copy()
        .rename(columns={'B_fighter': 'fighter'})
    )
    # Combine and keep the most recent per fighter
    # Drop duplicate column names like 'date' that appear in both
    common_cols = set(latest_red.columns) & set(latest_blue.columns) - {'fighter'}
    for col in common_cols:
        # We'll keep the version from latest_red, drop from latest_blue
        latest_blue.drop(columns=col, inplace=True, errors='ignore')
    latest = pd.concat([latest_red, latest_blue], ignore_index=True)
    # Now we have a single 'date' column from latest_red
    latest = latest.sort_values('date').groupby('fighter').tail(1)
    latest = latest.drop(columns='date', errors='ignore')
    
    save_to_db(latest, 'fighter_current_features')
    print(f"Saved current features for {len(latest)} fighters.")
    
    # Select final feature columns for training (no fighter names)
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
    
    # Store training data in database
    save_to_db(master)
    return master