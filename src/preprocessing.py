def add_rolling_features(tall, metrics, window=3):
    """Compute rolling averages of last `window` fights for each metric."""
    for col in metrics:
        tall[f'{col}_rolling{window}'] = tall.groupby('fighter')[col].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
    # Also rolling win rate
    tall[f'won_rolling{window}'] = tall.groupby('fighter')['won'].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
    return tall

def add_diff_features(master, feature_pairs):
    """For each pair (R_col, B_col), create a difference column: R_col - B_col."""
    for r_col, b_col in feature_pairs:
        master[f'{r_col}_minus_{b_col}'] = master[r_col] - master[b_col]
    return master