import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib
import os
from typing import List, Tuple

class SafeTimeSeriesSplit:
    """
    Time‑series cross‑validator that automatically skips training folds
    containing only one class.  The number of splits is determined by the
    data, up to a maximum of `n_splits`.
    """
    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        n_valid = 0
        for train_idx, _ in tscv.split(X):
            if len(np.unique(y.iloc[train_idx])) >= 2:
                n_valid += 1
        return n_valid

    def split(self, X, y, groups=None):
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        for train_idx, test_idx in tscv.split(X):
            if len(np.unique(y.iloc[train_idx])) < 2:
                continue
            yield train_idx, test_idx

def train_model():
    engine = create_engine('sqlite:///ufc.db')
    df = pd.read_sql('fight_features', engine)
    df = df.sort_values('date').reset_index(drop=True)

    target = df['target']
    X = df.drop(columns=['target', 'date'])

    # Encode stances
    for col in ['R_Stance', 'B_Stance']:
        X[col] = X[col].fillna('Unknown')
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Fill missing numeric values with median
    X = X.fillna(X.median(numeric_only=True))

    print(f"Features shape: {X.shape}")
    print(f"Target distribution:\n{target.value_counts(normalize=True)}")

    scale_pos_weight = (target == 0).sum() / (target == 1).sum()

    xgb = XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight
    )

    param_distributions = {
    'n_estimators': [100, 200, 300, 500, 700],
    'max_depth': [3, 4, 5, 6, 8, 10, 12],
    'learning_rate': [0.005, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.1, 0.5, 1, 5, 10],
    'reg_lambda': [0, 1, 5, 10, 20],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0, 0.1, 0.3, 0.5]   # min loss reduction for split
}

    # Use the safe splitter to avoid single‑class folds
    safe_cv = SafeTimeSeriesSplit(n_splits=5)

    search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_distributions,
        n_iter=500,
        scoring='neg_log_loss',
        cv=safe_cv,
        verbose=1,
        random_state=42,
        n_jobs=-1,
        error_score='raise'   # Now we can safely raise because we know folds are valid
    )

    search.fit(X, target)

    print("\nBest parameters:", search.best_params_)
    print(f"Best CV Log Loss: {-search.best_score_:.4f}")

    # Train final model on all data
    best_model = search.best_estimator_
    best_model.fit(X, target)

    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/ufc_model.pkl')
    print("Model saved to models/ufc_model.pkl")

if __name__ == '__main__':
    train_model()