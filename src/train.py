import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib
import os

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
    
    # Use scale_pos_weight for imbalance
    scale_pos_weight = (target == 0).sum() / (target == 1).sum()
    
    # Hyperparameter grid (you can expand)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8],
        'colsample_bytree': [0.8]
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    best_score = np.inf
    best_params = None
    
    # Simple manual grid search over time splits
    for n_est in param_grid['n_estimators']:
        for md in param_grid['max_depth']:
            for lr in param_grid['learning_rate']:
                model = XGBClassifier(
                    n_estimators=n_est,
                    max_depth=md,
                    learning_rate=lr,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    eval_metric='logloss'
                )
                fold_scores = []
                for train_idx, test_idx in tscv.split(X):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]
                    if len(np.unique(y_train)) < 2:
                        continue
                    model.fit(X_train, y_train)
                    probs = model.predict_proba(X_test)[:, 1]
                    if probs.shape[0] == 0 or len(np.unique(y_test)) < 2:
                        continue
                    ll = log_loss(y_test, probs)
                    fold_scores.append(ll)
                if fold_scores:
                    avg_ll = np.mean(fold_scores)
                    print(f"Params: n={n_est}, d={md}, lr={lr} -> Avg LogLoss: {avg_ll:.4f}")
                    if avg_ll < best_score:
                        best_score = avg_ll
                        best_params = {'n_estimators': n_est, 'max_depth': md, 'learning_rate': lr}
    
    print(f"\nBest params: {best_params}, Best LogLoss: {best_score:.4f}")
    
    # Train final model with best params on all data
    final_model = XGBClassifier(
        **best_params,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    final_model.fit(X, target)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(final_model, 'models/ufc_model.pkl')
    print("New model saved to models/ufc_model.pkl")

if __name__ == '__main__':
    train_model()