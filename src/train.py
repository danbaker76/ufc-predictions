import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import warnings

def train_model():
    # ---- Load data ----
    engine = create_engine('sqlite:///ufc.db')
    df = pd.read_sql('fight_features', engine)
    df = df.sort_values('date').reset_index(drop=True)

    target = df['target']
    X = df.drop(columns=['target', 'date'])

    # ---- Encode categoricals ----
    for col in ['R_Stance', 'B_Stance']:
        X[col] = X[col].fillna('Unknown')
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # ---- Fill missing numeric values ----
    X = X.fillna(X.median(numeric_only=True))

    print(f"Features shape: {X.shape}")
    print(f"Target distribution:\n{target.value_counts(normalize=True)}")

    # ---- Time series cross-validation with class-check ----
    tscv = TimeSeriesSplit(n_splits=5)
    model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, class_weight='balanced')
    fold_scores = []
    valid_folds = 0

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]

        # Skip folds where training data has only one class
        if len(np.unique(y_train)) < 2:
            print(f"Fold {fold+1} skipped – only one class in training set.")
            continue

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)

        # Handle case where test set has only one class (probs may have 1 column)
        if probs.shape[1] == 1:
            # If only one class present in test, we can't compute log-loss meaningfully
            acc = accuracy_score(y_test, preds)
            print(f"Fold {fold+1} – Accuracy: {acc:.3f} (only one class in test, no log-loss)")
            fold_scores.append(acc)
        else:
            acc = accuracy_score(y_test, preds)
            ll = log_loss(y_test, probs)
            print(f"Fold {fold+1} – Accuracy: {acc:.3f}, Log Loss: {ll:.3f}")
            fold_scores.append(acc)  # we'll average acc only for summary

        valid_folds += 1

    if valid_folds == 0:
        print("No valid folds with both classes. Cannot evaluate.")
    else:
        avg_acc = np.mean(fold_scores)
        print(f"\nAverage Accuracy over {valid_folds} valid folds: {avg_acc:.3f}")

    # ---- Final training on all data ----
    model.fit(X, target)
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/ufc_model.pkl')
    print("Model saved to models/ufc_model.pkl")

if __name__ == '__main__':
    train_model()