import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

from src.data import load_master_dataframe
from src.features import build_features
import joblib



def train():
    df = load_master_dataframe()

    X, y, encoder, feature_names=build_features(df,    fit_encoder=True)
    train_mask = df['year'] <= 2015
    val_mask   = df['year'].isin(range(2016,2021))

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val     = X[val_mask], y[val_mask]
    


    model = XGBClassifier(
        max_depth=4,
        n_estimators=100,
        learning_rate=0.05,
        random_state=42
    )

    model.fit(X_train, y_train)

    
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]

    auc = roc_auc_score(y_val, y_val_proba)
    acc = accuracy_score(y_val, y_val_pred)

    print(f"Validation AUC: {auc:.4f}")
    print(f"Validation Accuracy: {acc:.4f}")

    joblib.dump(model, "models/model_v1.pkl")
    joblib.dump(encoder, "models/encoder_v1.pkl")
    joblib.dump(feature_names, "models/features_v1.pkl")

    print("Model and artifacts saved.")    

if __name__ == "__main__":
    train()