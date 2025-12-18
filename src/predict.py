import joblib
import pandas as pd
from pathlib import Path

from src.features import build_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "model_v1.pkl"
ENCODER_PATH = PROJECT_ROOT / "models" / "encoder_v1.pkl"
FEATURES_PATH = PROJECT_ROOT / "models" / "features_v1.pkl"

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
feature_names = joblib.load(FEATURES_PATH)

def predict_one(raw_input: dict) -> dict:
    df = pd.DataFrame([raw_input])
    X, _, _, _ = build_features(
        df,
        encoder=encoder,
        fit_encoder=False
    )


    X = X[feature_names]

    # Predict
    podium_prob = model.predict_proba(X)[0, 1]
    podium_pred = int(podium_prob >= 0.5)

    return {
        "podium_probability": float(podium_prob),
        "podium_prediction": podium_pred
    }