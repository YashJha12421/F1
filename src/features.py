import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


NUMERIC_FEATURES = [
    'grid',
    'year',
    'round',
    'form_last3',
    'age',
    'team_strength'
]

CATEGORICAL_FEATURES = [
    'driverId',
    'constructorId',
    'circuitId'
]

TARGET_COL = 'podium'

def make_target(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].astype(int)
    return df

def build_features(
    df: pd.DataFrame,
    encoder: OrdinalEncoder | None = None,
    fit_encoder: bool = False
):
    df = df.copy()

    # Check if target exists (training vs inference)
    has_target = TARGET_COL in df.columns

    if has_target:
        df = make_target(df)

    # Split features
    X_num = df[NUMERIC_FEATURES].copy()
    X_cat = df[CATEGORICAL_FEATURES].copy()

    # Encoder setup
    if encoder is None:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )

    if fit_encoder:
        X_cat_encoded = encoder.fit_transform(X_cat)
    else:
        X_cat_encoded = encoder.transform(X_cat)

    X_cat_encoded = pd.DataFrame(
        X_cat_encoded,
        columns=[f"{c}_enc" for c in CATEGORICAL_FEATURES],
        index=df.index
    )

    # Final feature matrix
    X = pd.concat([X_num, X_cat_encoded], axis=1)
    feature_names = list(X.columns)

    # Target only if available
    y = df[TARGET_COL] if has_target else None

    return X, y, encoder, feature_names
