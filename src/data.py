import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def load_master_dataframe(
        path=PROJECT_ROOT / "data" / "master_features_v1.parquet"

        ):
    df = pd.read_parquet(path)
    print(f"Loaded dataframe with shape {df.shape}")
    
    

    return df
load_master_dataframe("data/master_features_v1.parquet")