"""
data_loader.py
---------------
Loads the transactions CSV and prepares train/test feature matrices.
Encapsulates the feature engineering shared by training and inference so the
CLI (predict.py) uses the exact same encoding as training (train.py).
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

NUMERIC_FEATURES = [
    "transaction_amount",
    "hour_of_day",
    "geo_ip_mismatch",
    "new_device_fingerprint",
    "behavioral_anomaly_score",
    "txn_velocity_last_1h",
    "distance_from_home_km",
    "account_age_days",
]
CATEGORICAL_FEATURES = ["merchant_category"]
TARGET = "is_fraud"

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_dataset(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Run: python data/generate_dataset.py first."
        )
    return pd.read_csv(path)


def split_dataset(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    X = df[ALL_FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
