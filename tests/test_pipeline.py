"""
test_pipeline.py
-----------------
Lightweight smoke tests for the Fraud Detection AI System.
Run with: pytest -q
"""

import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data"))

from generate_dataset import generate_transactions  # noqa: E402
from data_loader import ALL_FEATURES, TARGET  # noqa: E402


def test_dataset_generation_shape_and_balance():
    df = generate_transactions(n_samples=2000, fraud_ratio=0.05, seed=1)
    assert len(df) == 2000
    assert TARGET in df.columns
    fraud_rate = df[TARGET].mean()
    assert 0.03 < fraud_rate < 0.07  # within noise of the requested 5%


def test_dataset_has_all_expected_features():
    df = generate_transactions(n_samples=500, fraud_ratio=0.1, seed=2)
    for col in ALL_FEATURES:
        assert col in df.columns


def test_trained_model_loads_and_predicts(tmp_path):
    # Build a tiny dataset + train a tiny model to keep the test fast.
    df = generate_transactions(n_samples=3000, fraud_ratio=0.05, seed=3)
    csv_path = tmp_path / "txns.csv"
    df.to_csv(csv_path, index=False)

    model_path = tmp_path / "model.joblib"
    result = subprocess.run(
        [sys.executable, str(ROOT / "src" / "train.py"),
         "--data", str(csv_path),
         "--model_out", str(model_path),
         "--metrics_out", str(tmp_path / "metrics.json")],
        capture_output=True, text=True, cwd=str(ROOT / "src"),
    )
    assert result.returncode == 0, result.stderr
    assert model_path.exists()

    pipeline = joblib.load(model_path)
    sample = pd.DataFrame([{
        "transaction_amount": 50000.0, "merchant_category": "crypto_exchange",
        "hour_of_day": 3, "geo_ip_mismatch": 1, "new_device_fingerprint": 1,
        "behavioral_anomaly_score": 0.9, "txn_velocity_last_1h": 9,
        "distance_from_home_km": 1500.0, "account_age_days": 30,
    }])[ALL_FEATURES]
    proba = pipeline.predict_proba(sample)[0][1]
    assert 0.0 <= proba <= 1.0
